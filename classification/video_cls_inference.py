"""
Video Classification Inference for AddaxAI - Scalable Architecture

Handles classification of animals detected in video files by:
1. Extracting frames that contain detections (using MegaDetector video utils)
2. Cropping animal bounding boxes from those frames  
3. Calling model-specific subprocess for classification inference
4. Updating the JSON with classification results

This script runs in the base environment (has OpenCV + MegaDetector) and calls
model inference subprocesses in their specific environments.

Created by Claude for AddaxAI scalable video support
"""

import argparse
import os
import sys
import json
import subprocess
import tempfile
from tqdm import tqdm

# Add MegaDetector to path for video utilities
sys.path.append("/Users/peter/Desktop/MegaDetector")
from megadetector.detection.video_utils import run_callback_on_frames

# Add AddaxAI to path
sys.path.append(os.environ.get('PYTHONPATH', ''))
from utils.config import VIDEO_EXTENSIONS, ADDAXAI_ROOT


def is_video_file(filename):
    """Check if a file is a video based on its extension."""
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def call_model_inference(model_path, model_type, model_env, image_path, bbox_coords=None):
    """
    Call the model inference subprocess to classify a single image crop.
    
    Args:
        model_path (str): Path to the model file
        model_type (str): Type of model (directory name)
        model_env (str): Environment name (e.g., 'pytorch', 'tensorflow-v1')
        image_path (str): Path to the image file
        bbox_coords (list, optional): Normalized bounding box [x,y,w,h]
    
    Returns:
        list: Classification results as [["species_name", confidence], ...]
    """
    
    # Build the command
    python_executable = f"{ADDAXAI_ROOT}/envs/env-{model_env}/bin/python"
    inference_script = f"{ADDAXAI_ROOT}/classification/model_inference_wrapper.py"
    
    command = [
        python_executable,
        inference_script,
        '--model-path', model_path,
        '--model-type', model_type,
        '--image-path', image_path
    ]
    
    # Add bbox if provided
    if bbox_coords:
        bbox_str = ','.join(map(str, bbox_coords))
        command.extend(['--bbox', bbox_str])
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = ADDAXAI_ROOT
    
    try:
        # Run the subprocess
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout per inference
            env=env,
            cwd=ADDAXAI_ROOT
        )
        
        if result.returncode == 0:
            # Parse JSON output
            output = json.loads(result.stdout.strip())
            if 'error' in output:
                print(f"Model inference error: {output['error']}")
                return []
            return output.get('classifications', [])
        else:
            print(f"Model inference failed: {result.stderr}")
            return []
            
    except subprocess.TimeoutExpired:
        print(f"Model inference timed out for {image_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Failed to parse model output: {e}")
        print(f"Raw output: {result.stdout}")
        return []
    except Exception as e:
        print(f"Error calling model inference: {str(e)}")
        return []


def create_video_frame_callback(model_path, model_type, model_env, detections_by_frame, 
                                label_map, classification_data, temp_dir):
    """
    Create callback function to process video frames and classify detections.
    
    Args:
        model_path (str): Path to classification model
        model_type (str): Type of classification model  
        model_env (str): Environment for the model
        detections_by_frame: Dict mapping frame numbers to detection lists
        label_map: Detection category label mapping
        classification_data: Dict to store classification results and categories
        temp_dir: Temporary directory for frame images
    
    Returns:
        Callback function for video frame processing
    """
    def frame_callback(image_np, frame_filename):
        # Extract frame number from filename (e.g., "frame000024.jpg" -> 24)
        frame_number = int(frame_filename.replace("frame", "").replace(".jpg", ""))
        
        # Get detections for this frame
        frame_detections = detections_by_frame.get(frame_number, [])
        
        # Save the frame as a temporary image
        from PIL import Image
        frame_image = Image.fromarray(image_np)
        frame_path = os.path.join(temp_dir, f"frame_{frame_number:06d}.jpg")
        frame_image.save(frame_path)
        
        # Process each animal detection in this frame
        for detection in frame_detections:
            category_id = detection['category']
            category = label_map[category_id]
            
            # Only classify animal detections
            if category == 'animal' and detection.get('conf', 0) >= 0.1:  # confidence threshold
                bbox = detection['bbox']
                
                # Call model inference subprocess
                name_classifications = call_model_inference(
                    model_path, model_type, model_env, frame_path, bbox
                )
                
                # Process classification results
                if name_classifications:
                    idx_classifications = []
                    for name, confidence in name_classifications:
                        # Add to classification categories if not present
                        if name not in classification_data['inverted_label_map']:
                            # Find next available ID
                            max_id = max([int(v) for v in classification_data['inverted_label_map'].values()] + [0])
                            classification_data['inverted_label_map'][name] = str(max_id + 1)
                        
                        # Convert to indexed format
                        idx_classifications.append([
                            classification_data['inverted_label_map'][name], 
                            round(float(confidence), 5)
                        ])
                    
                    # Sort by confidence and keep only top result
                    idx_classifications = sorted(idx_classifications, key=lambda x: x[1], reverse=True)
                    if idx_classifications:
                        detection['classifications'] = [idx_classifications[0]]
        
        # Clean up temporary frame file
        if os.path.exists(frame_path):
            os.remove(frame_path)
        
        return frame_detections
    
    return frame_callback


def classify_video_detections(json_path, model_path, model_type, model_env):
    """
    Classify animal detections in video files using subprocess-based model inference.
    
    Args:
        json_path (str): Path to JSON file with video detection results
        model_path (str): Path to classification model file
        model_type (str): Type/architecture of the classification model
        model_env (str): Environment name for the model
    """
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loading classification model: {model_type}")
    
    # Get label mapping for detections
    label_map = data.get('detection_categories', {'1': 'animal', '2': 'person', '3': 'vehicle'})
    
    # Initialize classification data
    if 'classification_categories' not in data:
        data['classification_categories'] = {}
    
    classification_data = {
        'inverted_label_map': {v: k for k, v in data['classification_categories'].items()}
    }
    
    # Get directory containing videos
    video_dir = os.path.dirname(json_path)
    
    # Create temporary directory for frame processing
    with tempfile.TemporaryDirectory(prefix='addaxai_video_frames_') as temp_dir:
        
        # Process each video
        total_detections = 0
        for image_entry in data['images']:
            if is_video_file(image_entry['file']) and 'detections' in image_entry:
                # Count animal detections for progress tracking
                animal_detections = [d for d in image_entry['detections'] 
                                   if label_map.get(d['category'], '') == 'animal' and d.get('conf', 0) >= 0.1]
                total_detections += len(animal_detections)
        
        if total_detections == 0:
            print("No animal detections found in videos.")
            return
        
        print(f"Processing {total_detections} animal detections in videos...")
        
        # Process each video with progress tracking
        with tqdm(total=total_detections, desc="Classifying") as pbar:
            for image_entry in data['images']:
                file_path = image_entry['file']
                
                if not is_video_file(file_path):
                    continue
                
                video_path = os.path.join(video_dir, file_path)
                
                if not os.path.exists(video_path):
                    print(f"Warning: Video file {file_path} not found")
                    continue
                
                if 'detections' not in image_entry:
                    continue
                
                # Group detections by frame number
                detections_by_frame = {}
                animal_count = 0
                
                for detection in image_entry['detections']:
                    if label_map.get(detection['category'], '') == 'animal' and detection.get('conf', 0) >= 0.1:
                        frame_num = detection.get('frame_number', 0)
                        if frame_num not in detections_by_frame:
                            detections_by_frame[frame_num] = []
                        detections_by_frame[frame_num].append(detection)
                        animal_count += 1
                
                if not detections_by_frame:
                    continue
                
                # Get frames to process
                frames_to_process = list(detections_by_frame.keys())
                
                # Create callback for this video
                callback = create_video_frame_callback(
                    model_path, model_type, model_env, detections_by_frame,
                    label_map, classification_data, temp_dir
                )
                
                # Process video frames
                try:
                    run_callback_on_frames(
                        input_video_file=video_path,
                        frame_callback=callback,
                        frames_to_process=frames_to_process,
                        verbose=False
                    )
                    
                    # Update progress
                    pbar.update(animal_count)
                    
                except Exception as e:
                    print(f"Error processing video {file_path}: {str(e)}")
                    pbar.update(animal_count)  # Still update progress
    
    # Update classification categories in data
    data['classification_categories'] = {v: k for k, v in classification_data['inverted_label_map'].items()}
    
    # Write updated JSON back to file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=1)
    
    print("Video classification completed successfully!")


def main():
    """Main function for video classification script."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Video classification inference for AddaxAI')
    parser.add_argument('--model-path', required=True, help='Path to classification model file')
    parser.add_argument('--model-type', required=True, help='Type of classification model')
    parser.add_argument('--model-env', required=True, help='Environment name for the model')
    parser.add_argument('--json-path', required=True, help='Path to JSON file with video detection results')
    parser.add_argument('--country', default=None, help='Country code for geofencing')
    parser.add_argument('--state', default=None, help='State code for geofencing')
    
    args = parser.parse_args()
    
    try:
        classify_video_detections(args.json_path, args.model_path, args.model_type, args.model_env)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()