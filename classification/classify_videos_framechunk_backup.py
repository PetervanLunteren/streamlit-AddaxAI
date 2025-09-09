"""
Simple Video Classification for AddaxAI - Chunked Processing

Processes videos in chunks to balance memory usage and efficiency:
1. Extract frames in chunks (e.g., 1000 frames at a time)
2. Run classification model on each chunk 
3. Clean up temporary frames after each chunk

No more complex model servers - just direct model loading and processing.

Created by Claude for AddaxAI simplified architecture
"""

import argparse
import os
import sys
import json
import tempfile
import shutil
import importlib.util
from tqdm import tqdm
from pathlib import Path

# Add MegaDetector to path for video utilities
sys.path.append("/Users/peter/Desktop/MegaDetector")
from megadetector.detection.video_utils import run_callback_on_frames

# Add AddaxAI to path
sys.path.append(os.environ.get('PYTHONPATH', ''))
from utils.config import VIDEO_EXTENSIONS, ADDAXAI_ROOT


def is_video_file(filename):
    """Check if a file is a video based on its extension."""
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def load_classification_model(model_path, model_type, model_env):
    """
    Load a classification model directly into memory.
    
    Args:
        model_path (str): Path to model file
        model_type (str): Model type (directory name)  
        model_env (str): Environment name
        
    Returns:
        tuple: (get_crop_function, get_classification_function)
    """
    print(f"Loading classification model: {model_type}")
    
    # Get the path to the model-specific script
    model_script_path = os.path.join(
        ADDAXAI_ROOT, "classification", "model_types", model_type, "classify_detections.py"
    )
    
    if not os.path.exists(model_script_path):
        raise ValueError(f"Model script not found: {model_script_path}")
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("model_module", model_script_path)
    model_module = importlib.util.module_from_spec(spec)
    
    # Set up the model arguments
    original_argv = sys.argv.copy()
    sys.argv = ['classify_detections.py', '--model-path', model_path, '--json-path', '/dev/null']
    
    try:
        # Execute the module to define functions and classes
        spec.loader.exec_module(model_module)
        
        # Set the model path in the module
        model_module.cls_model_fpath = model_path
        model_module.json_path = '/dev/null'
        
        # Load the model (trigger lazy loading)
        if hasattr(model_module, 'load_model'):
            model_module.load_model()
        
        # Get the required functions
        get_crop = getattr(model_module, 'get_crop', None)
        get_classification = getattr(model_module, 'get_classification', None)
        
        if get_crop is None or get_classification is None:
            raise ValueError(f"Model script must define 'get_crop' and 'get_classification' functions")
        
        print(f"Model loaded successfully: {model_type}")
        return get_crop, get_classification
        
    except Exception as e:
        raise ValueError(f"Error loading model from {model_script_path}: {str(e)}")
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def process_video_chunk(video_path, frames_to_process, get_crop, get_classification, 
                       detections_by_frame, label_map, classification_data):
    """
    Process a chunk of video frames for classification.
    
    Args:
        video_path (str): Path to video file
        frames_to_process (list): List of frame numbers to process
        get_crop, get_classification: Model functions
        detections_by_frame (dict): Frame detections mapping
        label_map (dict): Detection category mapping  
        classification_data (dict): Classification data structure
        
    Returns:
        int: Number of animals processed
    """
    animals_processed = 0
    
    def frame_callback(image_np, frame_filename):
        nonlocal animals_processed
        
        # Extract frame number from filename
        frame_number = int(frame_filename.replace("frame", "").replace(".jpg", ""))
        
        # Get detections for this frame
        frame_detections = detections_by_frame.get(frame_number, [])
        
        # Process each animal detection
        for detection in frame_detections:
            category_id = detection['category']
            category = label_map[category_id]
            
            # Only classify animal detections above threshold
            if category == 'animal' and detection.get('conf', 0) >= 0.1:
                bbox = detection['bbox']
                
                # Convert numpy array to PIL Image
                from PIL import Image
                if image_np.dtype != 'uint8':
                    image_np = (image_np * 255).astype('uint8')
                image_pil = Image.fromarray(image_np)
                
                # Get crop using model-specific function
                try:
                    crop = get_crop(image_pil, bbox)
                    if crop is None:
                        continue  # Invalid crop
                    
                    # Classify the crop
                    classifications = get_classification(crop)
                    
                    if classifications:
                        # Convert to indexed format
                        idx_classifications = []
                        for name, confidence in classifications:
                            # Add to classification categories if not present
                            if name not in classification_data['inverted_label_map']:
                                max_id = max([int(v) for v in classification_data['inverted_label_map'].values()] + [0])
                                classification_data['inverted_label_map'][name] = str(max_id + 1)
                            
                            idx_classifications.append([
                                classification_data['inverted_label_map'][name], 
                                round(float(confidence), 5)
                            ])
                        
                        # Sort by confidence and keep top result
                        idx_classifications = sorted(idx_classifications, key=lambda x: x[1], reverse=True)
                        if idx_classifications:
                            detection['classifications'] = [idx_classifications[0]]
                    
                    animals_processed += 1
                    
                except Exception as e:
                    print(f"Error processing detection in frame {frame_number}: {str(e)}")
                    continue
        
        return frame_detections
    
    # Process this chunk of frames
    run_callback_on_frames(
        input_video_file=video_path,
        frame_callback=frame_callback,
        frames_to_process=frames_to_process,
        verbose=False
    )
    
    return animals_processed


def classify_video_detections_simple(json_path, model_path, model_type, model_env, chunk_size=1000):
    """
    Classify animal detections in videos using simple chunked processing.
    
    Args:
        json_path (str): Path to JSON with video detection results
        model_path (str): Path to classification model
        model_type (str): Model type/architecture  
        model_env (str): Environment name
        chunk_size (int): Number of frames to process per chunk
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get label mapping
    label_map = data.get('detection_categories', {'1': 'animal', '2': 'person', '3': 'vehicle'})
    
    # Initialize classification data
    if 'classification_categories' not in data:
        data['classification_categories'] = {}
    
    classification_data = {
        'inverted_label_map': {v: k for k, v in data['classification_categories'].items()}
    }
    
    # Get video directory
    video_dir = os.path.dirname(json_path)
    
    # Count total detections
    total_detections = 0
    videos_to_process = []
    
    for image_entry in data['images']:
        if is_video_file(image_entry['file']) and 'detections' in image_entry:
            video_path = os.path.join(video_dir, image_entry['file'])
            
            if not os.path.exists(video_path):
                print(f"Warning: Video {image_entry['file']} not found")
                continue
            
            # Count and group detections by frame
            detections_by_frame = {}
            animal_count = 0
            
            for detection in image_entry['detections']:
                if label_map.get(detection['category'], '') == 'animal' and detection.get('conf', 0) >= 0.1:
                    frame_num = detection.get('frame_number', 0)
                    if frame_num not in detections_by_frame:
                        detections_by_frame[frame_num] = []
                    detections_by_frame[frame_num].append(detection)
                    animal_count += 1
            
            if detections_by_frame:
                videos_to_process.append({
                    'entry': image_entry,
                    'path': video_path,
                    'detections_by_frame': detections_by_frame,
                    'animal_count': animal_count
                })
                total_detections += animal_count
    
    if total_detections == 0:
        print("No animal detections found in videos.")
        return
    
    print(f"Processing {total_detections} animal detections in {len(videos_to_process)} videos...")
    
    # Load classification model ONCE
    get_crop, get_classification = load_classification_model(model_path, model_type, model_env)
    
    # Process each video with progress tracking
    with tqdm(total=total_detections, desc="Classifying") as pbar:
        for video_info in videos_to_process:
            detections_by_frame = video_info['detections_by_frame']
            frames_to_process = list(detections_by_frame.keys())
            
            # Split frames into chunks
            frame_chunks = [frames_to_process[i:i + chunk_size] 
                          for i in range(0, len(frames_to_process), chunk_size)]
            
            for chunk in frame_chunks:
                try:
                    animals_processed = process_video_chunk(
                        video_info['path'],
                        chunk,
                        get_crop,
                        get_classification,
                        detections_by_frame,
                        label_map,
                        classification_data
                    )
                    pbar.update(animals_processed)
                    
                except Exception as e:
                    print(f"Error processing chunk in {video_info['entry']['file']}: {str(e)}")
                    # Still update progress for this chunk
                    chunk_animals = sum(len(detections_by_frame.get(f, [])) for f in chunk)
                    pbar.update(chunk_animals)
    
    # Update classification categories
    data['classification_categories'] = {v: k for k, v in classification_data['inverted_label_map'].items()}
    
    # Save results
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=1)
    
    print("Video classification completed successfully!")


def main():
    """Main function for simple video classification."""
    parser = argparse.ArgumentParser(description='Simple video classification for AddaxAI')
    parser.add_argument('--model-path', required=True, help='Path to classification model')
    parser.add_argument('--model-type', required=True, help='Type of classification model')
    parser.add_argument('--model-env', required=True, help='Environment name for the model')
    parser.add_argument('--json-path', required=True, help='Path to JSON with detection results')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Frames per chunk (default: 1000)')
    
    args = parser.parse_args()
    
    try:
        classify_video_detections_simple(
            args.json_path, args.model_path, args.model_type, 
            args.model_env, args.chunk_size
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()