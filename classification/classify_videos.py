"""
Super Simple Video Classification for AddaxAI - Video Chunking

Process videos in chunks to make everything simple and understandable:
1. Take 10 videos at a time
2. Run MegaDetector on all 10 videos → extract frames with detections to temp dir
3. Load classification model once → classify all frames from those 10 videos  
4. Clean up temp frames
5. Repeat for next 10 videos
6. Merge all results into final JSON

No more complex frame chunking - just simple video batches!

Created by Claude for AddaxAI super-simplified architecture
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


def extract_frames_with_detections(video_chunk, temp_dir, label_map):
    """
    Extract frames that contain animal detections from a chunk of videos.
    
    Args:
        video_chunk (list): List of video info dictionaries
        temp_dir (str): Temporary directory to save frames
        label_map (dict): Detection category mapping
        
    Returns:
        dict: Mapping of saved frame paths to their detection info
    """
    frame_detection_map = {}
    
    for video_info in video_chunk:
        video_path = video_info['path']
        detections_by_frame = video_info['detections_by_frame']
        frames_to_extract = list(detections_by_frame.keys())
        
        if not frames_to_extract:
            continue
        
        def frame_callback(image_np, frame_filename):
            # Extract frame number from filename  
            frame_number = int(frame_filename.replace("frame", "").replace(".jpg", ""))
            
            # Check if this frame has animal detections
            if frame_number in detections_by_frame:
                frame_detections = detections_by_frame[frame_number]
                
                # Check if any detections are animals above threshold
                animal_detections = []
                for detection in frame_detections:
                    category_id = detection['category']
                    category = label_map[category_id]
                    if category == 'animal' and detection.get('conf', 0) >= 0.1:
                        animal_detections.append(detection)
                
                if animal_detections:
                    # Save frame to temp directory
                    from PIL import Image
                    if image_np.dtype != 'uint8':
                        image_np = (image_np * 255).astype('uint8')
                    image_pil = Image.fromarray(image_np)
                    
                    # Create unique filename
                    video_name = Path(video_info['entry']['file']).stem
                    frame_path = os.path.join(temp_dir, f"{video_name}_frame{frame_number:06d}.jpg")
                    image_pil.save(frame_path, 'JPEG', quality=85)
                    
                    # Store detection info
                    frame_detection_map[frame_path] = {
                        'video_entry': video_info['entry'],
                        'frame_number': frame_number,
                        'detections': animal_detections
                    }
            
            return []  # Don't need to return anything for extraction
        
        # Extract frames from this video
        try:
            run_callback_on_frames(
                input_video_file=video_path,
                frame_callback=frame_callback,
                frames_to_process=frames_to_extract,
                verbose=False
            )
        except Exception as e:
            print(f"Error extracting frames from {video_info['entry']['file']}: {str(e)}")
    
    return frame_detection_map


def classify_extracted_frames(frame_detection_map, get_crop, get_classification, classification_data):
    """
    Classify all extracted frames using the loaded model.
    
    Args:
        frame_detection_map (dict): Map of frame paths to detection info
        get_crop, get_classification: Model functions
        classification_data (dict): Classification data structure
        
    Returns:
        int: Number of animals processed
    """
    animals_processed = 0
    
    for frame_path, frame_info in frame_detection_map.items():
        # Load the frame image
        try:
            from PIL import Image
            image_pil = Image.open(frame_path)
        except Exception as e:
            print(f"Error loading frame {frame_path}: {str(e)}")
            continue
        
        # Process each animal detection in this frame
        for detection in frame_info['detections']:
            bbox = detection['bbox']
            
            try:
                # Get crop using model-specific function
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
                print(f"Error classifying detection in frame {frame_path}: {str(e)}")
                continue
    
    return animals_processed


def process_video_chunk(video_chunk, model_path, model_type, model_env, label_map, classification_data):
    """
    Process a chunk of videos: extract frames with detections, classify them, clean up.
    
    Args:
        video_chunk (list): List of video info dictionaries
        model_path, model_type, model_env: Model parameters
        label_map (dict): Detection category mapping
        classification_data (dict): Classification data structure
        
    Returns:
        int: Number of animals processed in this chunk
    """
    # Create temporary directory for this chunk
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting frames from {len(video_chunk)} videos...")
        
        # Step 1: Extract frames with animal detections to temp directory
        frame_detection_map = extract_frames_with_detections(video_chunk, temp_dir, label_map)
        
        if not frame_detection_map:
            print("No frames with animal detections found in this chunk.")
            return 0
        
        print(f"Extracted {len(frame_detection_map)} frames, loading classification model...")
        
        # Step 2: Load classification model once for this chunk
        get_crop, get_classification = load_classification_model(model_path, model_type, model_env)
        
        print(f"Classifying {len(frame_detection_map)} frames...")
        
        # Step 3: Classify all extracted frames
        animals_processed = classify_extracted_frames(
            frame_detection_map, get_crop, get_classification, classification_data
        )
        
        print(f"Classified {animals_processed} animals in this chunk.")
        
        # Step 4: Cleanup happens automatically when temp_dir context exits
        
        return animals_processed


def classify_videos_by_chunks(json_path, model_path, model_type, model_env, chunk_size=10):
    """
    Classify animal detections in videos by processing videos in chunks.
    
    Args:
        json_path (str): Path to JSON with video detection results
        model_path (str): Path to classification model
        model_type (str): Model type/architecture  
        model_env (str): Environment name
        chunk_size (int): Number of videos to process per chunk
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
    
    # Prepare videos for processing
    videos_to_process = []
    total_detections = 0
    
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
    print(f"Using video chunks of size {chunk_size}")
    
    # Split videos into chunks
    video_chunks = [videos_to_process[i:i + chunk_size] 
                   for i in range(0, len(videos_to_process), chunk_size)]
    
    # Process each chunk
    total_processed = 0
    with tqdm(total=total_detections, desc="Processing video chunks") as pbar:
        for i, video_chunk in enumerate(video_chunks, 1):
            print(f"\n--- Processing chunk {i}/{len(video_chunks)} ({len(video_chunk)} videos) ---")
            
            try:
                animals_processed = process_video_chunk(
                    video_chunk, model_path, model_type, model_env, 
                    label_map, classification_data
                )
                total_processed += animals_processed
                pbar.update(animals_processed)
                
            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                # Still update progress for this chunk
                chunk_animals = sum(v['animal_count'] for v in video_chunk)
                pbar.update(chunk_animals)
    
    # Update classification categories
    data['classification_categories'] = {v: k for k, v in classification_data['inverted_label_map'].items()}
    
    # Save results
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=1)
    
    print(f"\nVideo classification completed! Processed {total_processed}/{total_detections} animals successfully.")


def main():
    """Main function for video chunk classification."""
    parser = argparse.ArgumentParser(description='Simple video classification by chunks for AddaxAI')
    parser.add_argument('--model-path', required=True, help='Path to classification model')
    parser.add_argument('--model-type', required=True, help='Type of classification model')
    parser.add_argument('--model-env', required=True, help='Environment name for the model')
    parser.add_argument('--json-path', required=True, help='Path to JSON with detection results')
    parser.add_argument('--chunk-size', type=int, default=10, help='Videos per chunk (default: 10)')
    
    args = parser.parse_args()
    
    try:
        classify_videos_by_chunks(
            args.json_path, args.model_path, args.model_type, 
            args.model_env, args.chunk_size
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()