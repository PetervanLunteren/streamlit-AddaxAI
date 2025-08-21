"""
AddaxAI Classification Inference Library - Scalable Subprocess Architecture

Core inference functions for classifying MegaDetector animal crops using subprocess-based
model inference. This provides a scalable interface that separates orchestration from 
model-specific inference.

Created by Claude for AddaxAI scalable architecture
Latest edit for subprocess-based inference
"""

import os
import json
import subprocess
from tqdm import tqdm
from PIL import Image

# Add AddaxAI to path
import sys
sys.path.append(os.environ.get('PYTHONPATH', ''))
from utils.config import ADDAXAI_ROOT


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


def fetch_label_map_from_json(path_to_json):
    """
    Extract detection category labels from MegaDetector JSON output.
    
    Args:
        path_to_json (str): Path to detection results JSON file
        
    Returns:
        dict: Label mapping for detection categories
    """
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
    label_map = data['detection_categories']
    return label_map


def create_raw_classifications_subprocess(json_path, model_path, model_type, model_env):
    """
    Create raw classifications for animal detections using subprocess-based inference.
    
    Args:
        json_path (str): Path to JSON file containing image detection data
        model_path (str): Path to the classification model file
        model_type (str): Type/architecture of the classification model
        model_env (str): Environment name for the model
    """
    
    # Get directory containing images from JSON path
    img_dir = os.path.dirname(json_path)
    
    # Set classification detection threshold for filtering low-confidence detections
    cls_detec_thresh = 0.1
    
    # Load JSON data and label mapping once to avoid repeated file I/O
    with open(json_path) as image_recognition_file_content:
        data = json.load(image_recognition_file_content)
        label_map = fetch_label_map_from_json(json_path)
    
    # First pass: count valid animal crops and filter detections by confidence
    n_crops_to_classify = 0
    for image in data['images']:
        if 'detections' in image:
            # Filter detections in place to remove low-confidence animal detections
            filtered_detections = []
            for detection in image['detections']:
                conf = detection["conf"]
                category_id = detection['category']
                category = label_map[category_id]
                
                if category == 'animal':
                    # Only keep animal detections above threshold
                    if conf >= cls_detec_thresh:
                        filtered_detections.append(detection)
                        n_crops_to_classify += 1
                    # Low confidence animal detections are excluded
                else:
                    # Keep non-animal detections regardless of confidence
                    filtered_detections.append(detection)
            
            # Update the detections list with filtered results
            image['detections'] = filtered_detections
    
    # Early return if no animals to classify - prevents unnecessary processing
    if n_crops_to_classify == 0:
        print("n_crops_to_classify is zero. Nothing to classify.")
        return

    # Begin crop and classification phase
    print(f"Processing {n_crops_to_classify} animal detections with model: {model_type}")
    
    # Initialize classification categories if not present
    if 'classification_categories' not in data:
        data['classification_categories'] = {}
    
    # Create reverse mapping from category names to IDs for efficient lookup
    inverted_cls_label_map = {v: k for k, v in data['classification_categories'].items()}
    
    # Process each animal detection with progress tracking
    with tqdm(total=n_crops_to_classify, desc="Classifying") as pbar:
        for image in data['images']:
            # Get image filename for cropping
            fname = image['file']
            if 'detections' in image:
                # Get full path to image
                img_fpath = os.path.join(img_dir, fname)
                
                for detection in image['detections']:
                    conf = detection["conf"]
                    category_id = detection['category']
                    category = label_map[category_id]
                    
                    # Process only animals (already filtered for confidence in first pass)
                    if category == 'animal':  # No need to check conf again
                        bbox = detection['bbox']
                        
                        # Call model inference subprocess
                        name_classifications = call_model_inference(
                            model_path, model_type, model_env, img_fpath, bbox
                        )

                        # Convert classification results to indexed format for JSON storage
                        idx_classifications = []
                        for name, confidence in name_classifications:
                            # Build label mapping for new classification names
                            if name not in inverted_cls_label_map:
                                # Find highest existing index to assign next sequential ID
                                highest_index = 0
                                for key, value in inverted_cls_label_map.items():
                                    value = int(value)
                                    if value > highest_index:
                                        highest_index = value
                                inverted_cls_label_map[name] = str(highest_index + 1)
                            
                            # Convert to indexed format
                            idx_classifications.append([inverted_cls_label_map[name], round(float(confidence), 5)])
                        
                        # Sort classifications by confidence (highest first)
                        idx_classifications = sorted(idx_classifications, key=lambda x: x[1], reverse=True)
                        
                        # Keep only the top classification result (if any classifications exist)
                        if idx_classifications:
                            only_top_classification = [idx_classifications[0]]
                            detection['classifications'] = only_top_classification
                        else:
                            # No valid classifications for this detection (e.g., invalid bounding box)
                            detection['classifications'] = []

                        # Update progress bar
                        pbar.update(1)

    # Update classification categories mapping in data structure
    data['classification_categories'] = {v: k for k, v in inverted_cls_label_map.items()}
    
    # Write updated data back to JSON file with formatting
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=1)


def main():
    """Main function for subprocess-based classification."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Subprocess-based classification inference for AddaxAI')
    parser.add_argument('--model-path', required=True, help='Path to classification model file')
    parser.add_argument('--model-type', required=True, help='Type of classification model')
    parser.add_argument('--model-env', required=True, help='Environment name for the model')
    parser.add_argument('--json-path', required=True, help='Path to JSON file with detection results')
    parser.add_argument('--country', default=None, help='Country code for geofencing')
    parser.add_argument('--state', default=None, help='State code for geofencing')
    
    args = parser.parse_args()
    
    try:
        create_raw_classifications_subprocess(
            args.json_path, args.model_path, args.model_type, args.model_env
        )
        print("Classification completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()