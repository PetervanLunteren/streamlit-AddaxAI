"""
Simple Image Classification for AddaxAI - Direct Model Loading

Simple, straightforward image classification without model servers:
1. Load classification model once
2. Process all images directly
3. No complex subprocess communication

Created by Claude for AddaxAI simplified architecture
"""

import os
import json
import importlib.util
import argparse
import sys
from tqdm import tqdm
from PIL import Image

# Add AddaxAI to path
sys.path.append(os.environ.get('PYTHONPATH', ''))
from utils.config import ADDAXAI_ROOT


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
    return data['detection_categories']


def create_raw_classifications_simple(json_path, model_path, model_type, model_env):
    """
    Create raw classifications for animal detections using direct model loading.
    
    Args:
        json_path (str): Path to JSON file containing image detection data
        model_path (str): Path to the classification model file
        model_type (str): Type/architecture of the classification model
        model_env (str): Environment name for the model
    """
    
    # Get directory containing images from JSON path
    img_dir = os.path.dirname(json_path)
    
    # Set classification detection threshold
    cls_detec_thresh = 0.1
    
    # Load JSON data and label mapping
    with open(json_path) as f:
        data = json.load(f)
    
    label_map = fetch_label_map_from_json(json_path)
    
    # Count valid animal crops and filter detections
    n_crops_to_classify = 0
    for image in data['images']:
        if 'detections' in image:
            # Filter detections in place
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
                else:
                    # Keep non-animal detections regardless of confidence
                    filtered_detections.append(detection)
            
            # Update detections list
            image['detections'] = filtered_detections
    
    # Early return if no animals to classify
    if n_crops_to_classify == 0:
        print("n_crops_to_classify is zero. Nothing to classify.")
        return

    print(f"Processing {n_crops_to_classify} animal detections with model: {model_type}")
    
    # Initialize classification categories
    if 'classification_categories' not in data:
        data['classification_categories'] = {}
    
    # Create reverse mapping from category names to IDs
    inverted_cls_label_map = {v: k for k, v in data['classification_categories'].items()}
    
    # Load classification model ONCE
    get_crop, get_classification = load_classification_model(model_path, model_type, model_env)
    
    # Process each animal detection
    with tqdm(total=n_crops_to_classify, desc="Classifying") as pbar:
        for image in data['images']:
            # Get image filename
            fname = image['file']
            if 'detections' in image:
                # Load image once per image file
                img_fpath = os.path.join(img_dir, fname)
                
                try:
                    image_pil = Image.open(img_fpath)
                except Exception as e:
                    print(f"Error loading image {fname}: {str(e)}")
                    continue
                
                for detection in image['detections']:
                    conf = detection["conf"]
                    category_id = detection['category']
                    category = label_map[category_id]
                    
                    # Process only animals (already filtered for confidence)
                    if category == 'animal':
                        bbox = detection['bbox']
                        
                        try:
                            # Get crop using model-specific function
                            crop = get_crop(image_pil, bbox)
                            if crop is None:
                                detection['classifications'] = []
                                pbar.update(1)
                                continue
                            
                            # Classify the crop
                            name_classifications = get_classification(crop)

                            # Convert classification results to indexed format
                            idx_classifications = []
                            for name, confidence in name_classifications:
                                # Build label mapping for new classification names
                                if name not in inverted_cls_label_map:
                                    # Find highest existing index
                                    highest_index = 0
                                    for key, value in inverted_cls_label_map.items():
                                        value = int(value)
                                        if value > highest_index:
                                            highest_index = value
                                    inverted_cls_label_map[name] = str(highest_index + 1)
                                
                                # Convert to indexed format
                                idx_classifications.append([
                                    inverted_cls_label_map[name], 
                                    round(float(confidence), 5)
                                ])
                            
                            # Sort classifications by confidence (highest first)
                            idx_classifications = sorted(idx_classifications, key=lambda x: x[1], reverse=True)
                            
                            # Keep only the top classification result
                            if idx_classifications:
                                detection['classifications'] = [idx_classifications[0]]
                            else:
                                detection['classifications'] = []

                        except Exception as e:
                            print(f"Error processing detection in {fname}: {str(e)}")
                            detection['classifications'] = []

                        # Update progress bar
                        pbar.update(1)

    # Update classification categories mapping
    data['classification_categories'] = {v: k for k, v in inverted_cls_label_map.items()}
    
    # Write updated data back to JSON file
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=1)


def main():
    """Main function for simple image classification."""
    parser = argparse.ArgumentParser(description='Simple image classification for AddaxAI')
    parser.add_argument('--model-path', required=True, help='Path to classification model file')
    parser.add_argument('--model-type', required=True, help='Type of classification model')
    parser.add_argument('--model-env', required=True, help='Environment name for the model')
    parser.add_argument('--json-path', required=True, help='Path to JSON file with detection results')
    parser.add_argument('--country', default=None, help='Country code for geofencing')
    parser.add_argument('--state', default=None, help='State code for geofencing')
    
    args = parser.parse_args()
    
    try:
        create_raw_classifications_simple(
            args.json_path, args.model_path, args.model_type, args.model_env
        )
        print("Classification completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()