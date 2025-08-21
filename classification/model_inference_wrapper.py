#!/usr/bin/env python3
"""
Generic Model Inference Wrapper for AddaxAI

This script runs in model-specific environments and provides a clean interface
for running classification inference on individual image crops.

It dynamically loads the model-specific functions (get_crop, get_classification)
and provides a subprocess interface for the base environment to call.

Usage:
    python model_inference_wrapper.py --model-path /path/to/model.pt --model-type addax-yolov8 --image-path /path/to/crop.jpg [--bbox x,y,w,h]

Arguments:
    --model-path: Path to the model file
    --model-type: Type of model (directory name under classification/model_types/)
    --image-path: Path to image file to classify
    --bbox: Optional bounding box coordinates (normalized: x,y,w,h) for cropping
    --country: Optional country code for geofencing
    --state: Optional state code for geofencing

Output:
    JSON to stdout with classification results:
    {"classifications": [["species_name", confidence], ...]}

Created by Claude for AddaxAI scalable architecture
"""

import argparse
import json
import sys
import os
import importlib.util
from PIL import Image, ImageFile

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_model_functions(model_path, model_type):
    """
    Dynamically load the model-specific functions from the classification script.
    
    Args:
        model_path (str): Path to the model file
        model_type (str): Type of model (directory name)
    
    Returns:
        tuple: (get_crop_function, get_classification_function)
    """
    # Get the path to the model-specific script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_script_path = os.path.join(
        base_dir, "classification", "model_types", model_type, "classify_detections.py"
    )
    
    if not os.path.exists(model_script_path):
        raise ValueError(f"Model script not found: {model_script_path}")
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("model_module", model_script_path)
    model_module = importlib.util.module_from_spec(spec)
    
    # Set up the model arguments as expected by the script
    sys.argv = ['classify_detections.py', '--model-path', model_path, '--json-path', '/dev/null']
    
    # Add the base directory to sys.path so imports work correctly
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    
    # Store original sys.argv to restore later
    original_argv = sys.argv.copy()
    
    try:
        # Execute the module to load the model and define functions
        # We'll catch any exception from the main execution part
        spec.loader.exec_module(model_module)
        
        # Extract the functions we need
        get_crop = getattr(model_module, 'get_crop', None)
        get_classification = getattr(model_module, 'get_classification', None)
        
        if get_crop is None:
            raise ValueError(f"Model script {model_script_path} must define 'get_crop' function")
        if get_classification is None:
            raise ValueError(f"Model script {model_script_path} must define 'get_classification' function")
        
        return get_crop, get_classification
        
    except Exception as e:
        # The model script might fail when trying to run the main classification
        # but we may have still loaded the functions we need
        get_crop = getattr(model_module, 'get_crop', None)
        get_classification = getattr(model_module, 'get_classification', None)
        
        if get_crop is not None and get_classification is not None:
            return get_crop, get_classification
        else:
            raise ValueError(f"Error loading model from {model_script_path}: {str(e)}")
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def main():
    """Main inference function."""
    
    parser = argparse.ArgumentParser(description='Generic model inference wrapper')
    parser.add_argument('--model-path', required=True, help='Path to model file')
    parser.add_argument('--model-type', required=True, help='Type of model')
    parser.add_argument('--image-path', required=True, help='Path to image file to classify')
    parser.add_argument('--bbox', default=None, help='Bounding box coordinates (x,y,w,h) for cropping')
    parser.add_argument('--country', default=None, help='Country code for geofencing')
    parser.add_argument('--state', default=None, help='State code for geofencing')
    
    args = parser.parse_args()
    
    try:
        # Load the model-specific functions
        get_crop, get_classification = load_model_functions(args.model_path, args.model_type)
        
        # Load the image
        image = Image.open(args.image_path)
        
        # Crop the image if bbox is provided
        if args.bbox:
            # Parse bbox coordinates
            bbox_coords = list(map(float, args.bbox.split(',')))
            if len(bbox_coords) != 4:
                raise ValueError("Bbox must have 4 coordinates: x,y,w,h")
            
            # Crop the image using model-specific crop function
            crop = get_crop(image, bbox_coords)
            if crop is None:
                # Invalid crop (e.g., zero size)
                result = {"classifications": []}
                print(json.dumps(result))
                return
        else:
            # Use the whole image
            crop = image
        
        # Run classification on the crop
        classifications = get_classification(crop)
        
        # Format and output results
        result = {
            "classifications": classifications
        }
        
        # Output JSON to stdout
        print(json.dumps(result))
        
    except Exception as e:
        # Output error as JSON
        error_result = {
            "error": str(e),
            "classifications": []
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == '__main__':
    main()