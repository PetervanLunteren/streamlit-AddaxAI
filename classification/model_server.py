#!/usr/bin/env python3
"""
Persistent Model Server for AddaxAI Classification

Long-running subprocess that loads a classification model once and processes
crops sent via stdin/stdout communication. This eliminates the overhead of
loading the model for every single crop.

Communication Protocol:
  Input (stdin):  {"image_data": "base64_jpeg", "bbox": [x,y,w,h], "request_id": "unique_id"}
  Output (stdout): {"request_id": "unique_id", "classifications": [["species", confidence], ...], "error": null}
  Shutdown: {"shutdown": true}

Usage:
  python model_server.py --model-path /path/to/model.pt --model-type addax-yolov8

Created by Claude for AddaxAI persistent model architecture
"""

import argparse
import json
import sys
import os
import importlib.util
import base64
import io
from PIL import Image, ImageFile

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ModelServer:
    def __init__(self, model_path, model_type):
        """
        Initialize the model server with the specified model.
        
        Args:
            model_path (str): Path to the model file
            model_type (str): Type of model (directory name)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.get_crop = None
        self.get_classification = None
        self.load_model()
    
    def load_model(self):
        """Load the model-specific functions once at startup."""
        try:
            # Get the path to the model-specific script
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_script_path = os.path.join(
                base_dir, "classification", "model_types", self.model_type, "classify_detections.py"
            )
            
            if not os.path.exists(model_script_path):
                raise ValueError(f"Model script not found: {model_script_path}")
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("model_module", model_script_path)
            model_module = importlib.util.module_from_spec(spec)
            
            # Set up the model arguments as expected by the script
            original_argv = sys.argv.copy()
            sys.argv = ['classify_detections.py', '--model-path', self.model_path, '--json-path', '/dev/null']
            
            # Add the base directory to sys.path so imports work correctly
            if base_dir not in sys.path:
                sys.path.insert(0, base_dir)
            
            try:
                # Execute the module to load the model and define functions
                spec.loader.exec_module(model_module)
                
                # Extract the functions we need
                self.get_crop = getattr(model_module, 'get_crop', None)
                self.get_classification = getattr(model_module, 'get_classification', None)
                
                if self.get_crop is None:
                    raise ValueError(f"Model script must define 'get_crop' function")
                if self.get_classification is None:
                    raise ValueError(f"Model script must define 'get_classification' function")
                
                print(f"Model loaded successfully: {self.model_type}", file=sys.stderr)
                
            except Exception as e:
                # The model script might fail when trying to run the main classification
                # but we may have still loaded the functions we need
                self.get_crop = getattr(model_module, 'get_crop', None)
                self.get_classification = getattr(model_module, 'get_classification', None)
                
                if self.get_crop is not None and self.get_classification is not None:
                    print(f"Model loaded successfully (with warnings): {self.model_type}", file=sys.stderr)
                else:
                    raise ValueError(f"Error loading model from {model_script_path}: {str(e)}")
            finally:
                # Restore original sys.argv
                sys.argv = original_argv
                
        except Exception as e:
            print(f"Failed to load model: {str(e)}", file=sys.stderr)
            raise
    
    def process_request(self, request):
        """
        Process a single classification request.
        
        Args:
            request (dict): Request containing image_data, bbox, and request_id
            
        Returns:
            dict: Response with classifications and request_id
        """
        try:
            request_id = request.get('request_id', 'unknown')
            
            # Decode base64 image data
            image_data = request.get('image_data', '')
            if not image_data:
                return {
                    "request_id": request_id,
                    "classifications": [],
                    "error": "No image data provided"
                }
            
            # Decode from base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Get bbox if provided
            bbox = request.get('bbox')
            if bbox:
                # Crop the image using model-specific crop function
                crop = self.get_crop(image, bbox)
                if crop is None:
                    # Invalid crop (e.g., zero size)
                    return {
                        "request_id": request_id,
                        "classifications": [],
                        "error": "Invalid crop"
                    }
            else:
                # Use the whole image
                crop = image
            
            # Run classification on the crop
            classifications = self.get_classification(crop)
            
            return {
                "request_id": request_id,
                "classifications": classifications,
                "error": None
            }
            
        except Exception as e:
            return {
                "request_id": request.get('request_id', 'unknown'),
                "classifications": [],
                "error": str(e)
            }
    
    def run(self):
        """
        Main server loop - listen for requests on stdin and respond on stdout.
        """
        print("Model server ready", file=sys.stderr)
        sys.stderr.flush()
        
        try:
            while True:
                # Read line from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON request
                    request = json.loads(line)
                    
                    # Check for shutdown request
                    if request.get('shutdown'):
                        print("Model server shutting down", file=sys.stderr)
                        break
                    
                    # Process the classification request
                    response = self.process_request(request)
                    
                    # Send response back on stdout
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    # Send error response for invalid JSON
                    error_response = {
                        "request_id": "unknown",
                        "classifications": [],
                        "error": f"Invalid JSON: {str(e)}"
                    }
                    print(json.dumps(error_response))
                    sys.stdout.flush()
                    
        except KeyboardInterrupt:
            print("Model server interrupted", file=sys.stderr)
        except Exception as e:
            print(f"Model server error: {str(e)}", file=sys.stderr)
        finally:
            print("Model server stopped", file=sys.stderr)


def main():
    """Main function for model server."""
    
    parser = argparse.ArgumentParser(description='Persistent model server for classification')
    parser.add_argument('--model-path', required=True, help='Path to model file')
    parser.add_argument('--model-type', required=True, help='Type of model')
    parser.add_argument('--country', default=None, help='Country code for geofencing')
    parser.add_argument('--state', default=None, help='State code for geofencing')
    
    args = parser.parse_args()
    
    try:
        # Create and run the model server
        server = ModelServer(args.model_path, args.model_type)
        server.run()
        
    except Exception as e:
        print(f"Server startup error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()