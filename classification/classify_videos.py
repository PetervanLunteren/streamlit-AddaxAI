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
import base64
import io
import uuid
import threading
import queue
import time
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


class ModelServerClient:
    """Client for communicating with persistent model server."""
    
    def __init__(self, model_path, model_type, model_env):
        """
        Initialize model server client.
        
        Args:
            model_path (str): Path to the model file
            model_type (str): Type of model (directory name)
            model_env (str): Environment name (e.g., 'pytorch', 'tensorflow-v1')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model_env = model_env
        self.server_process = None
        self.start_server()
    
    def start_server(self):
        """Start the persistent model server subprocess."""
        # Build the command
        python_executable = f"{ADDAXAI_ROOT}/envs/env-{self.model_env}/bin/python"
        server_script = f"{ADDAXAI_ROOT}/classification/model_server.py"
        
        command = [
            python_executable,
            server_script,
            '--model-path', self.model_path,
            '--model-type', self.model_type
        ]
        
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = ADDAXAI_ROOT
        
        try:
            # Start the server process
            self.server_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=ADDAXAI_ROOT,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Wait for server to be ready
            while True:
                line = self.server_process.stderr.readline()
                if 'Model server ready' in line:
                    print("Model server started successfully")
                    break
                elif self.server_process.poll() is not None:
                    # Process died
                    stdout, stderr = self.server_process.communicate()
                    raise RuntimeError(f"Model server failed to start. stderr: {stderr}")
                    
        except Exception as e:
            print(f"Failed to start model server: {str(e)}")
            raise
    
    def classify_crop(self, image_np, bbox_coords=None):
        """
        Classify a single crop using the persistent model server.
        
        Args:
            image_np (numpy.ndarray): Image as numpy array
            bbox_coords (list, optional): Normalized bounding box [x,y,w,h]
        
        Returns:
            list: Classification results as [["species_name", confidence], ...]
        """
        if self.server_process is None or self.server_process.poll() is not None:
            print("Model server is not running")
            return []
        
        try:
            # Convert numpy array to PIL Image
            from PIL import Image
            if image_np.dtype != 'uint8':
                image_np = (image_np * 255).astype('uint8')
            image_pil = Image.fromarray(image_np)
            
            # Encode image as base64 JPEG with reduced quality for memory efficiency
            buffer = io.BytesIO()
            image_pil.save(buffer, format='JPEG', quality=60)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Create request
            request = {
                "request_id": str(uuid.uuid4()),
                "image_data": image_data,
                "bbox": bbox_coords
            }
            
            # Send request to server
            request_json = json.dumps(request) + '\n'
            self.server_process.stdin.write(request_json)
            self.server_process.stdin.flush()
            
            # Read response
            response_line = self.server_process.stdout.readline()
            if not response_line:
                # Check if server died
                if self.server_process.poll() is not None:
                    print(f"Model server process died (exit code: {self.server_process.poll()})")
                return []
            
            response_line = response_line.strip()
            if not response_line:
                return []
            
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as e:
                print(f"Failed to parse server response: {response_line[:100]}... (JSON error: {str(e)})")
                return []
            
            if response.get('error'):
                print(f"Model server error: {response['error']}")
                return []
            
            return response.get('classifications', [])
            
        except Exception as e:
            print(f"Error communicating with model server: {str(e)}")
            # Check if server is still alive
            if self.server_process and self.server_process.poll() is not None:
                print(f"Server died during request processing")
            return []
    
    def shutdown(self):
        """Shutdown the model server."""
        if self.server_process and self.server_process.poll() is None:
            try:
                # Send shutdown signal
                shutdown_request = json.dumps({"shutdown": True}) + '\n'
                self.server_process.stdin.write(shutdown_request)
                self.server_process.stdin.flush()
                
                # Wait for graceful shutdown
                self.server_process.wait(timeout=10)
                
            except Exception:
                # Force kill if graceful shutdown fails
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            finally:
                self.server_process = None


def call_model_inference(model_server_client, image_np, bbox_coords=None):
    """
    Call the persistent model server to classify a single image crop.
    
    Args:
        model_server_client (ModelServerClient): Client for persistent model server
        image_np (numpy.ndarray): Image as numpy array
        bbox_coords (list, optional): Normalized bounding box [x,y,w,h]
    
    Returns:
        list: Classification results as [["species_name", confidence], ...]
    """
    return model_server_client.classify_crop(image_np, bbox_coords)


def create_video_frame_callback(model_server_client, detections_by_frame, 
                                label_map, classification_data):
    """
    Create callback function to process video frames and classify detections.
    
    Args:
        model_server_client (ModelServerClient): Client for persistent model server
        detections_by_frame: Dict mapping frame numbers to detection lists
        label_map: Detection category label mapping
        classification_data: Dict to store classification results and categories
    
    Returns:
        Callback function for video frame processing
    """
    def frame_callback(image_np, frame_filename):
        # Extract frame number from filename (e.g., "frame000024.jpg" -> 24)
        frame_number = int(frame_filename.replace("frame", "").replace(".jpg", ""))
        
        # Get detections for this frame
        frame_detections = detections_by_frame.get(frame_number, [])
        
        # Process each animal detection in this frame
        for detection in frame_detections:
            category_id = detection['category']
            category = label_map[category_id]
            
            # Only classify animal detections
            if category == 'animal' and detection.get('conf', 0) >= 0.1:  # confidence threshold
                bbox = detection['bbox']
                
                # Call persistent model server - no temp files needed!
                name_classifications = call_model_inference(
                    model_server_client, image_np, bbox
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
        
        return frame_detections
    
    return frame_callback


def classify_video_detections(json_path, model_path, model_type, model_env):
    """
    Classify animal detections in video files using persistent model server.
    
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
    
    # Count total detections for progress tracking
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
    
    # Start persistent model server - this loads the model ONCE
    model_server_client = None
    try:
        model_server_client = ModelServerClient(model_path, model_type, model_env)
        
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
                
                # Create callback for this video - uses persistent model server
                callback = create_video_frame_callback(
                    model_server_client, detections_by_frame,
                    label_map, classification_data
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
    
    finally:
        # Always shutdown the model server
        if model_server_client:
            model_server_client.shutdown()
    
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