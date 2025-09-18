"""
AddaxAI Classification Inference Library

Core inference functions for classifying MegaDetector animal crops using various ML models.
Provides standardized interface for different classification model types (PyTorch, TensorFlow, etc.).

Created by Peter van Lunteren
Latest edit by Peter van Lunteren on 19 May 2025
"""

# import logging
# print("=== CLS_INFERENCE.PY LOADED ===")


# import packages
# import io
import os
import json
# import datetime
# import contextlib
# import pandas as pd
from tqdm import tqdm
from PIL import Image
# from collections import defaultdict

from megadetector.data_management import read_exif

from utils.config import *


# Allow loading of truncated images to handle corrupted camera trap images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Global frame cache for video processing
FRAME_CACHE = {}

def extract_video_frames_to_cache(video_path, frame_numbers):
    """
    Extract specific frames from video into memory cache using MegaDetector's video_utils.
    
    Args:
        video_path (str): Path to video file
        frame_numbers (list): List of frame numbers to extract
    """
    # Import MegaDetector's video utilities
    from megadetector.detection.video_utils import run_callback_on_frames
    import cv2
    
    def frame_callback(image_np, frame_id):
        """Callback to store frame in memory cache"""
        # Convert numpy array (BGR) to PIL Image (RGB)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        FRAME_CACHE[frame_id] = image_pil
        return None
    
    try:
        # Extract only the required frames
        run_callback_on_frames(
            video_path, 
            frame_callback,
            frames_to_process=frame_numbers,
            verbose=False
        )
        print(f"Extracted {len(frame_numbers)} frames from video {video_path} to memory")
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")

def clear_frame_cache():
    """Clear all frames from memory cache to save memory"""
    global FRAME_CACHE
    FRAME_CACHE.clear()
    print("Cleared video frame cache from memory")

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

def create_raw_classifications(json_path,
                               GPU_availability,
                               crop_function,
                               inference_function):
    """
    Create raw classifications for animal detections in images.
    
    Args:
        json_path (str): Path to JSON file containing image detection data
        GPU_availability (bool): Whether GPU is available for processing
        crop_function (callable): Function to crop images based on bounding boxes
        inference_function (callable): Function to perform classification inference on crops
    """
    
    # Get directory containing images from JSON path
    img_dir = os.path.dirname(json_path)
    
    # Set classification detection threshold for filtering low-confidence detections
    cls_detec_thresh = 0.1
    
    # Load JSON data and label mapping once to avoid repeated file I/O
    with open(json_path) as image_recognition_file_content:
        data = json.load(image_recognition_file_content)
        label_map = fetch_label_map_from_json(json_path)
    
    # First pass: count valid animal crops, filter detections, and group by video files
    n_crops_to_classify = 0
    video_files_to_process = {}  # video_path -> list of image entries
    
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
            
            # Group video files for per-video processing
            if 'frames_processed' in image:
                video_path = os.path.join(img_dir, image['file'])
                if video_path not in video_files_to_process:
                    video_files_to_process[video_path] = []
                video_files_to_process[video_path].append(image)
    
    # Early return if no animals to classify - prevents unnecessary processing
    if n_crops_to_classify == 0:
        print("n_crops_to_classify is zero. Nothing to classify.")
        return

    # Begin crop and classification phase
    print(f"GPU available: {GPU_availability}")
    
    # Flag to track first iteration for label mapping initialization
    initial_it = True
    
    # Initialize classification categories if not present
    if 'classification_categories' not in data:
        data['classification_categories'] = {}
    
    # Create reverse mapping from category names to IDs for efficient lookup
    inverted_cls_label_map = {v: k for k, v in data['classification_categories'].items()}
    
    # Process each animal detection with progress tracking
    # smoothing=0 because the iterations are very inconsistent as it sometimes needs to extract frames first
    with tqdm(total=n_crops_to_classify, smoothing=0) as pbar: 
        
        # === STEP 1: Process all video files one by one ===
        for video_path, video_images in video_files_to_process.items():
            print(f"Processing video: {video_path}")
            
            # Extract frames for this video only
            frame_numbers = video_images[0]['frames_processed']  # All entries should have same frames_processed
            extract_video_frames_to_cache(video_path, frame_numbers)
            
            # Process all detections for this video
            for image in video_images:
                fname = image['file']
                if 'detections' in image:
                    for detection in image['detections']:
                        conf = detection["conf"]
                        category_id = detection['category']
                        category = label_map[category_id]
                        
                        # Process only animals (already filtered for confidence in first pass)
                        if category == 'animal':
                            # For video frames: get frame from memory cache
                            frame_number = detection.get('frame_number')
                            frame_key = f"frame{frame_number:06d}.jpg"  # Match MegaDetector format
                            
                            if frame_key in FRAME_CACHE:
                                # Load frame from memory cache
                                frame_image = FRAME_CACHE[frame_key]
                                bbox = detection['bbox']
                                crop = crop_function(frame_image, bbox)
                                
                                # Run classification inference on the cropped image
                                name_classifications = inference_function(crop)

                                # Convert classification results to indexed format for JSON storage
                                idx_classifications = []
                                for elem in name_classifications:
                                    name = elem[0]
                                    
                                    # On first iteration, build label mapping for new classification names
                                    if initial_it:
                                        if name not in inverted_cls_label_map:
                                            # Find highest existing index to assign next sequential ID
                                            highest_index = 0
                                            for key, value in inverted_cls_label_map.items():
                                                value = int(value)
                                                if value > highest_index:
                                                    highest_index = value
                                            inverted_cls_label_map[name] = str(highest_index + 1)
                                    
                                    # Convert numpy float32 to Python float for JSON serialization compatibility
                                    confidence = float(elem[1])
                                    idx_classifications.append([inverted_cls_label_map[name], round(confidence, 5)])
                                
                                # Set flag to false after first iteration
                                initial_it = False

                                # Sort classifications by confidence (highest first)
                                idx_classifications = sorted(idx_classifications, key=lambda x:x[1], reverse=True)
                                
                                # Keep only the top classification result (if any classifications exist)
                                if idx_classifications:
                                    only_top_classification = [idx_classifications[0]]
                                    detection['classifications'] = only_top_classification
                                else:
                                    # No valid classifications for this detection (e.g., invalid bounding box)
                                    detection['classifications'] = []

                                # Update progress bar
                                pbar.update(1)
                            else:
                                # Fallback: skip this detection if frame not in cache
                                print(f"Warning: Frame {frame_key} not found in cache")
            
            # Clear frame cache after processing this video to save memory
            clear_frame_cache()
            print(f"Completed video: {video_path} - cache cleared")
        
        # === STEP 2: Process regular images ===
        for image in data['images']:
            # Skip video files (already processed above)
            if 'frames_processed' in image:
                continue
                
            fname = image['file']
            if 'detections' in image:
                for detection in image['detections']:
                    conf = detection["conf"]
                    category_id = detection['category']
                    category = label_map[category_id]
                    
                    # Process only animals (already filtered for confidence in first pass)
                    if category == 'animal':
                        # Normal image processing: load from disk
                        img_fpath = os.path.join(img_dir, fname)
                        bbox = detection['bbox']
                        crop = crop_function(Image.open(img_fpath), bbox)
                        
                        # Run classification inference on the cropped image
                        name_classifications = inference_function(crop)

                        # Convert classification results to indexed format for JSON storage
                        idx_classifications = []
                        for elem in name_classifications:
                            name = elem[0]
                            
                            # On first iteration, build label mapping for new classification names
                            if initial_it:
                                if name not in inverted_cls_label_map:
                                    # Find highest existing index to assign next sequential ID
                                    highest_index = 0
                                    for key, value in inverted_cls_label_map.items():
                                        value = int(value)
                                        if value > highest_index:
                                            highest_index = value
                                    inverted_cls_label_map[name] = str(highest_index + 1)
                            
                            # Convert numpy float32 to Python float for JSON serialization compatibility
                            confidence = float(elem[1])
                            idx_classifications.append([inverted_cls_label_map[name], round(confidence, 5)])
                        
                        # Set flag to false after first iteration
                        initial_it = False

                        # Sort classifications by confidence (highest first)
                        idx_classifications = sorted(idx_classifications, key=lambda x:x[1], reverse=True)
                        
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
        