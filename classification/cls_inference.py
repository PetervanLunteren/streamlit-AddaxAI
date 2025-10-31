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
import csv
# import datetime
# import contextlib
# import pandas as pd
import sys
from tqdm import tqdm
from PIL import Image
# from collections import defaultdict

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
    import cv2
    
    # import from a separated utils file to avoid megadetector dependency for every model type
    from utils.md_video_utils import run_callback_on_frames
    
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


def _extract_taxon_component(value, prefix):
    """
    Normalize a taxonomy CSV value by removing the expected prefix and lowercasing.
    """
    if value is None:
        return ""
    value = str(value).strip()
    if not value:
        return ""
    prefix_lower = f"{prefix.lower()} "
    value_lower = value.lower()
    if value_lower.startswith(prefix_lower):
        return value[len(prefix) + 1:].strip().lower()
    return value_lower.strip()


def _load_taxonomy_lookup_from_csv(model_id):
    """
    Load taxonomy metadata for a classification model from its taxon-mapping.csv.

    Returns:
        dict: {model_class_label: "class;order;family;genus;species"}
    """
    if not model_id:
        return {}

    mapping_path = os.path.join(ADDAXAI_ROOT, "models", "cls", model_id, "taxon-mapping.csv")
    if not os.path.exists(mapping_path):
        print(f"Warning: taxon-mapping.csv not found for model {model_id} at {mapping_path}")
        return {}

    taxonomy_lookup = {}
    try:
        with open(mapping_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                label = (row.get("model_class") or "").strip()
                if not label:
                    continue

                class_name = _extract_taxon_component(row.get("level_class"), "class")
                order_name = _extract_taxon_component(row.get("level_order"), "order")
                family_name = _extract_taxon_component(row.get("level_family"), "family")
                genus_name = _extract_taxon_component(row.get("level_genus"), "genus")
                species_name = _extract_taxon_component(row.get("level_species"), "species")

                taxonomy_lookup[label] = ";".join([
                    class_name,
                    order_name,
                    family_name,
                    genus_name,
                    species_name
                ])
    except Exception as exc:
        print(f"Warning: failed to parse taxonomy CSV for model {model_id}: {exc}")

    return taxonomy_lookup

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
                               inference_function,
                               model_id=None):
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
    
    # Load JSON data and label mapping once to avoid repeated file I/O
    with open(json_path) as image_recognition_file_content:
        data = json.load(image_recognition_file_content)
        label_map = fetch_label_map_from_json(json_path)
        existing_taxonomy_descriptions = data.get('classification_category_descriptions', {})
    
    # First pass: count animal crops and group by video files
    # Note: No filtering applied - all detections in JSON are processed
    n_crops_to_classify = 0
    video_files_to_process = {}  # video_path -> list of image entries
    
    for image in data['images']:
        if 'detections' in image:
            # Count animal detections for progress tracking
            for detection in image['detections']:
                category_id = detection['category']
                category = label_map[category_id]
                
                if category == 'animal':
                    n_crops_to_classify += 1
            
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
                        
                        # Process only animals
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
                    
                    # Process only animals
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

    # Attach taxonomy descriptions derived from model metadata
    metadata = data.get("addaxai_metadata", {}) if data else {}
    if not model_id:
        model_id = metadata.get("selected_cls_modelID")
    if not model_id or model_id in ["NONE", "unknown"]:
        model_id = metadata.get("classification_model_id")

    taxonomy_lookup = _load_taxonomy_lookup_from_csv(model_id)
    classification_categories = data.get("classification_categories", {})
    taxonomy_descriptions = {}
    default_empty = ";".join([""] * 5)

    for category_id, label in classification_categories.items():
        if not isinstance(category_id, str):
            category_key = str(category_id)
        else:
            category_key = category_id

        label = label.strip() if isinstance(label, str) else label
        taxonomy_string = ""
        if label and taxonomy_lookup:
            taxonomy_string = taxonomy_lookup.get(label, "")

        if not taxonomy_string and existing_taxonomy_descriptions:
            taxonomy_string = existing_taxonomy_descriptions.get(category_id) or \
                              existing_taxonomy_descriptions.get(category_key, "")

        taxonomy_descriptions[category_key] = taxonomy_string if taxonomy_string else default_empty

    data['classification_category_descriptions'] = taxonomy_descriptions
    
    # Write updated data back to JSON file with formatting
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=1)
        
