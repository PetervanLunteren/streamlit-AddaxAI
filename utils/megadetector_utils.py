"""
MegaDetector Utilities

Clean interface for running MegaDetector on images and videos.
Keeps analysis_utils.py clean by separating MegaDetector-specific logic.

Created by Peter van Lunteren for AddaxAI video support
"""

import os
import subprocess
import streamlit as st
import json

from utils.config import ADDAXAI_ROOT, log
from utils.video_utils import get_video_datetime


def run_megadetector_on_images(det_modelID, model_meta, deployment_folder, output_file, pbars):
    """
    Run MegaDetector on images in a deployment folder.
    
    Args:
        det_modelID (str): Detection model ID
        model_meta (dict): Model metadata for the detection model
        deployment_folder (str): Path to deployment folder containing images
        output_file (str): Path to output JSON file
        pbars: Progress bar manager
        
    Returns:
        bool: True if successful, False if cancelled or failed
    """
    model_file = os.path.join(
        ADDAXAI_ROOT, "models", "det", det_modelID, model_meta["model_fname"])
    
    command = [
        f"{ADDAXAI_ROOT}/envs/env-megadetector/bin/python",
        "-m", "megadetector.detection.run_detector_batch", 
        "--recursive", 
        "--output_relative_filenames", 
        "--include_image_size", 
        "--include_image_timestamp", 
        "--include_exif_data",
        model_file,
        deployment_folder,
        output_file
    ]

    # Log the command for debugging/audit purposes
    log(f"\n\nRunning MegaDetector image detection command:\n{' '.join(command)}\n")

    return _run_megadetector_subprocess(command, pbars, "Detecting... (images)")


def run_megadetector_on_videos(det_modelID, model_meta, deployment_folder, output_file, pbars):
    """
    Run MegaDetector on videos in a deployment folder.
    
    Args:
        det_modelID (str): Detection model ID
        model_meta (dict): Model metadata for the detection model
        deployment_folder (str): Path to deployment folder containing videos
        output_file (str): Path to output JSON file
        pbars: Progress bar manager
        
    Returns:
        bool: True if successful, False if cancelled or failed
    """
    model_file = os.path.join(
        ADDAXAI_ROOT, "models", "det", det_modelID, model_meta["model_fname"])
    
    command = [
        f"{ADDAXAI_ROOT}/envs/env-megadetector/bin/python",
        "-m", "megadetector.detection.process_video",
        "--recursive",
        "--verbose",  # Add verbose flag for more output
        "--time_sample", "1",  # Extract 1 frame per second
        model_file,
        deployment_folder,
        "--output_json_file", output_file
    ]

    # Log the command for debugging/audit purposes
    log(f"\n\nRunning MegaDetector video detection command:\n{' '.join(command)}\n")

    return _run_megadetector_subprocess(command, pbars, "Detecting... (videos)")


def _run_megadetector_subprocess(command, pbars, progress_label):
    """
    Run MegaDetector subprocess with progress tracking and cancellation support.
    
    Args:
        command (list): Command to execute
        pbars: Progress bar manager
        progress_label (str): Label for progress tracking
        
    Returns:
        bool: True if successful, False if cancelled or failed
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=False,
        universal_newlines=True,
        cwd=ADDAXAI_ROOT  # Set working directory to project root
    )

    for line in process.stdout:
        # Check if processing was cancelled
        if st.session_state.get("cancel_processing", False):
            log(f"{progress_label} cancelled by user - terminating subprocess")
            process.terminate()
            process.wait()
            return False

        line = line.strip()
        print(line)
        pbars.update_from_tqdm_string(progress_label, line)

    process.stdout.close()
    process.wait()

    if process.returncode != 0:
        st.error(f"{progress_label} failed with exit code {process.returncode}.")
        return False
    
    return True


def enrich_video_results_with_datetime(video_results_file, deployment_folder):
    """
    Add datetime metadata to video detection results.
    
    Args:
        video_results_file (str): Path to video detection results JSON
        deployment_folder (str): Path to deployment folder containing videos
    """
    if not os.path.exists(video_results_file):
        return
    
    with open(video_results_file, 'r') as f:
        data = json.load(f)
    
    # Process each video entry
    for item in data.get('images', []):
        if 'file' in item:
            # Construct full path to video file
            video_path = os.path.join(deployment_folder, item['file'])
            
            # Extract datetime
            datetime_obj = get_video_datetime(video_path)
            if datetime_obj:
                # Format as requested: '2022:03:02 22:59:00'
                item['datetime'] = datetime_obj.strftime('%Y:%m:%d %H:%M:%S')
                log(f"Added datetime {item['datetime']} to video {item['file']}")
    
    # Write back the enriched data
    with open(video_results_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    log(f"Enriched video results with datetime metadata: {video_results_file}")


def merge_detection_results(video_results_file, image_results_file, output_file, deployment_folder=None):
    """
    Merge video and image detection results into a single JSON file.
    
    Args:
        video_results_file (str): Path to video detection results JSON
        image_results_file (str): Path to image detection results JSON  
        output_file (str): Path to merged output JSON file
        deployment_folder (str): Path to deployment folder (needed for datetime extraction)
    """
    merged_data = None
    
    # Load video results if they exist
    if os.path.exists(video_results_file):
        # Enrich video results with datetime before merging
        if deployment_folder:
            enrich_video_results_with_datetime(video_results_file, deployment_folder)
        
        with open(video_results_file, 'r') as f:
            video_data = json.load(f)
        merged_data = video_data
        log(f"Loaded video results: {len(video_data.get('images', []))} videos")
    
    # Load and merge image results if they exist
    if os.path.exists(image_results_file):
        with open(image_results_file, 'r') as f:
            image_data = json.load(f)
        
        if merged_data is None:
            # Only images, no videos
            merged_data = image_data
            log(f"Loaded image results: {len(image_data.get('images', []))} images")
        else:
            # Merge images into existing video data
            merged_data['images'].extend(image_data.get('images', []))
            log(f"Merged image results: {len(image_data.get('images', []))} images added")
            
            # Merge detection categories (should be the same, but just in case)
            merged_data['detection_categories'].update(
                image_data.get('detection_categories', {})
            )
    
    # Write merged results
    if merged_data:
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        total_items = len(merged_data.get('images', []))
        log(f"Merged detection results saved: {total_items} total items in {output_file}")
        
        # Clean up temporary files
        if os.path.exists(video_results_file):
            os.remove(video_results_file)
            log(f"Removed temporary video results file: {video_results_file}")
        if os.path.exists(image_results_file):
            os.remove(image_results_file)
            log(f"Removed temporary image results file: {image_results_file}")
    else:
        log("No detection results to merge - no videos or images found")


def count_media_files_in_folder(folder_path):
    """
    Count video and image files in a folder.
    
    Args:
        folder_path (str): Path to folder to scan
        
    Returns:
        tuple: (video_count, image_count)
    """
    from pathlib import Path
    from utils.config import VIDEO_EXTENSIONS, IMG_EXTENSIONS
    
    folder = Path(folder_path)
    video_count = 0
    image_count = 0
    
    for f in folder.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                video_count += 1
            elif ext in IMG_EXTENSIONS:
                image_count += 1
    
    return video_count, image_count