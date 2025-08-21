"""
Video Classification Utility for AddaxAI

Handles classification of video files by extracting frames and running 
classification on the detected animals within those frames.

Created by Claude for AddaxAI video support
"""

import os
import json
import subprocess
from utils.config import ADDAXAI_ROOT, VIDEO_EXTENSIONS, log


def is_video_file(filename):
    """Check if a file is a video based on its extension."""
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def run_cls_videos(cls_modelID, json_fpath, pbars, country=None, state=None):
    """
    Run classification on video files in the JSON results.
    
    Args:
        cls_modelID (str): Classification model ID
        json_fpath (str): Path to JSON file with video detection results
        pbars: Progress bar manager
        country (str, optional): Country code for geofencing
        state (str, optional): State code for geofencing
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Skip classification if no model selected
    if cls_modelID == "NONE":
        return True
    
    # Check if JSON contains any video files
    try:
        with open(json_fpath, 'r') as f:
            data = json.load(f)
            
        has_videos = any(is_video_file(img['file']) for img in data['images'])
        if not has_videos:
            log("No video files found in JSON - skipping video classification")
            return True
            
    except Exception as e:
        log(f"Error reading JSON file: {str(e)}")
        return False
    
    log(f"Starting video classification with model {cls_modelID}...")
    # The progress bar for classification is just called "Classification"
    
    # Get model info and setup paths
    from utils.analysis_utils import get_cached_model_meta
    model_meta = get_cached_model_meta()
    model_info = model_meta['cls'][cls_modelID]
    
    cls_model_fpath = os.path.join(
        ADDAXAI_ROOT, "models", "cls", cls_modelID, model_info["model_fname"])
    
    # Create the video classification script path
    video_cls_script = os.path.join(ADDAXAI_ROOT, "classification", "video_cls_inference.py")
    
    # Build command arguments - run video classification in MegaDetector environment (has OpenCV)
    megadetector_python_executable = f"{ADDAXAI_ROOT}/envs/env-megadetector/bin/python"
    command_args = [
        megadetector_python_executable,
        video_cls_script,
        '--model-path', cls_model_fpath,
        '--model-type', model_info["type"],
        '--model-env', model_info["env"],
        '--json-path', json_fpath
    ]
    
    # Add country and state parameters if provided
    if country:
        command_args.extend(['--country', country])
    if state:
        command_args.extend(['--state', state])
    
    # Set environment variables for subprocess
    env = os.environ.copy()
    env['PYTHONPATH'] = ADDAXAI_ROOT
    
    # Fix MPS device issue on macOS by enabling CPU fallback
    from utils.config import OS_NAME
    if OS_NAME == 'macos':
        env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Log the command for debugging
    log(f"\n\nRunning video classification command:\n{' '.join(command_args)}\n")
    
    import streamlit as st
    process = subprocess.Popen(
        command_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=False,
        universal_newlines=True,
        cwd=ADDAXAI_ROOT,
        env=env
    )
    
    # Process output
    for line in process.stdout:
        # Check if processing was cancelled
        if st.session_state.get("cancel_processing", False):
            log("Video classification cancelled by user - terminating subprocess")
            process.terminate()
            process.wait()
            return False
        
        line = line.strip()
        log(line)
        pbars.update_from_tqdm_string("Classification... (videos)", line)
    
    process.stdout.close()
    process.wait()
    
    success = process.returncode == 0
    
    if not success:
        log(f"Video classification failed with exit code {process.returncode}")
    
    return success