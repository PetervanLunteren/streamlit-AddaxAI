"""
AddaxAI Global Configuration

This module defines global constants and utility functions used across the entire project.
It establishes the core directory structure and platform-specific paths that all other
modules depend on.

Key Features:
- Cross-platform compatibility (Windows, Linux, macOS)
- Centralized path management for the AddaxAI ecosystem
- File extension definitions for media processing
- Unified logging function

Important: This file is imported by main.py before session_state exists, so it cannot
depend on Streamlit session state or appdirs (which may not be available in all conda envs).
"""

import os
import platform

# Note: appdirs import is commented out because this config file must work in multiple
# conda environments where appdirs may not be installed. Directory creation using
# appdirs is handled in main.py after startup detection.
# from appdirs import user_cache_dir, user_config_dir

# ═══════════════════════════════════════════════════════════════════════════════
# PLATFORM DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_os_name():
    """
    Detect the operating system and return a standardized name.
    
    Returns:
        str: 'windows', 'linux', or 'macos'
    
    Used for:
        - Selecting platform-specific binaries (micromamba)
        - OS-specific path handling
        - Platform-dependent tool configurations
    """
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Linux":
        return "linux"
    elif system == "Darwin":  # macOS
        return "macos"
    else:
        # Fallback for unknown platforms
        return "linux"

# Global OS identifier used throughout the application
OS_NAME = get_os_name()

# ═══════════════════════════════════════════════════════════════════════════════
# CORE DIRECTORY STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

# Root AddaxAI installation directory
# Path calculation: utils/config.py -> utils -> streamlit-AddaxAI -> AddaxAI -> AddaxAI_files
ADDAXAI_FILES = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

# Streamlit application directory (temporary during migration from tkinter)
# Structure: AddaxAI_files/AddaxAI/streamlit-AddaxAI/
ADDAXAI_FILES_ST = os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI")

# Platform-specific micromamba binary for conda environment management
# Used by tools to create and activate conda environments for different AI models
MICROMAMBA = os.path.join(ADDAXAI_FILES_ST, "bin", OS_NAME, "micromamba")

# folder for storing temporary files, like inomplete downloads
TEMP_DIR = os.path.join(ADDAXAI_FILES_ST, "assets", "temp")

# ═══════════════════════════════════════════════════════════════════════════════
# MEDIA FILE PROCESSING CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Supported video file extensions for camera trap analysis
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mpeg', '.mpg', '.mov', '.mkv')

# Supported image file extensions for camera trap analysis
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.gif', '.png', '.tif', '.tiff', '.bmp')

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def log(msg):
    """
    Unified logging function that writes to both file and console.
    
    Args:
        msg (str): Message to log
    
    Behavior:
        - Appends message to assets/logs/log.txt
        - Prints message to console (stdout)
        - Automatically adds newline character
    
    Note: This is a simple logging function. The main.py file handles log rotation
    and archival of previous sessions.
    """
    with open(os.path.join(ADDAXAI_FILES_ST, 'assets', 'logs', 'log.txt'), 'a') as f:
        f.write(f"{msg}\n")
    print(msg)

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT MODEL CONSTANTS  
# ═══════════════════════════════════════════════════════════════════════════════

# Default detection model ID
DEFAULT_DETECTION_MODEL = "MD5A-0-0"

# Default classification model ID (NONE means no classification)
DEFAULT_CLASSIFICATION_MODEL = "NONE"
