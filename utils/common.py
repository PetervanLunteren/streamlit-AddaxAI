
"""
AddaxAI Common Utilities

Shared utility functions used across the AddaxAI application including:
- Session state management and caching
- File I/O operations for JSON configuration
- Animal name generation for unique identifiers
- Model metadata fetching and management
- Logging wrapper functions
"""

import os
import json
import streamlit as st
from appdirs import user_config_dir
import string
import re
import random
from datetime import datetime
import requests
import pandas as pd


from utils.config import *
from components.ui_helpers import info_box

# MAP_FILE_PATH will be set from session state when available
# This prevents errors when importing outside of Streamlit context
try:
    MAP_FILE_PATH = st.session_state["shared"]["MAP_FILE_PATH"]
except (KeyError, AttributeError):
    MAP_FILE_PATH = None

# Load current AddaxAI version from file
with open(os.path.join(ADDAXAI_ROOT, 'assets', 'version.txt'), 'r') as file:
    
    current_AA_version = file.read().strip()

def unique_animal_string():
    """
    Generate a unique identifier string using UUID.

    Returns:
        str: UUID string (e.g., "a1b2c3d4-e5f6-7890-abcd-ef1234567890")

    Used for creating unique run IDs and deployment identifiers.
    """
    import uuid
    return str(uuid.uuid4())



def default_converter(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(
        f"Object of type {obj.__class__.__name__} is not JSON serializable")


def clear_vars(section):
    """
    Clear only temporary variables in session state for a specific section.
    This preserves persistent variables like process_queue.
    """
    if section in st.session_state:
        st.session_state[section] = {}


def init_session_state(section):
    """
    Initialize session state for a specific section if it doesn't exist.
    """
    if section not in st.session_state:
        st.session_state[section] = {}


def get_session_var(section, var_name, default=None):
    """
    Get a variable from session state for a specific section.
    """
    init_session_state(section)
    return st.session_state[section].get(var_name, default)


def set_session_var(section, var_name, value):
    """
    Set a variable in session state for a specific section.
    """
    init_session_state(section)
    st.session_state[section][var_name] = value


def update_session_vars(section, updates):
    """
    Update multiple variables in session state for a specific section.
    """
    init_session_state(section)
    st.session_state[section].update(updates)


def replace_vars(section, new_vars):
    vars_file = os.path.join(ADDAXAI_ROOT, "config", f"{section}.json")

    # Overwrite with only the new updates
    with open(vars_file, "w", encoding="utf-8") as file:
        json.dump(new_vars, file, indent=2, default=default_converter)


def update_vars(section, updates):
    """
    Update specific values in a configuration JSON file.
    
    Args:
        section (str): Configuration section name (becomes {section}.json)
        updates (dict): Key-value pairs to update in the configuration
    
    Creates the config file if it doesn't exist. Merges updates with existing data.
    """
    vars_file = os.path.join(ADDAXAI_ROOT, "config", f"{section}.json")
    if not os.path.exists(vars_file):
        with open(vars_file, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

    # read section vars
    with open(vars_file, "r", encoding="utf-8") as f:
        section_vars = json.load(f)

    # update
    section_vars.update(updates)

    # Use `default=default_converter` to catch any lingering datetime objects
    with open(vars_file, "w") as file:
        json.dump(section_vars, file, indent=2, default=default_converter)


def load_map():
    """Reads the data from the JSON file and returns it as a dictionary."""
    global MAP_FILE_PATH
    
    # Get MAP_FILE_PATH from session state if not already set
    if MAP_FILE_PATH is None:
        try:
            MAP_FILE_PATH = st.session_state["shared"]["MAP_FILE_PATH"]
        except (KeyError, AttributeError):
            # Fallback path if session state not available
            from appdirs import user_config_dir
            MAP_FILE_PATH = os.path.join(user_config_dir("AddaxAI"), "map.json")

    # Load full settings or initialize
    try:
        if os.path.exists(MAP_FILE_PATH):
            with open(MAP_FILE_PATH, "r", encoding="utf-8") as f:
                settings = json.load(f)
        else:
            settings = {}
    except (json.JSONDecodeError, IOError):
        settings = {}

    return settings, MAP_FILE_PATH


def load_vars(section):
    # if not exist, create empty vars file
    vars_file = os.path.join(ADDAXAI_ROOT, "config", f"{section}.json")
    if not os.path.exists(vars_file):
        with open(vars_file, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

    # read section vars
    with open(vars_file, "r", encoding="utf-8") as f:
        section_vars = json.load(f)

    return section_vars
    # return {var: section_vars.get(var, None) for var in requested_vars}.values()


def load_lang_txts():
    txts_fpath = os.path.join(ADDAXAI_ROOT, "assets", "language", "lang.json")
    with open(txts_fpath, "r", encoding="utf-8") as file:
        txts = json.load(file)
    return txts


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION SETTINGS (config/settings.json)
# ═══════════════════════════════════════════════════════════════════════════════

APP_SETTINGS_FILE = os.path.join(ADDAXAI_ROOT, "config", "settings.json")
DEFAULT_APP_SETTINGS = {
    "data_import": {
        "detection_conf_threshold": 0.5
    },
    "events": {
        "time_gap_seconds": 60
    }
}


def load_app_settings():
    """
    Load application-level settings from config/settings.json.
    Creates the file with defaults if missing or invalid.
    """
    os.makedirs(os.path.dirname(APP_SETTINGS_FILE), exist_ok=True)

    if not os.path.exists(APP_SETTINGS_FILE):
        save_app_settings(DEFAULT_APP_SETTINGS)
        return DEFAULT_APP_SETTINGS.copy()

    try:
        with open(APP_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("settings.json must contain an object")
        return data
    except (json.JSONDecodeError, ValueError):
        save_app_settings(DEFAULT_APP_SETTINGS)
        return DEFAULT_APP_SETTINGS.copy()


def save_app_settings(settings_dict):
    """
    Persist application-level settings to config/settings.json.
    """
    os.makedirs(os.path.dirname(APP_SETTINGS_FILE), exist_ok=True)
    with open(APP_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL METADATA MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_latest_model_info():
    """
    Downloads the latest model metadata from GitHub and creates folder structure
    for new models. Shows notifications for any new models found.
    
    This function:
    1. Downloads model_meta.json from the GitHub repository
    2. Compares models in metadata vs existing model directories
    3. Creates folders and variables.json for any new models
    4. Shows info_box notifications with model info popover for new models
    
    Called during app startup to keep model metadata current.
    """
    import streamlit as st
    
    from utils.config import log
    log(f"EXECUTED: fetch_latest_model_info()")
    
    # Define paths using the global ADDAXAI_FILES_ST from config
    model_meta_url = "https://raw.githubusercontent.com/PetervanLunteren/streamlit-AddaxAI/refs/heads/main/assets/model_meta/model_meta.json"
    model_meta_local = os.path.join(ADDAXAI_ROOT, "assets", "model_meta", "model_meta.json")
    models_dir = os.path.join(ADDAXAI_ROOT, "models")
    
    try:
        log("Starting model metadata download...")
        # Download latest model metadata with reasonable timeout
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Accept-Encoding": "*",
            "Connection": "keep-alive"
        }
        
        log(f"Downloading from: {model_meta_url}")
        response = requests.get(model_meta_url, timeout=10, headers=headers)
        log(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            # Save updated model metadata
            os.makedirs(os.path.dirname(model_meta_local), exist_ok=True)
            with open(model_meta_local, 'wb') as file:
                file.write(response.content)
            
            log("Updated model_meta.json successfully.")
            
            # Load the downloaded metadata
            with open(model_meta_local, 'r') as f:
                model_meta = json.load(f)
            
            # Check if this is first startup (no models exist at all)
            is_first_startup = True
            for model_type in ["det", "cls"]:
                type_dir = os.path.join(models_dir, model_type)
                if os.path.exists(type_dir):
                    existing_models = [d for d in os.listdir(type_dir) 
                                     if os.path.isdir(os.path.join(type_dir, d))]
                    if existing_models:
                        is_first_startup = False
                        break
            
            if is_first_startup:
                log("First startup detected - creating all model directories without notifications")
            
            # Process both detection and classification models
            for model_type in ["det", "cls"]:
                if model_type in model_meta:
                    model_dicts = model_meta[model_type]
                    all_models = list(model_dicts.keys())
                    
                    # Get existing model directories
                    type_dir = os.path.join(models_dir, model_type)
                    os.makedirs(type_dir, exist_ok=True)
                    
                    existing_models = []
                    if os.path.exists(type_dir):
                        existing_models = [d for d in os.listdir(type_dir) 
                                         if os.path.isdir(os.path.join(type_dir, d))]
                    
                    # Find new models (in metadata but not in filesystem)
                    new_models = [model for model in all_models if model not in existing_models]
                    
                    # Create directories for new models
                    for model_id in new_models:
                        model_info = model_dicts[model_id]
                        
                        # Create model directory
                        model_dir = os.path.join(type_dir, model_id)
                        os.makedirs(model_dir, exist_ok=True)
                        
                        # Create variables.json file with model metadata
                        variables_file = os.path.join(model_dir, "variables.json")
                        with open(variables_file, 'w') as f:
                            json.dump(model_info, f, indent=4)
                        
                        # Download taxonomy.csv for classification models
                        if model_type == "cls":
                            try:
                                taxonomy_url = f"https://huggingface.co/Addax-Data-Science/{model_id}/resolve/main/taxonomy.csv?download=true"
                                taxonomy_file = os.path.join(model_dir, "taxonomy.csv")
                                
                                log(f"Downloading taxonomy.csv for {model_id} from: {taxonomy_url}")
                                # Don't use gzip encoding in headers to avoid compression issues
                                taxon_headers = headers.copy()
                                taxon_headers["Accept-Encoding"] = "identity"  # Request uncompressed content
                                
                                taxon_response = requests.get(taxonomy_url, timeout=15, headers=taxon_headers)
                                
                                if taxon_response.status_code == 200:
                                    # Check if content might be compressed by looking at the raw bytes
                                    content = taxon_response.content
                                    
                                    # Try to detect if it's gzipped content
                                    if content.startswith(b'\x1f\x8b'):  # Gzip magic number
                                        import gzip
                                        content = gzip.decompress(content)
                                        log(f"Decompressed gzipped content for {model_id}")
                                    
                                    # Decode to text with UTF-8
                                    text_content = content.decode('utf-8')
                                    
                                    with open(taxonomy_file, 'w', encoding='utf-8', newline='') as f:
                                        f.write(text_content)
                                    log(f"Downloaded taxonomy.csv for {model_id}")
                                else:
                                    log(f"Failed to download taxonomy.csv for {model_id}. Status: {taxon_response.status_code}")
                            except Exception as e:
                                log(f"Error downloading taxonomy.csv for {model_id}: {e}")
                        
                        log(f"Created directory and variables.json for new {model_type} model: {model_id}")
                        
                        # Show notification for new model (only if not first startup)
                        if not is_first_startup:
                            friendly_name = model_info.get('friendly_name', model_id)
                            model_type_name = "species identification model" if model_type == "cls" else "detection"
                            
                            # Create info box
                            info_box(
                                msg = f"New {model_type_name.lower()} model available: {friendly_name}",
                                title="New model added",
                                icon=":material/new_releases:"
                            )
                                

            
        else:
            log(f"Failed to download model metadata. Status code: {response.status_code}")
            
    except requests.exceptions.Timeout:
        log("Request timed out. Model metadata update stopped.")
    except Exception as e:
        log(f"Could not update model metadata: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK ERROR LOGGING WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def logged_callback(func):
    """
    Decorator to wrap Streamlit callbacks with error logging.
    
    This ensures that any exceptions in callbacks are logged to the file
    before Streamlit catches and displays them in the UI.
    
    Usage:
        @logged_callback
        def on_button_click():
            # Your callback code here
            pass
    """
    import functools
    import traceback
    from utils.config import log
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log(f"ERROR in callback {func.__name__}: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            # Re-raise so Streamlit still shows the error in UI
            raise
    
    return wrapper

# Simple filesystem checks for model and environment availability
# Lightweight operations that run on every check - no caching complexity

def check_model_availability(model_type, model_id, model_meta):
    """
    Check model and environment availability.
    Simple filesystem checks - no caching complexity.
    
    Args: 
        model_type: 'cls' or 'det'
        model_id: Model identifier
        model_meta: Model metadata dictionary
    
    Returns:
        dict: {
            'env_exists': bool,
            'model_exists': bool,
            'env_name': str,
            'model_fname': str,
            'friendly_name': str
        }
    """
    model_info = model_meta[model_type][model_id]
    env_name = model_info["env"]
    model_fname = model_info["model_fname"]
    friendly_name = model_info["friendly_name"]
    
    env_path = os.path.join(ADDAXAI_ROOT, "envs", f"env-{env_name}")
    model_path = os.path.join(ADDAXAI_ROOT, "models", model_type, model_id, model_fname)
    
    return {
        'env_exists': os.path.exists(env_path),
        'model_exists': os.path.exists(model_path),
        'env_name': env_name,
        'model_fname': model_fname,
        'friendly_name': friendly_name
    }
