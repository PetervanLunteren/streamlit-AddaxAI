"""
AddaxAI Streamlit Application - Main Entry Point

This is the main entry point for the AddaxAI wildlife camera trap analysis platform.
The application uses an optimized startup/rerun pattern to minimize I/O operations:
- On startup (empty session_state): Load all files, initialize caches  
- On reruns: Use cached data from session_state for maximum performance

Architecture:
- Three-tier variable storage: session_state (temp) → vars/*.json (persistent) → ~/.config/AddaxAI/map.json (global)
- Write-through caching: Changes update both persistent storage and session_state 
- Startup detection: if st.session_state == {} triggers initialization

Run with micromamba:
./bin/macos/micromamba run -p ./envs/env-addaxai-base streamlit run main.py

How to pip install into an environment:
./bin/macos/micromamba run -p ./envs/env-addaxai-base pip install [package-name]
./bin/macos/micromamba run -p ./envs/env-addaxai-base pip install streamlit_image_zoom

TODOs FOR NOW:
- continue with the data brwoser page - fix the modal view for image details
- make the detections page for images too. 
- update to the newset MD version and make sure to adjust:
    - the way you get exif data: --include_exif_tags "datetimeoriginal,gpsinfo" (see https://github.com/agentmorris/MegaDetector/pull/193#issuecomment-3347432732)
    - make sure to check how I run SpeciesNet now: https://github.com/agentmorris/MegaDetector/pull/193#issuecomment-3347432732
    - the way the pbars get GPU / CPU / MPS info. (see https://github.com/agentmorris/MegaDetector/pull/187)
    

TODOs FOR LATER:
- https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/classification_postprocessing.py
- https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- material icons must be offline, but I can only do that at the end when I know which icons I'm using.
- if no CLS is selected, it should skip species selection and the button should be add to queue. 
- the license warning should be for all models, not just the yolov11 models. Where? During the installation wizard probabaly. The user has to agree to the licenses when installing addaxai.
"""

# Standard library imports
import streamlit as st
import sys
import os
import time
import json

# Third-party imports
from streamlit_lottie import st_lottie
from appdirs import user_config_dir, user_cache_dir
from st_modal import Modal
import pandas as pd
 

# Local imports - global config must be imported before anything else
from utils.config import *

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION & SETUP (runs on every request)
# ═══════════════════════════════════════════════════════════════════════════════

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)

# Streamlit config loaded from .streamlit/config.toml 

# Configure Streamlit page settings
st.set_page_config(
    initial_sidebar_state="auto", 
    page_icon=os.path.join(ADDAXAI_ROOT, "assets", "images", "logo_square.png"), 
    page_title="AddaxAI"
)

# Material Icons must be injected on every rerun for stepper components
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
""", unsafe_allow_html=True)

# Load and inject custom CSS (only on startup for performance)
with open(os.path.join(ADDAXAI_ROOT, "assets", "css", "styles.css"), "r") as f:
    css_content = f.read()
st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP INITIALIZATION (runs only when session_state is empty)
# ═══════════════════════════════════════════════════════════════════════════════

# Detect app startup: empty session_state means this is a fresh session
if st.session_state == {}:

    # ─────────────────────────────────────────────────────────────────────────
    # Display loading overlay during startup
    # ─────────────────────────────────────────────────────────────────────────
    
    # Import BlockingLoader class
    from components.ui_helpers import BlockingLoader
    
    # Create and open the blocking loader
    loader = BlockingLoader()
    loader.open("Loading AddaxAI Streamlit app...")
    
    time.sleep(0.5)  # Simulate loading time
    
    # ─────────────────────────────────────────────────────────────────────────
    # Initialize session state and directory structure
    # ─────────────────────────────────────────────────────────────────────────
    
    loader.update_text("Initializing session state...")
    
    time.sleep(0.5)  # Simulate loading time
    
    # Initialize shared session state container for cross-tool temporary variables
    st.session_state["shared"] = {}
    
    # Create config directories at startup (appdirs may not be available in all environments)
    CONFIG_DIR = user_config_dir("AddaxAI")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    MAP_FILE_PATH = os.path.join(CONFIG_DIR, "map.json")
    
    # Store paths in session state for tool access
    st.session_state["shared"] = {
        "CONFIG_DIR": CONFIG_DIR,
        "MAP_FILE_PATH": MAP_FILE_PATH
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load utility modules and initialize settings cache
    # ─────────────────────────────────────────────────────────────────────────
    
    loader.update_text("Loading utility modules...")
    
    time.sleep(0.5)  # Simulate loading time
    
    # Import utils now that shared session state exists (modules depend on it)
    from utils.common import load_lang_txts, load_vars, update_vars, set_session_var, get_session_var, fetch_latest_model_info
    from components import print_widget_label
    from utils.analysis_utils import load_known_projects, load_model_metadata
    

    
    # ─────────────────────────────────────────────────────────────────────────
    # Initialize global configuration files
    # ─────────────────────────────────────────────────────────────────────────
    
    # Initialize global map.json if it doesn't exist
    # (stores project definitions, camera locations, deployment history)
    if not os.path.exists(MAP_FILE_PATH):

        # Create empty projects structure
        map = {
            "projects": {}
        }

        with open(MAP_FILE_PATH, "w") as f:
            json.dump(map, f, indent=2)

    # Initialize general_settings.json if it doesn't exist
    general_settings_file = os.path.join(ADDAXAI_ROOT, "config", f"general_settings.json")
    if not os.path.exists(general_settings_file):
        
        # Create config directory if needed
        os.makedirs(os.path.dirname(general_settings_file), exist_ok=True)
        
        # Create default settings
        general_settings = {
            "lang": "en",
            "mode": 1,  # 0: simple mode, 1: advanced mode
            "selected_projectID": None,
            "INFERENCE_MIN_CONF_THRES_DETECTION": 0.1
        }
        with open(general_settings_file, "w") as f:
            json.dump(general_settings, f, indent=2)

    # Initialize explore_results.json if it doesn't exist or is empty
    explore_results_file = os.path.join(ADDAXAI_ROOT, "config", f"explore_results.json")
    if not os.path.exists(explore_results_file) or os.path.getsize(explore_results_file) == 0:
        # Create default filter settings
        explore_results_settings = {
            "aggrid_settings": {
                "date_start": None,
                "date_end": None,
                "det_conf_min": 0.0,
                "det_conf_max": 1.0,
                "cls_conf_min": 0.0,
                "cls_conf_max": 1.0,
                "include_unclassified": True,
                "image_size": "medium"
            }
        }
        with open(explore_results_file, "w") as f:
            json.dump(explore_results_settings, f, indent=2)

    # Load and cache general settings to avoid file reads on reruns
    general_settings_vars = load_vars(section = "general_settings")
    lang = general_settings_vars["lang"]
    mode = general_settings_vars["mode"]
    selected_projectID = general_settings_vars.get("selected_projectID", None)
    
    set_session_var("shared", "lang", lang)
    set_session_var("shared", "mode", mode)
    set_session_var("shared", "selected_projectID", selected_projectID)

    # ─────────────────────────────────────────────────────────────────────────
    # Load and cache expensive resources (language, models, UI assets)
    # ─────────────────────────────────────────────────────────────────────────
    
    loader.update_text("Loading language and model data...")
    
    time.sleep(0.5)  # Simulate loading time
    
    # Load and cache language texts to avoid file I/O on reruns
    if not st.session_state.get("txts"):
        full_txts = load_lang_txts()
        # Store only current language's texts in flattened structure for efficiency
        st.session_state["txts"] = {key: value[lang] for key, value in full_txts.items()}

    # Load and cache AI model metadata (large JSON file)
    if not st.session_state.get("model_meta"):
        st.session_state["model_meta"] = load_model_metadata()

    # Download latest model metadata and create folders for new models
    fetch_latest_model_info()
    
    # Reload model metadata after download to ensure session state has latest data
    st.session_state["model_meta"] = load_model_metadata()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load all detection results into dataframe for analysis and visualization
    # ─────────────────────────────────────────────────────────────────────────
    
    loader.update_text("Loading detection results...")
    
    time.sleep(0.5)  # Simulate loading time
    
    try:
        from utils.data_loading import load_detection_results_dataframe
        results_df = load_detection_results_dataframe()
        st.session_state["results_detections"] = results_df
        
        if len(results_df) > 0:
            log(f"Loaded {len(results_df)} detections from {len(results_df['run_id'].unique())} runs")
        else:
            log("No detection results found in completed runs")
            
    except Exception as e:
        error_msg = f"Failed to load detection results: {str(e)}"
        log(error_msg)
        st.warning(error_msg)
        # Create empty dataframe as fallback
        st.session_state["results_detections"] = pd.DataFrame()
    
    # Archive previous session log and create fresh log (only on startup)
    log_fpath = os.path.join(ADDAXAI_ROOT, "assets", "logs", "log.txt")
    previous_sessions_dir = os.path.join(ADDAXAI_ROOT, "assets", "logs", "previous_sessions")
    
    try:
        # Create previous_sessions directory if it doesn't exist
        os.makedirs(previous_sessions_dir, exist_ok=True)
        
        # If there's an existing log file, archive it with timestamp
        if os.path.exists(log_fpath) and os.path.getsize(log_fpath) > 0:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_log_path = os.path.join(previous_sessions_dir, f"log_{timestamp}.txt")
            
            # Move the existing log to archive
            import shutil
            shutil.move(log_fpath, archived_log_path)
            
        # Create a fresh log file for this session
        with open(log_fpath, "w", encoding="utf-8") as file:
            from datetime import datetime
            session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"AddaxAI Streamlit App Log - Session Started: {session_start}\n")
            file.write("=" * 60 + "\n")
            file.write("This log contains all output from the current session.\n")
            file.write("Previous sessions are archived in: assets/logs/previous_sessions/\n")
            file.write("=" * 60 + "\n\n")
        
        # Use simple log() function from config.py for reliable logging
        from utils.config import log
        log("Logging system initialized using simple log() function")
        log("Testing log function - this should appear in both console and log file")
            
    except PermissionError:
        print(f"Permission denied when accessing {log_fpath}. Could not setup logging.")
    except Exception as e:
        print(f"Error setting up logging: {e}")
    
    # Close the blocking loader
    loader.close()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION (runs on startup and every rerun)
# ═══════════════════════════════════════════════════════════════════════════════

# Get cached values from session state (efficient - no file reads on rerun)
mode = st.session_state["shared"]["mode"]
lang = st.session_state["shared"]["lang"]
txts = st.session_state["txts"]
model_meta = st.session_state["model_meta"]

# ─────────────────────────────────────────────────────────────────────────
# Configure page navigation based on mode
# ─────────────────────────────────────────────────────────────────────────

# Display application logo
st.logo(os.path.join(ADDAXAI_ROOT, "assets", "images", "logo.png"), size="large")

# Create navigation based on selected mode
if mode == 0:  # Simple mode - single analysis tool only
    analysis_quick_page = st.Page(
        os.path.join("pages", "analysis_quick.py"), title="Add data", icon=":material/rocket_launch:")
    pg = st.navigation([analysis_quick_page])
    
elif mode == 1:  # Advanced mode - full toolkit
    analysis_advanced_page = st.Page(
        os.path.join("pages", "analysis_advanced.py"), title="Add data", icon=":material/add:")
    remove_duplicates_page = st.Page(
        os.path.join("pages", "remove_duplicates.py"), title="Remove duplicates", icon=":material/reset_image:")
    human_verification_page = st.Page(
        os.path.join("pages", "human_verification.py"), title="Human verification", icon=":material/checklist:")
    depth_estimation_page = st.Page(
        os.path.join("pages", "depth_estimation.py"), title="Depth estimation", icon=":material/square_foot:")
    explore_results_page = st.Page(
        os.path.join("pages", "explore_results.py"), title="Explore results", icon=":material/bar_chart_4_bars:")
    data_browser_page = st.Page(
        os.path.join("pages", "data_browser.py"), title="Data browser", icon=":material/grid_on:")
    post_processing_page = st.Page(
        os.path.join("pages", "post_processing.py"), title=txts["postprocess_txt"], icon=":material/stylus_note:")
    camera_management_page = st.Page(
        os.path.join("pages", "camera_management.py"), title="Camera management", icon=":material/photo_camera:")
    settings_page = st.Page(
        os.path.join("pages", "settings.py"), title=txts["settings_txt"], icon=":material/settings:")
    pg = st.navigation([analysis_advanced_page, remove_duplicates_page, human_verification_page,
                       depth_estimation_page, explore_results_page, data_browser_page, post_processing_page, camera_management_page, settings_page])

# Run the selected page
pg.run()

# ─────────────────────────────────────────────────────────────────────────
# Sidebar controls (mode and project selection)
# ─────────────────────────────────────────────────────────────────────────

# Import utils and components
from utils.common import load_lang_txts, load_vars, update_vars, set_session_var, get_session_var, logged_callback
from components import print_widget_label
from utils.analysis_utils import load_known_projects, load_model_metadata, add_project_modal

# Mode selection options
mode_options = {
    0: "Simple",
    1: "Advanced",
}


@logged_callback
def on_mode_change():
    """Write-through callback: updates both persistent file and session state cache"""
    if "mode_selection" in st.session_state:
        mode_selection = st.session_state["mode_selection"]
        
        # Write to persistent file for next session
        update_vars("general_settings", {"mode": mode_selection})
        
        # Update session state for immediate use this session
        set_session_var("shared", "mode", mode_selection)

# Mode selector in sidebar
print_widget_label("Mode", help_text="help text", sidebar=True)
mode_selected = st.sidebar.segmented_control( 
    "Mode",
    options=mode_options.keys(),
    format_func=mode_options.get,
    selection_mode="single",
    label_visibility="collapsed",
    help=txts["mode_explanation_txt"],
    key="mode_selection",
    on_change=on_mode_change,
    width = "stretch",
    default=mode)

@logged_callback
def on_project_change():
    """Write-through callback: updates both persistent file and session state cache"""
    if "project_selection_sidebar" in st.session_state:
        project_selection = st.session_state["project_selection_sidebar"]
        
        # Write to persistent file for next session
        update_vars("general_settings", {"selected_projectID": project_selection})
        
        # Update session state for immediate use this session
        set_session_var("shared", "selected_projectID", project_selection)
        

# Project selector (only shown in advanced mode)
if mode == 1:  # Advanced mode requires project context

    # Load existing projects from global map.json
    projects, selected_projectID = load_known_projects()

    # Show project selector only if projects exist
    if not projects == {}:
        
        options = list(projects.keys())
        # Find index of currently selected project
        selected_index = options.index(selected_projectID) if selected_projectID in options else 0

        # Project selector in sidebar with columns for selectbox and new project button
        print_widget_label("Project", help_text="Selected project applies to all tools.", sidebar=True)
        col1, col2 = st.sidebar.columns([3, 1])
        
        with col1:
            selected_projectID = st.selectbox(
                "Project",
                options=options,
                index=selected_index,
                label_visibility="collapsed",
                key="project_selection_sidebar",
                on_change=on_project_change
            )
            
            # Ensure the currently selected project is stored in session state and persistent storage
            # (handles case where selectbox shows a project but session state is None)
            if selected_projectID and get_session_var("shared", "selected_projectID", None) != selected_projectID:
                set_session_var("shared", "selected_projectID", selected_projectID)
                # Also save to persistent storage so it survives restarts
                update_vars("general_settings", {"selected_projectID": selected_projectID})
        
        with col2:
            if st.button(":material/add_circle:", help="Add new project", width="stretch"):
                set_session_var("analyse_advanced", "show_modal_add_project", True)
                st.rerun()
    
    else:
        # No projects exist - show button to create first project
        print_widget_label("Project", help_text="Create your first project to get started.", sidebar=True)
        if st.sidebar.button(":material/add_circle: Create first project", width="stretch"):
            set_session_var("analyse_advanced", "show_modal_add_project", True)
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# Modal handling for project creation (triggered from sidebar)
# ─────────────────────────────────────────────────────────────────────────

# Modal for adding new project - only create when needed
if get_session_var("analyse_advanced", "show_modal_add_project", False):
    modal_add_project = Modal(
        title="#### Describe new project", key="add_project", show_close_button=False)
    with modal_add_project.container():
        add_project_modal()