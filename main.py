
# WAAR WAS IK? 
# TODO: 
# - daarna testen met een fresh install en schone conda env
# - dan testen op windows, werkt het ook zonder in de folder van addaxai te zitten?
# - dan opschonen, commenten, teksten, readme, mds, etc. 


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

CLI to run:
cd /Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI && conda activate env-streamlit-addaxai && streamlit run main.py

TODOs:
- https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/classification_postprocessing.py
- https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- RDE
- material icons must be offline, but I can only do that at the end when I know which icons I'm using.
- make it accessible for all models and envs
- also save the image or video that had the min_datetime, so that we can calculate the diff every time we need it "deployment_start_file". Then it can read the exif from the path. No need to read all exifs of all images.  searc h for deployment_start_file, deployment_start_datetime
- add SpeciesNet
- add video processing 
- reformat obj names, function names, and variable names, and classes, etc
- add GPU / CPU icon to pbars 
- do the loading squirell in a modal too, to you cant click anything else while loading
- open a model when the folder_selection.py script is done with an info box saying a new window has opened. It the model will only close once the tkinter script is done sucessfully.
- if NONE cls model is selected, then dont show the cls pbars. 
- The pbars for progress should be taken care of. They should be updated, the texts proper,etc
- if no CLS is selected, it should skip species selection and the button should be add to queue. 
- the license warning should be for all models, not just the yolov11 models. Where?
- delete the all_classes from the variables json? It should probabaly take it from the taxon csv right? That is redundant and error prone.
- open a modal that tells the user to select a folder in a new window. That wat the user cannot click anything else while the folder selection is happening.
- browse folder must open last chosen folder, not the root of the filesystem
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
 

# Local imports - global config must be imported before anything else
from utils.config import *

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION & SETUP (runs on every request)
# ═══════════════════════════════════════════════════════════════════════════════

# Force all output to be unbuffered for real-time logging
sys.stdout.reconfigure(line_buffering=True)
# Note: stderr redirection moved to after TeeOutput setup to ensure errors are logged

# Streamlit config is loaded from .streamlit/config.toml (standard location) 

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
    # Display loading animation and status during startup
    # ─────────────────────────────────────────────────────────────────────────
    
    # Load Lottie animation for startup loading screen
    lottie_animation_fpath = os.path.join(ADDAXAI_ROOT, "assets", "loaders", "squirrel.json")
    with open(lottie_animation_fpath, "r") as f:
        lottie_animation = json.load(f)

    _, col_animation, _ = st.columns([1, 2, 1])
    with col_animation:
        # Create a container for the animation that we can clear later
        animation_container = st.empty()
        
        with animation_container:
            st_lottie(
                lottie_animation,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                key="lottie_animation"
            )

    status_placeholder = st.empty()
    status_placeholder.status("Loading AddaxAI Streamlit app...")
    time.sleep(0.5)  # Simulate loading time
    
    # ─────────────────────────────────────────────────────────────────────────
    # Initialize session state and directory structure
    # ─────────────────────────────────────────────────────────────────────────
    
    # Initialize shared session state container for cross-tool temporary variables
    st.session_state["shared"] = {}
    
    # Create config and temp directories (done at startup since appdirs 
    # may not be available in different conda environments used by tools)
    CONFIG_DIR = user_config_dir("AddaxAI")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    MAP_FILE_PATH = os.path.join(CONFIG_DIR, "map.json")
    
    # Store paths in session state for access by all tools
    st.session_state["shared"] = {
        "CONFIG_DIR": CONFIG_DIR,
        "MAP_FILE_PATH": MAP_FILE_PATH
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load utility modules and initialize settings cache
    # ─────────────────────────────────────────────────────────────────────────
    
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
            "selected_projectID": None
        }
        with open(general_settings_file, "w") as f:
            json.dump(general_settings, f, indent=2)

    # Load general settings from file and cache in session state
    # (avoids file reads on every rerun for lang/mode)
    general_settings_vars = load_vars(section = "general_settings")
    lang = general_settings_vars["lang"]
    mode = general_settings_vars["mode"]
    
    set_session_var("shared", "lang", lang)
    set_session_var("shared", "mode", mode)

    # ─────────────────────────────────────────────────────────────────────────
    # Load and cache expensive resources (language, models, UI assets)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Load language texts and cache in session state (avoids file I/O on reruns)
    if not st.session_state.get("txts"):
        full_txts = load_lang_txts()
        # Store only current language's texts in flattened structure for efficiency
        st.session_state["txts"] = {key: value[lang] for key, value in full_txts.items()}

    # Load AI model metadata and cache in session state (large JSON file)
    if not st.session_state.get("model_meta"):
        st.session_state["model_meta"] = load_model_metadata()

    # Download latest model metadata and create folder structure for new models
    # This will show notifications for any new models that aren't in local filesystem
    fetch_latest_model_info()
    
    # Reload model metadata after download to ensure session state has latest data
    st.session_state["model_meta"] = load_model_metadata()
    
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
    
    # Clean up loading UI elements
    status_placeholder.empty()
    animation_container.empty()

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
        os.path.join("pages", "analysis_quick.py"), title=txts["analyse_txt"], icon=":material/rocket_launch:")
    pg = st.navigation([analysis_quick_page])
    
elif mode == 1:  # Advanced mode - full toolkit
    analysis_advanced_page = st.Page(
        os.path.join("pages", "analysis_advanced.py"), title=txts["analyse_txt"], icon=":material/add:")
    remove_duplicates_page = st.Page(
        os.path.join("pages", "remove_duplicates.py"), title="Remove duplicates", icon=":material/reset_image:")
    human_verification_page = st.Page(
        os.path.join("pages", "human_verification.py"), title="Human verification", icon=":material/checklist:")
    depth_estimation_page = st.Page(
        os.path.join("pages", "depth_estimation.py"), title="Depth estimation", icon=":material/square_foot:")
    explore_results_page = st.Page(
        os.path.join("pages", "explore_results.py"), title="Explore results", icon=":material/bar_chart_4_bars:")
    post_processing_page = st.Page(
        os.path.join("pages", "post_processing.py"), title=txts["postprocess_txt"], icon=":material/stylus_note:")
    camera_management_page = st.Page(
        os.path.join("pages", "camera_management.py"), title="Camera management", icon=":material/photo_camera:")
    settings_page = st.Page(
        os.path.join("pages", "settings.py"), title=txts["settings_txt"], icon=":material/settings:")
    pg = st.navigation([analysis_advanced_page, remove_duplicates_page, human_verification_page,
                       depth_estimation_page, explore_results_page, post_processing_page, camera_management_page, settings_page])

# Run the selected page
pg.run()

# ─────────────────────────────────────────────────────────────────────────
# Sidebar controls (mode and project selection)
# ─────────────────────────────────────────────────────────────────────────

# Import utils and components
from utils.common import load_lang_txts, load_vars, update_vars, set_session_var, get_session_var, logged_callback
from components import print_widget_label
from utils.analysis_utils import load_known_projects, load_model_metadata

# Mode selection options
mode_options = {
    0: "Simple",
    1: "Advanced",
}


@logged_callback
def on_mode_change():
    """Write-through callback: updates both persistent file and session state cache"""
    if "mode_selection" in st.session_state:
        mode_selection = st.session_state["mode_selection"]  # Intentional error for testing
        
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
    default=mode)

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

        # Project selector in sidebar
        print_widget_label("Project", help_text="The project selected here is the one that all tools will work with. If you have a new project, you can add it at + add deployment when you want to process the first batch data.", sidebar=True)
        selected_projectID = st.sidebar.selectbox(
            "Project",
            options=options,
            index=selected_index,
            label_visibility="collapsed",
            key="project_selection_sidebar",
            on_change=on_project_change
        )