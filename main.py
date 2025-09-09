
# WAAR WAS IK? 
# TODO: 
# - Adding new project does not auto select it, it stays on the previous project. Fix that.
# - zorg voor dat de data van de deployments wordt opgehaald in een session state df wordt gezet, test het op windows inclusief de instalatie, en zorg voor dat de vrijwilliger aan de slag kan
# - dan testen op windows, werkt het ook zonder in de folder van addaxai te zitten?
# - dan opschonen, commenten, teksten, readme, mds, etc. 
# - make the stepper bar vertical, so we can keep adding more steps without running out of space, and it makes it more clear we're working towards the queue
# - make the format for classify_detections.py simpler for collaborators. Bascially, we need three functions, one to lead the model, one to crop an image, and one to classify a crop. Three functions, always the same. 

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

TODOs:
- https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/classification_postprocessing.py
- https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- RDE
- material icons must be offline, but I can only do that at the end when I know which icons I'm using.
- also save the image or video that had the min_datetime, so that we can calculate the diff every time we need it "deployment_start_file". Then it can read the exif from the path. No need to read all exifs of all images.  searc h for deployment_start_file, deployment_start_datetime
- reformat obj names, function names, and variable names, and classes, etc
- do the loading squirell in a modal too, to you cant click anything else while loading
- if no CLS is selected, it should skip species selection and the button should be add to queue. 
"""

# Standard library imports
import streamlit as st
import sys
import os
import time
import json

# Third-party imports
from streamlit_lottie import st_lottie
from st_modal import Modal
from appdirs import user_config_dir, user_cache_dir
 

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
    # Set flag to show startup modal
    st.session_state["show_startup_modal"] = True

# Create startup modal only when flag is set (following existing modal pattern)
if st.session_state.get("show_startup_modal", False):
    modal_startup = Modal(
        title=None, 
        key="startup-loading", 
        show_close_button=False
    )
    
    with modal_startup.container():
        # Load and display Lottie squirrel animation
        lottie_animation_fpath = os.path.join(ADDAXAI_ROOT, "assets", "loaders", "squirrel.json")
        with open(lottie_animation_fpath, "r") as f:
            lottie_animation = json.load(f)
            
        # Center the animation
        _, col_animation, _ = st.columns([1, 2, 1])
        with col_animation:
            st_lottie(
                lottie_animation,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                key="startup_lottie_animation"
            )
    
        
        # Initialize startup with detailed progress tracking
        _, col, _ = st.columns([1, 2, 1])
        with col:
            with st.status("Initializing AddaxAI...", expanded=False) as startup_status:
                
                # ─────────────────────────────────────────────────────────────────────────
                # Step 1: Initialize session state and directory structure
                # ─────────────────────────────────────────────────────────────────────────
                try:
                    st.write("Setting up session state...")
                    startup_status.update(label="Setting up session state...")
                    # Initialize shared session state container for cross-tool temporary variables
                    st.session_state["shared"] = {}
                    
                    st.write("Creating configuration directories...")
                    startup_status.update(label="Creating configuration directories...")
                    # Create config directories at startup (appdirs may not be available in all environments)
                    CONFIG_DIR = user_config_dir("AddaxAI")
                    os.makedirs(CONFIG_DIR, exist_ok=True)
                    
                    MAP_FILE_PATH = os.path.join(CONFIG_DIR, "map.json")
                    
                    # Store paths in session state for tool access
                    st.session_state["shared"] = {
                        "CONFIG_DIR": CONFIG_DIR,
                        "MAP_FILE_PATH": MAP_FILE_PATH
                    }
                    st.write("Initializing directories...")
                    startup_status.update(label="Initializing directories...")
                    
                except Exception as e:
                    st.error(f"Failed to initialize directories: {str(e)}")
                    startup_status.update(label="Startup failed", state="error", expanded=True)
                    st.stop()
    
                # ─────────────────────────────────────────────────────────────────────────
                # Step 2: Load utility modules
                # ─────────────────────────────────────────────────────────────────────────
                try:
                    st.write("Loading utility modules...")
                    startup_status.update(label="Loading utility modules...")
                    # Import utils now that shared session state exists (modules depend on it)
                    from utils.common import load_lang_txts, load_vars, update_vars, set_session_var, get_session_var, fetch_latest_model_info
                    from components import print_widget_label
                    from utils.analysis_utils import load_known_projects, load_model_metadata
                    st.write("Loading utility modules...")
                    startup_status.update(label="Loading utility modules...")
                    
                except Exception as e:
                    st.error(f"Failed to load utility modules: {str(e)}")
                    startup_status.update(label="Startup failed", state="error", expanded=True)
                    st.stop()

                # ─────────────────────────────────────────────────────────────────────────
                # Step 3: Initialize configuration files
                # ─────────────────────────────────────────────────────────────────────────
                try:
                    st.write("Creating configuration files...")
                    startup_status.update(label="Creating configuration files...")
                    
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

                    st.write("Loading general settings...")
                    startup_status.update(label="Loading general settings...")
                    # Load and cache general settings to avoid file reads on reruns
                    general_settings_vars = load_vars(section = "general_settings")
                    lang = general_settings_vars["lang"]
                    mode = general_settings_vars["mode"]
                    
                    set_session_var("shared", "lang", lang)
                    set_session_var("shared", "mode", mode)
                    st.write("Initializing configuration files...")
                    startup_status.update(label="Initializing configuration files...")
                    
                except Exception as e:
                    st.error(f"Failed to initialize configuration files: {str(e)}")
                    startup_status.update(label="Startup failed", state="error", expanded=True)
                    st.stop()

                # ─────────────────────────────────────────────────────────────────────────
                # Step 4: Load language files
                # ─────────────────────────────────────────────────────────────────────────
                try:
                    st.write("Loading language files...")
                    startup_status.update(label="Loading language files...")
                    # Load and cache language texts to avoid file I/O on reruns
                    if not st.session_state.get("txts"):
                        full_txts = load_lang_txts()
                        # Store only current language's texts in flattened structure for efficiency
                        st.session_state["txts"] = {key: value[lang] for key, value in full_txts.items()}
                    st.write("Loading language files...")
                    startup_status.update(label="Loading language files...")
                    
                except Exception as e:
                    st.error(f"Failed to load language files: {str(e)}")
                    startup_status.update(label="Startup failed", state="error", expanded=True)
                    st.stop()

                # ─────────────────────────────────────────────────────────────────────────
                # Step 5: Load AI model metadata
                # ─────────────────────────────────────────────────────────────────────────
                try:
                    st.write("Loading AI model metadata...")
                    startup_status.update(label="Loading AI model metadata...")
                    # Load and cache AI model metadata (large JSON file)
                    if not st.session_state.get("model_meta"):
                        st.session_state["model_meta"] = load_model_metadata()
                    st.write("Loading model metadata...")
                    startup_status.update(label="Loading model metadata...")
                    
                except Exception as e:
                    st.error(f"Failed to load model metadata: {str(e)}")
                    startup_status.update(label="Startup failed", state="error", expanded=True)
                    st.stop()

                # ─────────────────────────────────────────────────────────────────────────
                # Step 6: Download latest model information
                # ─────────────────────────────────────────────────────────────────────────
                try:
                    st.write("Downloading latest model information...")
                    startup_status.update(label="Downloading latest model information...")
                    # Download latest model metadata and create folders for new models
                    fetch_latest_model_info()
                    
                    # Reload model metadata after download to ensure session state has latest data
                    st.session_state["model_meta"] = load_model_metadata()
                    st.write("Downloading model information...")
                    startup_status.update(label="Downloading model information...")
                    
                except Exception as e:
                    st.error(f"Failed to download model information: {str(e)}")
                    startup_status.update(label="Startup failed", state="error", expanded=True)
                    st.stop()

                # ─────────────────────────────────────────────────────────────────────────
                # Step 7: Setup logging system
                # ─────────────────────────────────────────────────────────────────────────
                try:
                    st.write("Setting up logging system...")
                    startup_status.update(label="Setting up logging system...")
                    # Archive previous session log and create fresh log (only on startup)
                    log_fpath = os.path.join(ADDAXAI_ROOT, "assets", "logs", "log.txt")
                    previous_sessions_dir = os.path.join(ADDAXAI_ROOT, "assets", "logs", "previous_sessions")
                    
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
                    st.write("Setting up logging system...")
                    startup_status.update(label="Setting up logging system...")
                    
                except PermissionError:
                    st.warning(f"Permission denied when accessing log file. Logging disabled.")
                except Exception as e:
                    st.warning(f"Failed to setup logging: {str(e)}. Continuing without logging.")
                
                # ─────────────────────────────────────────────────────────────────────────
                # Startup complete
                # ─────────────────────────────────────────────────────────────────────────
                startup_status.update(
                    label="AddaxAI startup complete!", 
                    state="complete", 
                    expanded=False
                )
                
                # Clear startup modal flag to close the modal
                st.session_state["show_startup_modal"] = False
                
                # Force rerun to apply modal closure
                st.rerun()

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

        # Project selector in sidebar
        print_widget_label("Project", help_text="Selected project applies to all tools. Add new projects via + add deployment.", sidebar=True)
        selected_projectID = st.sidebar.selectbox(
            "Project",
            options=options,
            index=selected_index,
            label_visibility="collapsed",
            key="project_selection_sidebar",
            on_change=on_project_change
        )