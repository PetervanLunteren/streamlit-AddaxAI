
# for later
# TODO: https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/classification_postprocessing.py
# TODO: https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
# TODO: RDE
# todo: put all golabl variables in st.sessionstate, like model_meta, txts, vars, map, etc.
# todo: make function that checks the model_meta online and adds the variables.json file to the model folder if it does not exist


# CLI to run it
# cd /Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI && conda activate env-streamlit-addaxai && streamlit run main.py >> assets/logs/log.txt 2>&1 &


# packages
from appdirs import user_config_dir
import streamlit as st
import sys
import os
import json
from utils.analyse_advanced import load_known_projects
from utils.common import print_widget_label
import folium

from utils.config import ADDAXAI_FILES
from utils.common import load_lang_txts, load_vars, update_vars

# Force all output to be unbuffered and go to stdout
sys.stdout.reconfigure(line_buffering=True)
sys.stderr = sys.stdout  # Redirect stderr to stdout too

# make sure the config file is set
os.environ["STREAMLIT_CONFIG"] = os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI", ".streamlit", "config.toml") 

st.set_page_config(initial_sidebar_state="auto", page_icon=os.path.join(
    ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI", "assets", "images", "logo_square.png"), page_title="AddaxAI")

# Load custom CSS from external file
with open(os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI", "assets", "css", "styles.css"), "r") as f:
    css_content = f.read()

st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# Inject Material Icons for the header stepper bar
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
""", unsafe_allow_html=True)


config_dir = user_config_dir("AddaxAI")
os.makedirs(config_dir, exist_ok=True)

# TODO: put the config_dir in session state 


# this is where the overall settings are stored
# these will remain constant over different versions of the app
# so think of language settings, mode settings, etc.
# locations per camera, paths to deployments, etc.
map_file = os.path.join(config_dir, "map.json") 
if not os.path.exists(map_file):

    # start with a clean slate
    map = {
        "projects": {}
    }

    with open(map_file, "w") as f:
        json.dump(map, f, indent=2)

general_settings_file = os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI", "vars", f"general_settings.json")
if not os.path.exists(general_settings_file):
    
    # make directory if it does not exist
    os.makedirs(os.path.dirname(general_settings_file), exist_ok=True)
    
    # create a default settings file
    general_settings = {
        "lang": "en",
        "mode": 1,  # 0: simple mode, 1: advanced mode
        "selected_projectID": None
    }
    with open(general_settings_file, "w") as f:
        json.dump(general_settings, f, indent=2)



general_settings_vars = load_vars(section = "general_settings")
lang = general_settings_vars["lang"]
mode = general_settings_vars["mode"]

# Load language texts into session state on startup only
if not st.session_state.get("txts"):
    full_txts = load_lang_txts()
    # Store only the current language's texts in flattened structure
    st.session_state["txts"] = {key: value[lang] for key, value in full_txts.items()}

# Use session state texts
txts = st.session_state["txts"]

# render a dummy map with a marker so that the rest of the markers this session will be rendered
m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
folium.Marker([39.949610, -75.150282]).add_to(m)

# Initialize shared session state for cross-tool temporary variables
if "shared" not in st.session_state:
    st.session_state["shared"] = {}

# save only the last 1000 lines of the log file
# log_fpath = "/Users/peter/Desktop/streamlit_app/frontend/streamlit_log.txt"
log_fpath = os.path.join(ADDAXAI_FILES, "AddaxAI",
                         "streamlit-AddaxAI", "assets", "logs", "log.txt")
if not os.path.exists(log_fpath):
    # create an empty log file
    with open(log_fpath, "w", encoding="utf-8") as file:
        file.write("AddaxAI Streamlit App Log\n")
        file.write("========================================\n")
        file.write("This is the log file for the AddaxAI Streamlit app.\n")
        file.write("It will contain the last 1000 lines of the log.\n")
        file.write("========================================\n\n")  
with open(log_fpath, "r", encoding="utf-8") as file:
    log = file.readlines()
    if len(log) > 1000:
        log = log[-1000:]
with open(log_fpath, "w", encoding="utf-8") as file:
    file.writelines(log)
try:  # DEBUG there seems to be a problem with the permissions here
    with open(log_fpath, "r", encoding="utf-8") as file:
        log = file.readlines()
    if len(log) > 1000:
        log = log[-1000:]
    with open(log_fpath, "w", encoding="utf-8") as file:
        file.writelines(log)
except PermissionError:
    print(
        f"Permission denied when accessing {log_fpath}. Could not trim log file.")
except FileNotFoundError:
    print(f"Log file {log_fpath} not found.")
except Exception as e:
    print(f"Unexpected error: {e}")

# page navigation
# st.logo("/Users/peter/Desktop/streamlit_app/frontend/logo.png", size = "large")
st.logo(os.path.join(ADDAXAI_FILES, "AddaxAI",
        "streamlit-AddaxAI", "assets", "images", "logo.png"), size="large")
if mode == 0:  # simple mode
    analyse_sim_page = st.Page(
        os.path.join("tools", "analyse_simple.py"), title=txts["analyse_txt"], icon=":material/rocket_launch:")
    pg = st.navigation([analyse_sim_page])
elif mode == 1:  # advanced mode
    analyse_adv_page = st.Page(
        os.path.join("tools", "analyse_advanced.py"), title=txts["analyse_txt"], icon=":material/add:")
    repeat_detection_elimination_page = st.Page(
        os.path.join("tools", "repeat_detection_elimination.py"), title="Repeat detection elimination", icon=":material/reset_image:")
    verify_page = st.Page(
        os.path.join("tools", "verify.py"), title="Human verification", icon=":material/checklist:")
    depth_estimation_page = st.Page(
        os.path.join("tools", "depth_estimation.py"), title="Depth estimation", icon=":material/square_foot:")
    explore_page = st.Page(
        os.path.join("tools", "explore.py"), title="Explore results", icon=":material/bar_chart_4_bars:")
    postprocess_page = st.Page(
        os.path.join("tools", "postprocess.py"), title=txts["postprocess_txt"], icon=":material/stylus_note:")
    camera_management_page = st.Page(
        os.path.join("tools", "camera_management.py"), title="Metadata management", icon=":material/photo_camera:")
    settings_page = st.Page(
        os.path.join("tools", "settings.py"), title=txts["settings_txt"], icon=":material/settings:")
    pg = st.navigation([analyse_adv_page, repeat_detection_elimination_page, verify_page,
                       depth_estimation_page, explore_page, postprocess_page, camera_management_page, settings_page])
pg.run()

# mode settings
mode_options = {
    0: "Simple",
    1: "Advanced",
}


def on_mode_change():
    # Only update persistent storage, no session state needed
    if "mode_selection" in st.session_state:
        mode_selection = st.session_state["mode_selection"]
        update_vars("general_settings", {
            "mode": mode_selection
        })


print_widget_label("Mode", help_text="help text", sidebar=True)
# Use persistent value directly
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

# No session state cleanup needed - only persistent storage used


def on_project_change():
    # Only update persistent storage, no session state needed
    if "project_selection_sidebar" in st.session_state:
        project_selection = st.session_state["project_selection_sidebar"]
        update_vars("general_settings", {
            "selected_projectID": project_selection
        })

if mode == 1:  # advanced mode

    # check what is already known and selected
    projects, selected_projectID = load_known_projects()

    # if first project, show only button and no dropdown
    if not projects == {}:
        
        options = list(projects.keys())
        # Use persistent value directly
        selected_index = options.index(selected_projectID) if selected_projectID in options else 0

        # overwrite selected_projectID if user has selected a different project
        print_widget_label("Project", help_text="The project selected here is the one that all tools will work with. If you have a new project, you can add it at + add deployment when you want to process the first batch data.", sidebar=True)
        selected_projectID = st.sidebar.selectbox(
            "Project",
            options=options,
            index=selected_index,
            label_visibility="collapsed",
            key="project_selection_sidebar",
            on_change=on_project_change
        )
        
        # No session state cleanup needed - only persistent storage used