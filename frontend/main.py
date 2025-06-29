


from appdirs import user_config_dir

import streamlit as st
import sys
import os
import platform
import json
import folium
from PIL import Image

AddaxAI_files = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
st.set_page_config(initial_sidebar_state="auto", page_icon=os.path.join(
    AddaxAI_files, "AddaxAI", "streamlit-AddaxAI", "frontend", "logo_square.png"), page_title="AddaxAI")

# st.write(f"AddaxAI files: {AddaxAI_files}")

# get rid of the Streamlit menu
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}  /* Hides the menu */
        .stDeployButton {display: none;}  /* Hides the deploy button */
        footer {visibility: hidden;}  /* Hides the footer */
        #stDecoration {display: none;}  /* Hides the Streamlit decoration */
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        /* Reduce space above widgets */
        .block-container {
            padding-top: 15px !important;  /* Adjust as needed */
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Inject custom CSS
# to adjust the width of the popover container
st.markdown("""
    <style>
        /* Target the popover container */
        [data-testid="stPopoverBody"] {
            width: 800px !important;  /* Adjust the width as needed */
            max-width: 90vw;          /* Optional: prevent overflow on small screens */
        }
    </style>
""", unsafe_allow_html=True)

# Inject Material Icons for the header stepper bar
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
""", unsafe_allow_html=True)

# find addaxAI files folder path
# find folder of script


# paths = [
#     AddaxAI_files,
#     os.path.join(AddaxAI_files, "AddaxAI", "streamlit-AddaxAI"),
# ]

# # Add paths to sys.path
# for path in paths:
#     if path not in sys.path:
#         sys.path.append(path)

# # Ensure all paths are in PYTHONPATH while keeping existing ones
# existing_pythonpath = os.environ.get("PYTHONPATH", "").strip(
#     ":").split(":")  # Strip leading/trailing `:`
# # Preserve order and remove duplicates
# updated_pythonpath = list(dict.fromkeys(paths + existing_pythonpath))

# # Set PYTHONPATH
# os.environ["PYTHONPATH"] = ":".join(updated_pythonpath)

# insert dependencies to system variables
cuda_toolkit_path = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
paths_to_add = [
    # "/Users/peter/Desktop/streamlit_tree_selector/streamlit_tree_select",
    os.path.join(AddaxAI_files),
    os.path.join(AddaxAI_files, "cameratraps"),
    os.path.join(AddaxAI_files, "cameratraps", "megadetector"),
    os.path.join(AddaxAI_files, "AddaxAI", "streamlit-AddaxAI"),
    os.path.join(AddaxAI_files, "AddaxAI")
]
if cuda_toolkit_path:
    paths_to_add.append(os.path.join(cuda_toolkit_path, "bin"))
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Update PYTHONPATH env var without duplicates
PYTHONPATH_separator = ":" if platform.system() != "Windows" else ";"
existing_paths = os.environ.get("PYTHONPATH", "").split(PYTHONPATH_separator)

# Remove empty strings, duplicates, and keep order
existing_paths = [p for i, p in enumerate(
    existing_paths) if p and p not in existing_paths[:i]]

for path in paths_to_add:
    if path not in existing_paths:
        existing_paths.append(path)

os.environ["PYTHONPATH"] = PYTHONPATH_separator.join(existing_paths)

from backend.utils import *


# Custom CSS to style the button in the header
st.markdown(
    """
    <style>
    .header-button {
        position: absolute;
        top: 16px;
        right: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# # Create the header with a button
# col1, col2 = st.columns([0.8, 0.2])  # Adjust column ratio to position the button

# with col1:
#     st.title("My Streamlit App")  # Header title

# with col2:
#     if st.button("Click Me", key="header_button"):
#         st.write("Button clicked!")  # Action when button is clicked


# make a settings file
# this should actually be done during initual setup but for now, here is OK.

config_dir = user_config_dir("AddaxAI")
os.makedirs(config_dir, exist_ok=True)

# this is where the overall settings are stored
# these will remain constant over different versions of the app
# so think of language settings, mode settings, etc.
# locations per camera, paths to deployments, etc.
settings_file = os.path.join(config_dir, "settings.json")
if not os.path.exists(settings_file):

    # start with a clean slate
    settings = {
        "lang": "en",  # default language
        "mode": 1,  # default mode (0: simple, 1: advanced)
        "selected_folder": None,
        "selected_project": None,
        "projects": {}
    }

    # start with a clean slate
    settings = {
        "vars": {
            "global": {
                "lang": "en",
                "mode": 1,
            },
            "analyse_advanced": {
            }
        },
        "projects": {}
    }

    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=2)


# DEBUG
# st.write(fetch_known_locations())


# load language settings
txts = load_txts()
# vars = load_project_vars()
settings, _ = load_settings()
lang = settings["vars"]["global"]["lang"]
mode = settings["vars"]["global"]["mode"]

# render a dummy map with a marker so that the rest of the markers this session will be rendered
m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
folium.Marker([39.949610, -75.150282]).add_to(m)


# save only the last 1000 lines of the log file
# log_fpath = "/Users/peter/Desktop/streamlit_app/frontend/streamlit_log.txt"
log_fpath = os.path.join(AddaxAI_files, "AddaxAI",
                         "streamlit-AddaxAI", "frontend", "streamlit_log.txt")
# with open(log_fpath, "r", encoding="utf-8") as file:
#     log = file.readlines()
#     if len(log) > 1000:
#         log = log[-1000:]
# with open(log_fpath, "w", encoding="utf-8") as file:
#     file.writelines(log)
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
st.logo(os.path.join(AddaxAI_files, "AddaxAI",
        "streamlit-AddaxAI", "frontend", "logo.png"), size="large")
if mode == 0:  # simple mode
    analyse_sim_page = st.Page(
        "analyse_simple.py", title=txts["analyse_txt"][lang], icon=":material/rocket_launch:")
    pg = st.navigation([analyse_sim_page])
elif mode == 1:  # advanced mode
    analyse_adv_page = st.Page(
        "analyse_advanced.py", title=txts["analyse_txt"][lang], icon=":material/add:")
    repeat_detection_elimination_page = st.Page(
        "repeat_detection_elimination.py", title="Repeat detection elimination", icon=":material/reset_image:")
    verify_page = st.Page(
        "verify.py", title="Human verification", icon=":material/checklist:")
    depth_estimation_page = st.Page(
        "depth_estimation.py", title="Depth estimation", icon=":material/square_foot:")
    explore_page = st.Page(
        "explore.py", title="Explore results", icon=":material/bar_chart_4_bars:")
    postprocess_page = st.Page(
        "postprocess.py", title=txts["postprocess_txt"][lang], icon=":material/stylus_note:")
    camera_management_page = st.Page(
        "camera_management.py", title="Metadata management", icon=":material/photo_camera:")
    settings_page = st.Page(
        "settings.py", title=txts["settings_txt"][lang], icon=":material/settings:")
    pg = st.navigation([analyse_adv_page, repeat_detection_elimination_page, verify_page,
                       depth_estimation_page, explore_page, postprocess_page, camera_management_page, settings_page])
pg.run()

# mode settings
mode_options = {
    0: "Simple",
    1: "Advanced",
}


def on_mode_change():
    save_global_vars({"mode": st.session_state["mode_selection"]})


mode_selected = st.sidebar.segmented_control(
    "Mode",
    options=mode_options.keys(),
    format_func=mode_options.get,
    selection_mode="single",
    label_visibility="visible",
    help=txts["mode_explanation_txt"][lang],
    key="mode_selection",
    on_change=on_mode_change,
    default=mode
)
