

import streamlit as st
import os
from tools import analyse_advanced, analyse_simple, camera_management, depth_estimation, explore, postprocess, repeat_detection_elimination, settings, verify
from ads_utils.common import load_lang_txts, load_vars, update_vars
from appdirs import user_config_dir
import json
import folium

# Set page config
st.set_page_config(
    initial_sidebar_state="auto",
    page_icon="assets/images/logo_square.png",
    page_title="AddaxAI"
)

# Hide default Streamlit elements
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        .stDeployButton {display: none;}
        footer {visibility: hidden;}
        #stDecoration {display: none;}
    </style>
""", unsafe_allow_html=True)

# --- PATHS AND CONFIG ---
APP_DIR = os.path.dirname(os.path.realpath(__file__))
ADDAXAI_FILES = os.path.dirname(APP_DIR)
CONFIG_DIR = user_config_dir("AddaxAI")
os.makedirs(CONFIG_DIR, exist_ok=True)

MAP_FILE = os.path.join(CONFIG_DIR, "map.json")
if not os.path.exists(MAP_FILE):
    with open(MAP_FILE, "w") as f:
        json.dump({"projects": {}}, f, indent=2)

GENERAL_SETTINGS_FILE = os.path.join(APP_DIR, "vars", "general_settings.json")
if not os.path.exists(GENERAL_SETTINGS_FILE):
    with open(GENERAL_SETTINGS_FILE, "w") as f:
        json.dump({"lang": "en", "mode": 1}, f, indent=2)

# --- LOAD SETTINGS ---
txts = load_lang_txts()
general_settings_vars = load_vars(section="general_settings")
lang = general_settings_vars["lang"]
mode = general_settings_vars["mode"]

# --- PAGE NAVIGATION ---
st.logo("assets/images/logo.png", size="large")

if mode == 0:  # Simple mode
    pg = st.navigation([
        st.Page(os.path.join("tools", "analyse_simple.py"), title=txts["analyse_txt"][lang], icon=":material/rocket_launch:")
    ])
else:  # Advanced mode
    pg = st.navigation([
        st.Page(os.path.join("tools", "analyse_advanced.py"), title=txts["analyse_txt"][lang], icon=":material/add:"),
        st.Page(os.path.join("tools", "repeat_detection_elimination.py"), title="Repeat detection elimination", icon=":material/reset_image:"),
        st.Page(os.path.join("tools", "verify.py"), title="Human verification", icon=":material/checklist:"),
        st.Page(os.path.join("tools", "depth_estimation.py"), title="Depth estimation", icon=":material/square_foot:"),
        st.Page(os.path.join("tools", "explore.py"), title="Explore results", icon=":material/bar_chart_4_bars:"),
        st.Page(os.path.join("tools", "postprocess.py"), title=txts["postprocess_txt"][lang], icon=":material/stylus_note:"),
        st.Page(os.path.join("tools", "camera_management.py"), title="Metadata management", icon=":material/photo_camera:"),
        st.Page(os.path.join("tools", "settings.py"), title=txts["settings_txt"][lang], icon=":material/settings:")
    ])

pg.run()

# --- MODE SELECTION ---
mode_options = {0: "Simple", 1: "Advanced"}

def on_mode_change():
    update_vars("general_settings", {"mode": st.session_state["mode_selection"]})

st.sidebar.segmented_control(
    "Mode",
    options=list(mode_options.keys()),
    format_func=lambda k: mode_options[k],
    key="mode_selection",
    on_change=on_mode_change,
    default=mode,
    help=txts["mode_explanation_txt"][lang]
)
