
"""
AddaxAI Application Settings

Configuration interface for global application settings including:
- Language selection
- Application mode (Simple/Advanced)
- System information display
- Version and environment details
"""

import streamlit as st
import sys
import platform
import os

from utils.config import ADDAXAI_ROOT
from utils.common import (
    load_lang_txts,
    load_vars,
    load_map,
    set_session_var,
    update_vars,
    load_app_settings,
    save_app_settings,
)

st.set_page_config(layout="centered")


# DATA IMPORT
app_settings = load_app_settings()
data_import_settings = app_settings.get("data_import", {})
detection_threshold = float(data_import_settings.get("detection_conf_threshold", 0.5))

st.subheader(":material/upload_file: Data import", divider="grey")
st.caption(
    "Each run stores the raw detection data in its own JSON file. "
    "The settings below control which detections AddaxAI loads from those raw files. "
    "You can tweak these values at any time. The underlying data remains untouched."
)

with st.form("data_import_settings_form"):
    detection_threshold_value = st.slider(
        "Detection confidence threshold",
        min_value=0.10,
        max_value=1.0,
        value=detection_threshold,
        step=0.01,
        help="Detections below this threshold will be ignored during import. Detections below 0.10 are not saved or classified to conserve storage and compute.",
    )

    submitted = st.form_submit_button("Save and reload app", type="primary", width="stretch")

    if submitted:
        app_settings.setdefault("data_import", {})["detection_conf_threshold"] = float(detection_threshold_value)
        save_app_settings(app_settings)
        st.session_state["_force_reset"] = True
        st.rerun()


# LANGUAGE
txts = load_lang_txts()
general_settings_vars = load_vars(section="general_settings")
lang = general_settings_vars["lang"]
lang_options = txts["languages"]
lang_idx = list(lang_options.keys()).index(lang)

st.subheader(":material/language: Language", divider="grey")
lang_selected = st.selectbox(
    "Language",
    options=lang_options.keys(),
    format_func=lambda option: lang_options[option],
    index=lang_idx,
    label_visibility="collapsed"
)

if st.button(":material/save: Save language", width="stretch", type="primary", key="language_btn"):
    set_session_var("shared", "lang", lang_selected)
    update_vars("general_settings", {"lang": lang_selected})
    st.rerun()


# OTHER STUFF
st.subheader(":material/more_horiz: Other stuff", divider="grey")

if st.button("Start from scratch", width="stretch", type="primary"):
    config_dir = os.path.join(ADDAXAI_ROOT, "config")
    for filename in os.listdir(config_dir):
        file_path = os.path.join(config_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    map, MAP_FILE_PATH = load_map()
    if os.path.exists(MAP_FILE_PATH):
        os.remove(MAP_FILE_PATH)
    st.session_state.clear()
    st.rerun()

map, MAP_FILE_PATH = load_map()
st.write("map_file:", MAP_FILE_PATH)
if st.button("Open map.json"):
    if platform.system() == "Darwin":
        os.system(f'open "{MAP_FILE_PATH}"')
    else:
        st.warning("This action is only supported on macOS.")
with st.expander("Current settings", expanded=False):
    st.write("map:", map)
with st.expander("st.session_state", expanded=False):
    st.write(st.session_state)
with st.expander("Detection Results DataFrame", expanded=False):
    if "observations_source_df" in st.session_state:
        st.dataframe(st.session_state["observations_source_df"], width='stretch')
    else:
        st.warning("Detection results not loaded in session state")
    
    
# # logs
# st.subheader(":material/description: Logs", divider="grey")
# st.write("Here, you can export the application logs to review errors, warnings, and other messages, which can be helpful when sharing details with the development team.")
# log_fpath = "/Users/peter/Desktop/streamlit_app/frontend/streamlit_log.txt"
# import time
# print(f"\n\nExported Log file '{log_fpath}' on {time.strftime('%Y-%m-%d %H:%M:%S')}")
# sys.stdout.flush()
# with open(log_fpath, "r") as file:
#     lines = file.readlines()[-1000:]
# file_data = "".join(lines)
# st.download_button(
#     label=":material/save: Export logs",
#     data=file_data,
#     file_name="streamlit_log.txt",
#     mime="text/plain"
# )
# st.text("")



# # Save settings
# save_btn = st.button(":material/save: Save settings", width='stretch',
#                         type = "primary")
# if save_btn:
#     save_vars({"mode": mode_selected, "lang": lang_selected})
#     st.rerun()
