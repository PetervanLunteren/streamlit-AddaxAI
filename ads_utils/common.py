
# from streamlit_tree_select import tree_select
import os
import json
import streamlit as st
# import sys
# import folium as fl
# from streamlit_folium import st_folium
import streamlit as st
from appdirs import user_config_dir
# import pandas as pd
# import statistics
# from collections import defaultdict
# import subprocess
# import string
# import math
# import time as sleep_time
from datetime import datetime #, time, timedelta
# from datetime import datetime
import os
# from pathlib import Path
from datetime import datetime
# from PIL import Image
from st_flexible_callout_elements import flexible_callout
# import random
# from PIL.ExifTags import TAGS
# from hachoir.metadata import extractMetadata
# from hachoir.parser import createParser
# import piexif
# from cameratraps.megadetector.detection.video_utils import VIDEO_EXTENSIONS
# from cameratraps.megadetector.utils.path_utils import IMG_EXTENSIONS

# set global variables
AddaxAI_files = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
CLS_DIR = os.path.join(AddaxAI_files, "models", "cls")
DET_DIR = os.path.join(AddaxAI_files, "models", "det")

# load camera IDs
config_dir = user_config_dir("AddaxAI")
map_file = os.path.join(config_dir, "map.json")

# set versions
with open(os.path.join(AddaxAI_files, 'AddaxAI', 'version.txt'), 'r') as file:
    current_AA_version = file.read().strip()


class StepperBar:
    def __init__(self, steps, orientation='horizontal', active_color='blue', completed_color='green', inactive_color='gray'):
        self.steps = steps
        self.step = 0
        self.orientation = orientation
        self.active_color = active_color
        self.completed_color = completed_color
        self.inactive_color = inactive_color

    def set_step(self, step):
        if 0 <= step < len(self.steps):
            self.step = step
        else:
            raise ValueError("Step index out of range")

    def display(self):
        if self.orientation == 'horizontal':
            return self._display_horizontal()
        elif self.orientation == 'vertical':
            return self._display_vertical()
        else:
            raise ValueError(
                "Orientation must be either 'horizontal' or 'vertical'")

    def _display_horizontal(self):
        stepper_html = "<div style='display:flex; justify-content:space-between; align-items:center;'>"
        for i, step in enumerate(self.steps):
            if i < self.step:
                icon = "check_circle"
                color = self.completed_color
            elif i == self.step:
                icon = "radio_button_checked"
                color = self.active_color
            else:
                icon = "radio_button_unchecked"
                color = self.inactive_color

            stepper_html += f"""
            <div style='text-align:center;'>
                <span class="material-icons" style="color:{color}; font-size:30px;">{icon}</span>
                <div>{step}</div>
            </div>"""
            if i < len(self.steps) - 1:
                stepper_html += f"<div style='flex-grow:1; height:2px; background-color:{self.inactive_color};'></div>"
        stepper_html += "</div>"
        return stepper_html

    def _display_horizontal(self):
        stepper_html = "<div style='display:flex; justify-content:space-between; align-items:center;'>"
        for i, step in enumerate(self.steps):
            if i < self.step:
                icon = "check_circle"
                color = self.completed_color
            elif i == self.step:
                icon = "radio_button_checked"
                color = self.active_color
            else:
                icon = "radio_button_unchecked"
                color = self.inactive_color

            stepper_html += f"""
            <div style='text-align:center;'>
                <span class="material-icons" style="color:{color}; font-size:30px;">{icon}</span>
                <div style="color:{color};">{step}</div>
            </div>"""
            if i < len(self.steps) - 1:
                stepper_html += f"<div style='flex-grow:1; height:2px; background-color:{self.inactive_color};'></div>"
        stepper_html += "</div>"
        return stepper_html

    def _display_vertical(self):
        stepper_html = "<div style='display:flex; flex-direction:column; align-items:flex-start;'>"
        for i, step in enumerate(self.steps):
            color = self.completed_color if i < self.step else self.inactive_color
            current_color = self.active_color if i == self.step else color
            stepper_html += f"""
            <div style='display:flex; align-items:center; margin-bottom:10px;'>
                <div style='width:30px; height:30px; border-radius:50%; background-color:{current_color}; margin-right:10px;'></div>
                <div>{step}</div>
            </div>"""
            if i < len(self.steps) - 1:
                stepper_html += f"<div style='width:2px; height:20px; background-color:{self.inactive_color}; margin-left:14px;'></div>"
        stepper_html += "</div>"
        return stepper_html


# def load_step(section):
#     # load
#     # settings, _ = load_settings()
#     # selected_projectID = settings["vars"]["analyse_advanced"].get("selected_projectID")
#     # project_vars = settings["projects"][project]["vars"]
#     # step = project_vars.get("step", 0)
#     # return step

#     # settings, _ = load_map()
#     analyse_advanced_vars = load_vars(section="analyse_advanced")
#     step = analyse_advanced_vars.get("step", 0)
#     return step




def default_converter(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(
        f"Object of type {obj.__class__.__name__} is not JSON serializable")


def clear_vars(section):
    """
    Clear all variables in a specific section of the settings.
    """
    # settings, settings_file = load_map()
    
    # TODO: this is messing with the queue. it is deleting the queue too...
    # it should clear only the temp vars in the section, so if we are back to st.session_state, ot should clear only the session state vars of the selected session.
    
    # if not exist, create empty vars file
    vars_file = os.path.join(AddaxAI_files, "AddaxAI", "streamlit-AddaxAI", "vars", f"{section}.json")
    # if os.path.exists(vars_file): # temorarily disabled
    #     os.remove(vars_file)

def replace_vars(section, new_vars):
    vars_file = os.path.join(AddaxAI_files, "AddaxAI", "streamlit-AddaxAI", "vars", f"{section}.json")

    # # Ensure the directory exists (optional, but safe)
    # os.makedirs(os.path.dirname(vars_file), exist_ok=True)

    # Overwrite with only the new updates
    with open(vars_file, "w", encoding="utf-8") as file:
        json.dump(new_vars, file, indent=2, default=default_converter)

def update_vars(section, updates):
    # settings, settings_file = load_map()

    vars_file = os.path.join(AddaxAI_files, "AddaxAI", "streamlit-AddaxAI", "vars", f"{section}.json")
    # /Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/vars/general_settings.json
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

    # Load full settings or initialize
    try:
        if os.path.exists(map_file):
            with open(map_file, "r", encoding="utf-8") as f:
                settings = json.load(f)
        else:
            settings = {}
    except (json.JSONDecodeError, IOError):
        settings = {}

    return settings, map_file


def load_vars(section):

    # if not exist, create empty vars file
    vars_file = os.path.join(AddaxAI_files, "AddaxAI", "streamlit-AddaxAI", "vars", f"{section}.json")
    if not os.path.exists(vars_file):
        with open(vars_file, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
    
    # read section vars
    with open(vars_file, "r", encoding="utf-8") as f:
        section_vars = json.load(f)
    
    return section_vars
    # return {var: section_vars.get(var, None) for var in requested_vars}.values()



def load_lang_txts():
    txts_fpath = os.path.join(AddaxAI_files, "AddaxAI",
                              "streamlit-AddaxAI", "assets", "language", "lang.json")
    with open(txts_fpath, "r", encoding="utf-8") as file:
        txts = json.load(file)
    return txts


def settings(txts, lang, mode):
    _, col2 = st.columns([5, 1])
    with col2:
        with st.popover(":material/menu: Menu", use_container_width=True,
                        ):
            # with st.form("my_form", border=False):
            # # Mode settings
            # st.subheader(":material/toggle_on: Mode", divider="grey")
            # st.write(txts["mode_explanation_txt"][lang])
            # vars = load_vars()
            # mode_options = {
            #     0: "Simple",
            #     1: "Advanced",
            # }
            # mode_selected = st.segmented_control(
            #     "Mode",
            #     options=mode_options.keys(),
            #     format_func=lambda option: mode_options[option],
            #     selection_mode="single",
            #     label_visibility="collapsed",
            #     default=mode
            # )
            # st.text("")

            # # Language settings
            # st.subheader(":material/language: Language", divider="grey")
            # lang_options = txts["languages"]
            # lang_idx = list(lang_options.keys()).index(lang)
            # lang_selected = st.selectbox(
            #     "Language",
            #     options=lang_options.keys(),
            #     format_func=lambda option: lang_options[option],
            #     index=lang_idx,
            #     label_visibility="collapsed"
            # )
            st.text("")






#########################
### GENERAL UTILITIES ###
#########################

def multiselect_checkboxes(classes, preselected):

    # select all or none buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button(":material/check_box: Select all", use_container_width=True):
            for species in classes:
                st.session_state[f"species_{species}"] = True
    with col2:
        if st.button(":material/check_box_outline_blank: Select none", use_container_width=True):
            for species in classes:
                st.session_state[f"species_{species}"] = False

    # checkboxes in a scrollable container
    selected_species = []
    with st.container(border=True, height=300):
        for species in classes:
            key = f"species_{species}"
            checked = st.session_state.get(
                key, True if species in preselected else False)
            if st.checkbox(species, value=checked, key=key):
                selected_species.append(species)

    # log selected species
    st.markdown(
        f'&nbsp; You selected the presence of <code style="color:#086164; font-family:monospace;">{len(selected_species)}</code> classes', unsafe_allow_html=True)

    # return list
    return selected_species


def print_widget_label(label_text, icon=None, help_text=None):
    if icon:
        line = f":material/{icon}: &nbsp; "
    else:
        line = ""
    st.markdown(f"{line}**{label_text}**", help=help_text)


def radio_buttons_with_captions(option_caption_dict, key, scrollable, default_option):
    # Extract option labels and captions from the dictionary
    options = [v["option"] for v in option_caption_dict.values()]
    captions = [v["caption"] for v in option_caption_dict.values()]
    key_map = {v["option"]: k for k, v in option_caption_dict.items()}

    # Get default index based on default_key
    default_index = list(option_caption_dict.keys()).index(default_option)

    # Create a radio button selection with captions
    with st.container(border=True, height=275 if scrollable else None):
        selected_option = st.radio(
            label=key,
            options=options,
            index=default_index,
            label_visibility="collapsed",
            captions=captions
        )

    # Return the corresponding key
    return key_map[selected_option]


# check if the user needs an update
def requires_addaxai_update(required_version):
    current_parts = list(map(int, current_AA_version.split('.')))
    required_parts = list(map(int, required_version.split('.')))

    # Pad the shorter version with zeros
    while len(current_parts) < len(required_parts):
        current_parts.append(0)
    while len(required_parts) < len(current_parts):
        required_parts.append(0)

    # Compare each part of the version
    for current, required in zip(current_parts, required_parts):
        if current < required:
            return True  # current_version is lower than required_version
        elif current > required:
            return False  # current_version is higher than required_version

    # All parts are equal, consider versions equal
    return False




def info_box(msg, icon=":material/info:"):
    flexible_callout(msg,
                     icon=icon,
                     background_color="#d9e3e7af",
                     font_color="#086164",
                     icon_size=23)


# import streamlit as st


# SPECIES SELECTOR
