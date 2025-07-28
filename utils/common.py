
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
from tqdm import tqdm
import subprocess
import re
# import math
# import time as sleep_time
from datetime import datetime  # , time, timedelta
# from datetime import datetime
import os
# from pathlib import Path
from datetime import datetime
# from PIL import Image
from st_flexible_callout_elements import flexible_callout
import requests
import tarfile
# import random
# from PIL.ExifTags import TAGS
# from hachoir.metadata import extractMetadata
# from hachoir.parser import createParser
# import piexif
# from cameratraps.megadetector.detection.video_utils import VIDEO_EXTENSIONS
# from cameratraps.megadetector.utils.path_utils import IMG_EXTENSIONS

# set global variables
# AddaxAI_files = os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from utils.config import ADDAXAI_FILES
CLS_DIR = os.path.join(ADDAXAI_FILES, "models", "cls")
DET_DIR = os.path.join(ADDAXAI_FILES, "models", "det")

# load camera IDs
config_dir = user_config_dir("AddaxAI")
map_file = os.path.join(config_dir, "map.json")

# set versions
with open(os.path.join(ADDAXAI_FILES, 'AddaxAI', 'version.txt'), 'r') as file:
    current_AA_version = file.read().strip()


class MultiProgressBars:
    def __init__(self, container_label="Progress Bars"):
        self.container = st.container(border=True)
        self.label_placeholder = self.container.empty()
        if container_label:
            self.label_placeholder.markdown(container_label)
        self.bars = {}
        self.states = {}
        self.max_values = {}
        self.active_prefixes = {}
        self.wait_labels = {}
        self.pre_labels = {}
        self.done_labels = {}
        self.statuses = {}
        self.label_divider = " \u00A0\u00A0\u00A0 | \u00A0\u00A0\u00A0 "

    def update_label(self, new_label):
        """Update the container label dynamically."""
        if new_label:
            self.label_placeholder.markdown(new_label)
        else:
            self.label_placeholder.empty()

    def add_pbar(self, pbar_id, pre_label, active_prefix, done_label, wait_label=None, max_value=None):
        container = self.container.container()
        self.states[pbar_id] = 0
        self.max_values[pbar_id] = max_value or 1  # temporary placeholder
        self.active_prefixes[pbar_id] = active_prefix
        self.wait_labels[pbar_id] = wait_label
        self.pre_labels[pbar_id] = pre_label
        self.done_labels[pbar_id] = done_label
        
        # Show wait_label if provided, otherwise show pre_label
        initial_label = wait_label if wait_label else pre_label
        self.bars[pbar_id] = container.progress(0, text=initial_label)

    def start_pbar(self, pbar_id):
        """Transition from wait_label to pre_label state."""
        if pbar_id not in self.bars:
            raise ValueError(f"Progress bar '{pbar_id}' not found.")
        
        # Reset state and show pre_label
        self.states[pbar_id] = 0
        self.bars[pbar_id].progress(0, text=self.pre_labels[pbar_id])

    def set_max_value(self, pbar_id, max_value):
        if pbar_id not in self.bars:
            raise ValueError(f"Progress bar '{pbar_id}' not found.")
        self.max_values[pbar_id] = max_value
        self.states[pbar_id] = 0
        # If we have a wait_label and haven't started yet, keep showing it
        if self.wait_labels[pbar_id] and self.states[pbar_id] == 0:
            self.bars[pbar_id].progress(0, text=self.wait_labels[pbar_id])
        else:
            self.bars[pbar_id].progress(0, text=self.pre_labels[pbar_id])

    def update(self, pbar_id, n=1, text=""):
        if pbar_id not in self.bars:
            raise ValueError(f"Progress bar '{pbar_id}' not found.")

        self.states[pbar_id] += n
        if self.states[pbar_id] > self.max_values[pbar_id]:
            self.states[pbar_id] = self.max_values[pbar_id]

        progress = self.states[pbar_id] / self.max_values[pbar_id]
        display_text = (
            self.done_labels[pbar_id]
            if self.states[pbar_id] >= self.max_values[pbar_id]
            else f"{self.active_prefixes[pbar_id]} {text}".strip()
        )

        self.bars[pbar_id].progress(progress, text=display_text)

    def add_status(self, status_id, pre_label="Waiting...", mid_label="Working...", post_label="Done!"):
        import streamlit_nested_layout
        container = self.container.container()
        text_placeholder = container.empty()
        status_placeholder = container.empty()  # for the st.status()

        # Initial small gray text
        text_placeholder.markdown(
            f"<span style='font-size: 0.9rem;'>{pre_label}</span>", unsafe_allow_html=True)

        self.statuses[status_id] = {
            "text": text_placeholder,
            "status": status_placeholder,
            "labels": (pre_label, mid_label, post_label),
            "status_obj": None,
        }

    def update_status(self, status_id, phase: str):
        if status_id not in self.statuses:
            return None

        labels = self.statuses[status_id]["labels"]
        text = self.statuses[status_id]["text"]
        status_slot = self.statuses[status_id]["status"]

        if phase == "mid":
            text.markdown(
                f"<span style='font-size: 0.9rem;'>{labels[1]}</span>", unsafe_allow_html=True)
            status = status_slot.status("Details", expanded=False)
            self.statuses[status_id]["status_obj"] = status
            return status

        elif phase == "post":
            text.markdown(
                f"<span style='font-size: 0.9rem;'>{labels[2]}</span>", unsafe_allow_html=True)
            status_obj = self.statuses[status_id]["status_obj"]
            if status_obj:
                status_obj.update(label="Details", state="complete")        
        
    def update_from_tqdm_string(self, pbar_id, tqdm_line: str):
        """
        Parse a tqdm output string and update the corresponding Streamlit progress bar, including ETA.
        """
        tqdm_pattern = r"(\d+)%\|.*\|\s*(\d+)/(\d+).*?\[(.*?)<([^,]+),\s*([\d.]+)\s*(\S+)?/s\]"
        match = re.search(tqdm_pattern, tqdm_line)

        if not match:
            return  # Skip lines that do not match tqdm format

        percent = int(match.group(1))
        n = int(match.group(2))
        total = int(match.group(3))
        elapsed_str = match.group(4).strip()
        eta_str = match.group(5).strip()
        rate = float(match.group(6))
        unit = match.group(7) or ""

        self.set_max_value(pbar_id, total)
        self.states[pbar_id] = n  # Sync directly to avoid increment error

        label = (
            f"{self.label_divider}"
            f":material/clock_loader_40: {percent}%{self.label_divider}"
            f":material/laps: {n} {unit} / {total} {unit}{self.label_divider}"
            f":material/speed: {rate:.2f} {unit}/s{self.label_divider}"
            f":material/timer: {elapsed_str}{self.label_divider}"
            f":material/sports_score: {eta_str}"
        )

        self.update(pbar_id, n - self.states[pbar_id], text=label)

    def update_from_tqdm_object(self, pbar_id, pbar):
        """
        Update the progress bar directly from a tqdm object.
        """
        if pbar_id not in self.bars:
            return
        
        fmt = pbar.format_dict
        n = fmt.get("n", 0)
        total = fmt.get("total", 1)
        rate = fmt.get("rate")
        unit = fmt.get("unit", "B")
        elapsed = fmt.get("elapsed")

        def fmt_time(s):
            if s is None:
                return ""
            s = int(s)
            return f"{s // 60}:{s % 60:02}"
        
        def fmt_bytes(bytes_val, suffix="B"):
            """Format bytes into human readable format"""
            if bytes_val < 1024:
                return f"{bytes_val:.0f} {suffix}"
            elif bytes_val < 1024**2:
                return f"{bytes_val/1024:.1f} K{suffix}"
            elif bytes_val < 1024**3:
                return f"{bytes_val/(1024**2):.1f} M{suffix}"
            else:
                return f"{bytes_val/(1024**3):.1f} G{suffix}"

        # Update max value if needed
        if self.max_values[pbar_id] != total:
            self.max_values[pbar_id] = total
        
        # Calculate progress (allow over 100% like you wanted)
        progress = min(n / total, 1.0) if total > 0 else 0
        
        # Generate label with icons and proper units
        percent = int(n / total * 100) if total > 0 else 0
        percent_str = f":material/clock_loader_40: {percent}%"
        
        # Format current/total - only format bytes if unit is "B"
        if unit == "B":
            n_formatted = fmt_bytes(n)
            total_formatted = fmt_bytes(total)
            laps_str = f":material/laps: {n_formatted} / {total_formatted}"
            rate_str = f":material/speed: {fmt_bytes(rate, 'B/s')}" if rate else ""
        else:
            # For other units (files, items, animals, etc.), show as-is
            laps_str = f":material/laps: {int(n)} {unit} / {int(total)} {unit}"
            rate_str = f":material/speed: {rate:.1f} {unit}/s" if rate else ""
        
        elapsed_str = f":material/timer: {fmt_time(elapsed)}" if elapsed else ""
        eta_str = f":material/sports_score: {fmt_time((total - n) / rate)}" if rate and total > n else ""
        
        label = self.label_divider + self.label_divider.join(filter(None, [
            percent_str, laps_str, rate_str, elapsed_str, eta_str
        ]))
        
        # Update the progress bar
        self.update(pbar_id, n - self.states[pbar_id], text=label)

# class MultiProgressBars:
#     def __init__(self, container_label="Progress Bars"):
#         self.container = st.container(border=True)
#         self.label_placeholder = self.container.empty()
#         if container_label:
#             self.label_placeholder.markdown(container_label)
#         self.bars = {}
#         self.states = {}
#         self.max_values = {}
#         self.active_prefixes = {}
#         self.pre_labels = {}
#         self.done_labels = {}
#         self.statuses = {}
#         self.label_divider = " \u00A0\u00A0\u00A0 | \u00A0\u00A0\u00A0 "

#     def update_label(self, new_label):
#         """Update the container label dynamically."""
#         if new_label:
#             self.label_placeholder.markdown(new_label)
#         else:
#             self.label_placeholder.empty()

#     def add_pbar(self, pbar_id, pre_label, active_prefix, done_label, max_value=None):
#         container = self.container.container()
#         self.states[pbar_id] = 0
#         self.max_values[pbar_id] = max_value or 1  # temporary placeholder
#         self.active_prefixes[pbar_id] = active_prefix
#         self.pre_labels[pbar_id] = pre_label
#         self.done_labels[pbar_id] = done_label
#         self.bars[pbar_id] = container.progress(0, text=pre_label)

#     def set_max_value(self, pbar_id, max_value):
#         if pbar_id not in self.bars:
#             raise ValueError(f"Progress bar '{pbar_id}' not found.")
#         self.max_values[pbar_id] = max_value
#         self.states[pbar_id] = 0
#         self.bars[pbar_id].progress(0, text=self.pre_labels[pbar_id])

#     def update(self, pbar_id, n=1, text=""):
#         if pbar_id not in self.bars:
#             raise ValueError(f"Progress bar '{pbar_id}' not found.")

#         self.states[pbar_id] += n
#         if self.states[pbar_id] > self.max_values[pbar_id]:
#             self.states[pbar_id] = self.max_values[pbar_id]

#         progress = self.states[pbar_id] / self.max_values[pbar_id]
#         display_text = (
#             self.done_labels[pbar_id]
#             if self.states[pbar_id] >= self.max_values[pbar_id]
#             else f"{self.active_prefixes[pbar_id]} {text}".strip()
#         )

#         self.bars[pbar_id].progress(progress, text=display_text)

#     def add_status(self, status_id, pre_label="Waiting...", mid_label="Working...", post_label="Done!"):
#         import streamlit_nested_layout
#         container = self.container.container()
#         text_placeholder = container.empty()
#         status_placeholder = container.empty()  # for the st.status()

#         # Initial small gray text
#         text_placeholder.markdown(
#             f"<span style='font-size: 0.9rem;'>{pre_label}</span>", unsafe_allow_html=True)

#         self.statuses[status_id] = {
#             "text": text_placeholder,
#             "status": status_placeholder,
#             "labels": (pre_label, mid_label, post_label),
#             "status_obj": None,
#         }

#     def update_status(self, status_id, phase: str):
#         if status_id not in self.statuses:
#             return None

#         labels = self.statuses[status_id]["labels"]
#         text = self.statuses[status_id]["text"]
#         status_slot = self.statuses[status_id]["status"]

#         if phase == "mid":
#             text.markdown(
#                 f"<span style='font-size: 0.9rem;'>{labels[1]}</span>", unsafe_allow_html=True)
#             status = status_slot.status("Details", expanded=False)
#             self.statuses[status_id]["status_obj"] = status
#             return status

#         elif phase == "post":
#             text.markdown(
#                 f"<span style='font-size: 0.9rem;'>{labels[2]}</span>", unsafe_allow_html=True)
#             status_obj = self.statuses[status_id]["status_obj"]
#             if status_obj:
#                 status_obj.update(label="Details", state="complete")        
        
#     def update_from_tqdm_string(self, pbar_id, tqdm_line: str):
#         """
#         Parse a tqdm output string and update the corresponding Streamlit progress bar, including ETA.
#         """
#         tqdm_pattern = r"(\d+)%\|.*\|\s*(\d+)/(\d+).*?\[(.*?)<([^,]+),\s*([\d.]+)\s*(\S+)?/s\]"
#         match = re.search(tqdm_pattern, tqdm_line)

#         if not match:
#             return  # Skip lines that do not match tqdm format

#         percent = int(match.group(1))
#         n = int(match.group(2))
#         total = int(match.group(3))
#         elapsed_str = match.group(4).strip()
#         eta_str = match.group(5).strip()
#         rate = float(match.group(6))
#         unit = match.group(7) or ""

#         self.set_max_value(pbar_id, total)
#         self.states[pbar_id] = n  # Sync directly to avoid increment error

#         label = (
#             f"{self.label_divider}"
#             f":material/clock_loader_40: {percent}%{self.label_divider}"
#             f":material/laps: {n} {unit} / {total} {unit}{self.label_divider}"
#             f":material/speed: {rate:.2f} {unit}/s{self.label_divider}"
#             f":material/timer: {elapsed_str}{self.label_divider}"
#             f":material/sports_score: {eta_str}"
#         )

#         self.update(pbar_id, n - self.states[pbar_id], text=label)

#     def update_from_tqdm_object(self, pbar_id, pbar):
#         """
#         Update the progress bar directly from a tqdm object.
#         """
#         if pbar_id not in self.bars:
#             return
        
#         fmt = pbar.format_dict
#         n = fmt.get("n", 0)
#         total = fmt.get("total", 1)
#         rate = fmt.get("rate")
#         unit = fmt.get("unit", "B")
#         elapsed = fmt.get("elapsed")

#         def fmt_time(s):
#             if s is None:
#                 return ""
#             s = int(s)
#             return f"{s // 60}:{s % 60:02}"
        
#         def fmt_bytes(bytes_val, suffix="B"):
#             """Format bytes into human readable format"""
#             if bytes_val < 1024:
#                 return f"{bytes_val:.0f} {suffix}"
#             elif bytes_val < 1024**2:
#                 return f"{bytes_val/1024:.1f} K{suffix}"
#             elif bytes_val < 1024**3:
#                 return f"{bytes_val/(1024**2):.1f} M{suffix}"
#             else:
#                 return f"{bytes_val/(1024**3):.1f} G{suffix}"

#         # Update max value if needed
#         if self.max_values[pbar_id] != total:
#             self.max_values[pbar_id] = total
        
#         # Calculate progress (allow over 100% like you wanted)
#         progress = min(n / total, 1.0) if total > 0 else 0
        
#         # Generate label with icons and proper units
#         percent = int(n / total * 100) if total > 0 else 0
#         percent_str = f":material/clock_loader_40: {percent}%"
        
#         # Format current/total - only format bytes if unit is "B"
#         if unit == "B":
#             n_formatted = fmt_bytes(n)
#             total_formatted = fmt_bytes(total)
#             laps_str = f":material/laps: {n_formatted} / {total_formatted}"
#             rate_str = f":material/speed: {fmt_bytes(rate, 'B/s')}" if rate else ""
#         else:
#             # For other units (files, items, animals, etc.), show as-is
#             laps_str = f":material/laps: {int(n)} {unit} / {int(total)} {unit}"
#             rate_str = f":material/speed: {rate:.1f} {unit}/s" if rate else ""
        
#         elapsed_str = f":material/timer: {fmt_time(elapsed)}" if elapsed else ""
#         eta_str = f":material/sports_score: {fmt_time((total - n) / rate)}" if rate and total > n else ""
        
#         label = self.label_divider + self.label_divider.join(filter(None, [
#             percent_str, laps_str, rate_str, elapsed_str, eta_str
#         ]))
        
#         # Update the progress bar
#         self.update(pbar_id, n - self.states[pbar_id], text=label)


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
    vars_file = os.path.join(ADDAXAI_FILES, "AddaxAI",
                             "streamlit-AddaxAI", "vars", f"{section}.json")
    # if os.path.exists(vars_file): # temorarily disabled
    #     os.remove(vars_file)


def replace_vars(section, new_vars):
    vars_file = os.path.join(ADDAXAI_FILES, "AddaxAI",
                             "streamlit-AddaxAI", "vars", f"{section}.json")

    # # Ensure the directory exists (optional, but safe)
    # os.makedirs(os.path.dirname(vars_file), exist_ok=True)

    # Overwrite with only the new updates
    with open(vars_file, "w", encoding="utf-8") as file:
        json.dump(new_vars, file, indent=2, default=default_converter)


def update_vars(section, updates):
    # settings, settings_file = load_map()

    vars_file = os.path.join(ADDAXAI_FILES, "AddaxAI",
                             "streamlit-AddaxAI", "vars", f"{section}.json")
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
    vars_file = os.path.join(ADDAXAI_FILES, "AddaxAI",
                             "streamlit-AddaxAI", "vars", f"{section}.json")
    if not os.path.exists(vars_file):
        with open(vars_file, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

    # read section vars
    with open(vars_file, "r", encoding="utf-8") as f:
        section_vars = json.load(f)

    return section_vars
    # return {var: section_vars.get(var, None) for var in requested_vars}.values()


def load_lang_txts():
    txts_fpath = os.path.join(ADDAXAI_FILES, "AddaxAI",
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
