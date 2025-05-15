
import os
import json
import streamlit as st
import sys
import folium as fl
from streamlit_folium import st_folium
import streamlit as st
import pandas as pd
from collections import defaultdict
import subprocess
import time
from datetime import datetime

# set global variables
AddaxAI_files = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
CLS_DIR = os.path.join(AddaxAI_files, "models", "cls")
DET_DIR = os.path.join(AddaxAI_files, "models", "det")

# set versions
with open(os.path.join(AddaxAI_files, 'AddaxAI', 'version.txt'), 'r') as file:
    current_AA_version = file.read().strip()

# print a markdown label with an icon and help text
def print_widget_label(label_text, icon = None, help_text = None):
    if icon:
        line = f":material/{icon}: &nbsp; "
    else:
        line = ""
    # st.divider()
    st.markdown(f"{line}**{label_text}**", help = help_text)

# # check which models are known and should be listed in the dpd
# def fetch_model_info():
#     return sorted([subdir for subdir in os.listdir(CLS_DIR) if os.path.isdir(os.path.join(CLS_DIR, subdir))])

# check which models are known and should be listed in the dpd
def fetch_all_model_info(type):
    
    # fetch
    model_info_json = os.path.join(AddaxAI_files, "AddaxAI", "app", "model_info.json")
    with open(model_info_json, "r") as file:
        model_info = json.load(file)

    # sort by release date
    sorted_det_models = dict(
        sorted(
            model_info[type].items(),
            # key=lambda item: datetime.strptime(item[1].get("release", "01-1900"), "%m-%Y"), # on release date
            key=lambda item: item[1].get("friendly_name", ""), # on frendly name
            reverse=False 
        )
    )
    
    # return
    return sorted_det_models

# run a parallel tkinter script to select a folder
def select_folder():
    result = subprocess.run([sys.executable, os.path.join(AddaxAI_files, "AddaxAI", "app", "frontend", "folder_selector.py")], capture_output=True, text=True)
    folder_path = result.stdout.strip()
    if folder_path != "" and result.returncode == 0:
        return folder_path
    else:
        return None

# check if the user needs an update
def needs_EA_update(required_version):
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

def fetch_known_locations():
    home_folder = os.path.expanduser("~")
    file_path = os.path.join(home_folder, "AddaxAI_locations.json")
    
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
            return data["locations"], data["selected"]
    else:
        # If the file doesn't exist, create an empty one with default values
        with open(file_path, "w") as file:
            json.dump({"selected": 0, "locations": []}, file, indent=4)
        return [], 0  # Return empty list and default selected index

def add_location(location_id, lat, lng):
    home_folder = os.path.expanduser("~")
    file_path = os.path.join(home_folder, "AddaxAI_locations.json")
    
    # Fetch existing locations and the selected index
    locations, selected_index = fetch_known_locations()

    # Ensure the new location ID is unique
    if any(location["id"] == location_id for location in locations):
        raise ValueError(f"Location ID '{location_id}' already exists.")
    
    # Append the new location
    locations.append({"id": location_id, "coords": [lat, lng]})

    # Sort locations alphabetically by location ID
    locations.sort(key=lambda loc: loc["id"].lower())  # Sort case-insensitive

    # Update the selected location index (you can modify this based on your logic)
    selected_index = next(i for i, loc in enumerate(locations) if loc["id"] == location_id)

    # Save the updated locations and selected index to the file
    with open(file_path, "w") as file:
        json.dump({"selected": selected_index, "locations": locations}, file, indent=4)

    return selected_index, locations  # Return u

# # show popup with model information
@st.dialog("New location", width="large")
def add_new_location():
    
    # Initialize session state for lat/lng if not set
    if "lat_selected" not in st.session_state:
        st.session_state.lat_selected = None
    if "lng_selected" not in st.session_state:
        st.session_state.lng_selected = None

    # List of locations
    known_locations, _ = fetch_known_locations()
    
    # Create the base map with the default terrain layer
    m = fl.Map(
        location=[0, 0],
        zoom_start=1,
        control_scale=True
    )

    # Add the terrain layer
    fl.TileLayer(
        tiles='https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.jpg',
        attr='© Stamen, © OpenStreetMap',
        name='Stamen Terrain',
        overlay=False,
        control=True
    ).add_to(m)

    # Add the satellite layer
    fl.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='© Esri',
        name='Esri Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # Add layer control to switch between them
    fl.LayerControl().add_to(m)

    # Add markers
    if known_locations:
        # show_markers = st.checkbox("Show other locations", value=True)
        bounds = []

        # add the selected location
        if st.session_state.lat_selected and st.session_state.lng_selected:
            fl.Marker(
                [st.session_state.lat_selected, st.session_state.lng_selected],
                title="Selected location",
                tooltip="Selected location",
                icon=fl.Icon(icon="camera", prefix="fa", color="Yellow")
            ).add_to(m)
            bounds.append([st.session_state.lat_selected, st.session_state.lng_selected])

        # add the known location
        for loc in known_locations:
            coords = loc.get("coords", None)
            # if show_markers:                
            fl.Marker(
                coords,
                tooltip=loc["id"],
                icon=fl.Icon(icon="camera", prefix="fa", color="darkblue")
            ).add_to(m)
            bounds.append(coords)

    # Fit map to markers with extra padding
    if bounds:
        m.fit_bounds(bounds, padding=(75, 75))

    # Add lat/lng popup on click
    m.add_child(fl.LatLngPopup())

    # Render map
    map_data = st_folium(m, height=300, width=700)

    # Update lat/lng when clicking on map
    if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
        st.session_state.lat_selected = map_data["last_clicked"]["lat"]
        st.session_state.lng_selected = map_data["last_clicked"]["lng"]
        fl.Marker(
            [st.session_state.lat_selected, st.session_state.lng_selected],
            title="Selected location",
            tooltip="Selected location",
            icon=fl.Icon(icon="camera", prefix="fa", color="yellow")
        ).add_to(m)
        st.rerun()

    # User input section
    col1, col2 = st.columns([1, 1])

    with col1:
        print_widget_label("Enter latitude or click on the map", help_text="Enter the latitude of the location.")
        old_lat = st.session_state.get("lat_selected", 0.0)
        new_lat = st.number_input(
            "Enter latitude or click on the map",
            value=st.session_state.lat_selected,
            format="%.6f",
            step=0.000001,
            min_value=-90.0,
            max_value=90.0,
            label_visibility="collapsed",
        )
        st.session_state.lat_selected = new_lat
        if new_lat != old_lat:
            st.rerun()

    with col2:
        print_widget_label("Enter longitude or click on the map", help_text= "Enter the longitude of the location.")
        old_lng = st.session_state.get("lng_selected", 0.0)
        new_lng = st.number_input(
            "Enter longitude or click on the map",
            value=st.session_state.lng_selected,
            format="%.6f",
            step=0.000001,
            min_value=-180.0,
            max_value=180.0,
            label_visibility="collapsed",
        )
        st.session_state.lng_selected = new_lng
        if new_lng != old_lng:
            st.rerun()
    
    print_widget_label("Enter unique location ID", help_text= "This ID will be used to identify the location in the system.")
    new_location_id = st.text_input(
        "Enter new Location ID",
        label_visibility="collapsed",
    )
    
    if new_location_id:
        if st.button(":material/save: Save location", use_container_width=True):
            if any(loc['id'] == new_location_id for loc in known_locations):
                st.error(f"Error: The ID '{new_location_id}' is already taken. Please choose a unique ID.")
            elif st.session_state.lat_selected == 0.0 and st.session_state.lng_selected == 0.0:
                st.error("Error: Latitude and Longitude cannot be (0, 0). Please select a valid location.")
            else:
                add_location(new_location_id, st.session_state.lat_selected, st.session_state.lng_selected)
                new_location_id = None
                st.session_state.lat_selected = None
                st.session_state.lng_selected = None
                st.session_state.metadata_checked = False
                st.rerun()

# show popup with model information
@st.dialog("Model information", width="large")
def show_model_info(model_info):
    friendly_name = model_info.get('friendly_name', None)
    if friendly_name and friendly_name != "":
        st.write("")
        print_widget_label("Name", "rocket_launch")
        st.write(friendly_name)
    # print_widget_label("Name", "rocket_launch")
    # st.write(model_name)
    
    description = model_info.get('description', None)
    if description and description != "":
        st.write("")
        print_widget_label("Description", "history_edu")
        st.write(description)
    
    all_classes = model_info.get('all_classes', None)
    if all_classes and all_classes != []:
        st.write("")
        print_widget_label("Classes", "pets")
        formatted_classes = [all_classes[0].replace('_', ' ').capitalize()] + [cls.replace('_', ' ').lower() for cls in all_classes[1:]]
        output = ', '.join(formatted_classes[:-1]) + ', and ' + formatted_classes[-1] + "."
        st.write(output)
    
    developer = model_info.get('developer', None)
    if developer and developer != "":
        st.write("")
        print_widget_label("Developer", "code")
        st.write(developer)
    
    owner = model_info.get('owner', None)
    if owner and owner != "":
        st.write("")
        print_widget_label("Owner", "account_circle")
        st.write(owner)  
    
    info_url = model_info.get('info_url', None)
    if info_url and info_url != "":
        st.write("")
        print_widget_label("More information", "info")
        st.write(info_url)
    
    citation = model_info.get('citation', None)
    if citation and citation != "":
        st.write("")
        print_widget_label("Citation", "article")
        st.write(citation)
    
    license = model_info.get('license', None)
    if license and license != "":
        st.write("")
        print_widget_label("License", "copyright")
        st.write(license)
    
    min_version = model_info.get('min_version', None)
    if min_version and min_version != "":
        st.write("")
        print_widget_label("Required AddaxAI version", "verified")
        needs_EA_update_bool = needs_EA_update(min_version)
        if needs_EA_update_bool:
            st.write(f"This model requires AddaxAI version {min_version}. Your current AddaxAI version {current_AA_version} will not be able to run this model. An update is required. Update via the [Addax Data Science website](https://addaxdatascience.com/addaxai/).")
        else:
            st.write(f"Current version of AddaxAI (v{current_AA_version}) is able to use this model. No update required.")

# st.selectbox("Select model", ["Model 1", "Model 2", "Model 3"])
# if st.button("Deploy model"):
#     # Simulate deployment process with a progress bar
#     progress_bar = st.progress(0, text="Model deployment in progress... :material/menu:")

#     # Simulate a task with time delay (e.g., model deployment)
#     import time
#     for i in range(1, 101):
#         time.sleep(0.05)  # Simulate work by waiting
#         progress_text = f"Deploy model... " \
#                         f"{'&nbsp;' * 10} :material/clock_loader_40: {i}% " \
#                         f"{'&nbsp;' * 10} :material/timer: {i} < {100-i} " \
#                         f"{'&nbsp;' * 10} :material/speed: {i/10} it/s " \
#                         f"{'&nbsp;' * 10} :material/memory: GPU"
#         progress_bar.progress(i, text=progress_text)
#     st.success("Model deployment complete!")

def fetch_model_info(model_name):
    return json.load(open(os.path.join(CLS_DIR, model_name, "variables.json"), "r"))

def save_cls_classes(cls_model_key, slected_classes):
    # fetch
    model_info_json = os.path.join(AddaxAI_files, "AddaxAI", "app", "model_info.json")
    with open(model_info_json, "r") as file:
        model_info = json.load(file)
    model_info['cls'][cls_model_key]['selected_classes'] = slected_classes
    
    # save
    with open(model_info_json, "w") as file:
        json.dump(model_info, file, indent=4)
    

def save_global_vars(new_data):
    """Updates or creates a JSON file with multiple key-value pairs.
    
    - `new_data` should be a dictionary.
    - If the file doesn't exist, it creates it.
    - If keys exist, their values are updated.
    - If keys don’t exist, they are added.
    """

    if not isinstance(new_data, dict):
        raise ValueError("Expected a dictionary as input")

    # fpath = os.path.join("/Users/peter/Desktop/streamlit_app/frontend", "vars.json")
    fpath = os.path.join(AddaxAI_files, "AddaxAI", "app", "frontend", "vars.json")
    temp_fpath = fpath + ".tmp"  # Temporary file for atomic write

    # Load existing data or start with an empty dictionary
    try:
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as file:
                data = json.load(file)
        else:
            data = {}
    except (json.JSONDecodeError, IOError):
        data = {}  # Handle corrupt file or read errors

    # Update data with new values
    data.update(new_data)

    # Write updated data to a temporary file first (atomic write)
    try:
        with open(temp_fpath, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        os.replace(temp_fpath, fpath)  # Replace old file only after successful write
    except IOError as e:
        st.warning(f"Error writing to file: {e}")

def load_vars():
    """Reads the data from the JSON file and returns it as a dictionary."""
    
    # Define file path
    # fpath = os.path.join("/Users/peter/Desktop/streamlit_app/frontend", "vars.json")
    fpath = os.path.join(AddaxAI_files, "AddaxAI", "app", "frontend", "vars.json")
    
    # Check if the file exists and load its data
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
                return data
            except json.JSONDecodeError:
                return {}  # Return empty dictionary if the file is corrupt or empty
    else:
        return {}  # Return empty dictionary if the file doesn't exist


def load_txts():
    # txts_fpath = "/Users/peter/Desktop/streamlit_app/frontend/txts.json"
    txts_fpath = os.path.join(AddaxAI_files, "AddaxAI", "app", "frontend", "txts.json")
    with open(txts_fpath, "r", encoding="utf-8") as file:
        txts = json.load(file)
    return txts

# def language_options(txts, lang):
#     _, col2 = st.columns([4, 1])
#     with col2:
#         with st.popover(":material/language:", use_container_width=True):
#             # Language settings
#             # st.subheader(":material/language: Language", divider="grey")
#             st.markdown("&nbsp;" * 50)

#             lang_options = txts["languages"]
#             lang_idx = list(lang_options.keys()).index(lang)
#             lang_selected = st.selectbox(
#                 "Language",
#                 options=lang_options.keys(),
#                 format_func=lambda option: lang_options[option],
#                 index=lang_idx,
#                 label_visibility="collapsed"
#             )
            
#             save_btn = st.button("    :material/save: Save settings     ", use_container_width=True,
#                                  type = "primary")
#             if save_btn:
#                 save_vars({"lang": lang_selected})
#                 st.rerun()
    

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





# from streamlit_extras.tags import tagger_component 
# tagger_component("Here is a feature request", ["p2", "🚩triaged", "backlog"])
# tagger_component(
#     "Here are colored tags",
#     ["Project: DDL", "Location: Duinen", "Deployment: Dep01"],
#     color_name=["blue", "#086164", "lightblue"],
# )
# tagger_component(
#     "Annotate the feature",
#     ["hallucination"],
#     color_name=["blue"],
# )
