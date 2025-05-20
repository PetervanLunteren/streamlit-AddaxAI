
import os
import json
import streamlit as st
import sys
import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from appdirs import user_config_dir
import pandas as pd
from collections import defaultdict
import subprocess
import time
from datetime import datetime

# set global variables
AddaxAI_files = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
CLS_DIR = os.path.join(AddaxAI_files, "models", "cls")
DET_DIR = os.path.join(AddaxAI_files, "models", "det")

# set versions
with open(os.path.join(AddaxAI_files, 'AddaxAI', 'version.txt'), 'r') as file:
    current_AA_version = file.read().strip()

# print a markdown label with an icon and help text


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


# def location_selector_widget():
#     # check if the user has any known locations
#     locations, selected_index = fetch_known_locations()
#     if locations == []:
#         if st.button(":material/add_circle: Add location", key="add_new_location_button", use_container_width=False):
#             add_new_location()
#     else:
#         location_ids = [location["locationID"] for location in locations]
#         selected_location = st.selectbox(
#             "Choose a location ID",
#             options=location_ids + ["+ Add new"],
#             index=selected_index,
#         )
#         # If "Add a new location" is selected, show a dialog with a map and text input
#         if selected_location == "+ Add new":
#             add_new_location()

#         return selected_location

def location_selector_widget():
    # Initialize the popup state in session_state if it doesn't exist yet
    if "show_add_location_popup" not in st.session_state:
        st.session_state.show_add_location_popup = False

    locations, selected_index = fetch_known_locations()

    if locations == []:
        if st.button(":material/add_circle: Add location", key="add_new_location_button", use_container_width=False):
            # Set the flag to show the popup
            st.session_state.show_add_location_popup = True

    else:
        location_ids = [location["locationID"] for location in locations]
        selected_location = st.selectbox(
            "Choose a location ID",
            options=location_ids + ["+ Add new"],
            index=selected_index,
        )
        if selected_location == "+ Add new":
            st.session_state.show_add_location_popup = True
        else:
            # If user selects anything else, close popup
            st.session_state.show_add_location_popup = False

        if st.session_state.show_add_location_popup:
            add_new_location()

        return selected_location

    # If no known locations and popup should be shown
    if st.session_state.show_add_location_popup:
        add_new_location()

def print_widget_label(label_text, icon=None, help_text=None):
    if icon:
        line = f":material/{icon}: &nbsp; "
    else:
        line = ""
    # st.divider()
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


def select_model_widget(model_type, prev_selected_model):
    # prepare radio button options
    model_info = fetch_all_model_info(model_type)
    model_options = {}
    for key, info in model_info.items():
        model_options[key] = {"option": info["friendly_name"],
                              "caption": f":material/calendar_today: Released {info['release']} &nbsp;|&nbsp; "
                              f":material/code_blocks: Developed by {info['developer']} &nbsp;|&nbsp; "
                              f":material/description: {info['short_description']}"}
    selected_model = radio_buttons_with_captions(
        option_caption_dict=model_options,
        key=f"{model_type}_model",
        scrollable=True,
        default_option=prev_selected_model)

    # more info button
    friendly_name = model_info[selected_model]["friendly_name"]
    if st.button(f":material/info: More info about :grey-background[{friendly_name}]", key=f"{model_type}_model_info_button"):
        show_model_info(model_info[selected_model])

    return selected_model

# check which models are known and should be listed in the dpd
def fetch_all_model_info(type):

    # fetch
    model_info_json = os.path.join(
        AddaxAI_files, "AddaxAI", "streamlit-AddaxAI", "model_info.json")
    with open(model_info_json, "r") as file:
        model_info = json.load(file)

    # sort by release date
    sorted_det_models = dict(
        sorted(
            model_info[type].items(),
            # key=lambda item: datetime.strptime(item[1].get("release", "01-1900"), "%m-%Y"), # on release date
            key=lambda item: item[1].get(
                "friendly_name", ""),  # on frendly name
            reverse=False
        )
    )

    # return
    return sorted_det_models

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
    config_dir = user_config_dir("AddaxAI")
    settings_file = os.path.join(config_dir, "settings.json")
    current_project = "VeluweProject" # DEBUG this must be set in the settings file somewhere

    if not os.path.exists(settings_file):
        return [], 0  # No settings file yet

    with open(settings_file, "r") as f:
        settings = json.load(f)

    # Check if the project and locations exist
    if current_project in settings and "locations" in settings[current_project]:
        locations_dict = settings[current_project]["locations"]
        # Convert to a list (or keep as dict if needed)
        locations = [
            {"locationID": loc_id, **loc_info}
            for loc_id, loc_info in locations_dict.items()
        ]
        return locations, settings[current_project]['selected_location_idx']  # Assume first one is selected by default
    else:
        return [], 0  # Project or locations missing


def add_location(location_id, lat, lon):
    config_dir = user_config_dir("AddaxAI")
    settings_file = os.path.join(config_dir, "settings.json")
    current_project = "VeluweProject"  # TODO: DEBUG optionally fetch from main settings

    # Load existing settings
    if os.path.exists(settings_file):
        with open(settings_file, "r") as file:
            settings = json.load(file)
    else:
        settings = {}

    # Get existing locations or initialize
    locations = settings[current_project].get("locations", {})

    # Check if location_id is unique
    if location_id in locations:
        raise ValueError(f"Location ID '{location_id}' already exists.")

    # Add new location
    locations[location_id] = {
        "lat": lat,
        "lon": lon,
        "locationID": location_id  # redundant but keeps structure consistent
    }

    # Sort locations (optional: dicts don't preserve order unless using OrderedDict or Python 3.7+)
    sorted_locations = dict(sorted(locations.items(), key=lambda item: item[0].lower()))
    settings[current_project]["locations"] = sorted_locations
    settings[current_project]["selected_location_idx"] = list(sorted_locations.keys()).index(location_id) # update selected index

    # Save updated settings
    with open(settings_file, "w") as file:
        json.dump(settings, file, indent=2)

    # Return list of locations and index of the new one
    location_list = list(sorted_locations.values())
    selected_index = location_list.index(sorted_locations[location_id])

    return selected_index, location_list


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
    bounds = []
    if known_locations:
        # show_markers = st.checkbox("Show other locations", value=True)
        

        # add the selected location
        if st.session_state.lat_selected and st.session_state.lng_selected:
            fl.Marker(
                [st.session_state.lat_selected, st.session_state.lng_selected],
                title="Selected location",
                tooltip="Selected location",
                icon=fl.Icon(icon="camera", prefix="fa", color="darkred")
            ).add_to(m)
            bounds.append([st.session_state.lat_selected,
                          st.session_state.lng_selected])

        # add the known locations
        for loc in known_locations:
            coords = [loc["lat"], loc["lon"]]
            # if show_markers:
            fl.Marker(
                coords,
                tooltip=loc["locationID"],
                icon=fl.Icon(icon="camera", prefix="fa", color="darkblue")
            ).add_to(m)
            bounds.append(coords)
            m.fit_bounds(bounds, padding=(75, 75))
    
    else:
        # bounds = []
        # add the selected location
        if st.session_state.lat_selected and st.session_state.lng_selected:
            fl.Marker(
                [st.session_state.lat_selected, st.session_state.lng_selected],
                title="Selected location",
                tooltip="Selected location",
                icon=fl.Icon(icon="camera", prefix="fa", color="yellow")
            ).add_to(m)

            # only one marker, so set bounds to the selected location
            buffer = 0.0006
            bounds = [
                [st.session_state.lat_selected - buffer, st.session_state.lng_selected - buffer],
                [st.session_state.lat_selected + buffer, st.session_state.lng_selected + buffer]
            ]
            m.fit_bounds(bounds)

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
        print_widget_label("Enter latitude or click on the map",
                           help_text="Enter the latitude of the location.")
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
        print_widget_label("Enter longitude or click on the map",
                           help_text="Enter the longitude of the location.")
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

    print_widget_label("Enter unique location ID",
                       help_text="This ID will be used to identify the location in the system.")
    new_location_id = st.text_input(
        "Enter new Location ID",
        label_visibility="collapsed",
    )

    if new_location_id:
        if st.button(":material/save: Save location", use_container_width=True):
            if any(loc['locationID'] == new_location_id for loc in known_locations):
                st.error(
                    f"Error: The ID '{new_location_id}' is already taken. Please choose a unique ID.")
            elif st.session_state.lat_selected == 0.0 and st.session_state.lng_selected == 0.0:
                st.error(
                    "Error: Latitude and Longitude cannot be (0, 0). Please select a valid location.")
            else:
                add_location(
                    new_location_id, st.session_state.lat_selected, st.session_state.lng_selected)
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
        formatted_classes = [all_classes[0].replace('_', ' ').capitalize(
        )] + [cls.replace('_', ' ').lower() for cls in all_classes[1:]]
        output = ', '.join(
            formatted_classes[:-1]) + ', and ' + formatted_classes[-1] + "."
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
            st.write(
                f"This model requires AddaxAI version {min_version}. Your current AddaxAI version {current_AA_version} will not be able to run this model. An update is required. Update via the [Addax Data Science website](https://addaxdatascience.com/addaxai/).")
        else:
            st.write(
                f"Current version of AddaxAI (v{current_AA_version}) is able to use this model. No update required.")

def fetch_model_info(model_name):
    return json.load(open(os.path.join(CLS_DIR, model_name, "variables.json"), "r"))


def save_cls_classes(cls_model_key, slected_classes):
    # fetch
    model_info_json = os.path.join(
        AddaxAI_files, "AddaxAI", "streamlit-AddaxAI", "model_info.json")
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
    fpath = os.path.join(AddaxAI_files, "AddaxAI",
                         "streamlit-AddaxAI", "frontend", "vars.json")
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
        # Replace old file only after successful write
        os.replace(temp_fpath, fpath)
    except IOError as e:
        st.warning(f"Error writing to file: {e}")


def load_vars():
    """Reads the data from the JSON file and returns it as a dictionary."""

    # Define file path
    # fpath = os.path.join("/Users/peter/Desktop/streamlit_app/frontend", "vars.json")
    fpath = os.path.join(AddaxAI_files, "AddaxAI",
                         "streamlit-AddaxAI", "frontend", "vars.json")

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
    txts_fpath = os.path.join(AddaxAI_files, "AddaxAI",
                              "streamlit-AddaxAI", "frontend", "txts.json")
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
