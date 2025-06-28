
from streamlit_tree_select import tree_select
import os
import json
import streamlit as st
import sys
import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from appdirs import user_config_dir
import pandas as pd
import statistics
from collections import defaultdict
import subprocess
import string
import math
# import time
from datetime import datetime, time
# from datetime import datetime
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
from st_flexible_callout_elements import flexible_callout
import random
from PIL.ExifTags import TAGS
from hachoir.metadata import extractMetadata
from hachoir.parser import createParser
import piexif
from cameratraps.megadetector.detection.video_utils import VIDEO_EXTENSIONS
from cameratraps.megadetector.utils.path_utils import IMG_EXTENSIONS

# set global variables
AddaxAI_files = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
CLS_DIR = os.path.join(AddaxAI_files, "models", "cls")
DET_DIR = os.path.join(AddaxAI_files, "models", "det")

# fetch camera IDs
config_dir = user_config_dir("AddaxAI")
settings_file = os.path.join(config_dir, "settings.json")

# set versions
with open(os.path.join(AddaxAI_files, 'AddaxAI', 'version.txt'), 'r') as file:
    current_AA_version = file.read().strip()

# print a markdown label with an icon and help text

############################
### DEPLOYMENT UTILITIES ###
############################

# this widget lets the user select a project from a dropdown menu
# or add a new pojrect via popover


def project_selector_widget():

    # check what is already known and selected
    projects, selected_projectID = fetch_known_projects()

    # if first project, show only button and no dropdown
    if projects == {}:
        add_new_project_popover("Define your first project")

    # if there are projects, show dropdown and button
    else:
        col1, col2 = st.columns([3, 1])

        # dropdown for existing projects
        with col1:
            options = list(projects.keys())
            selected_index = options.index(
                selected_projectID) if selected_projectID in options else 0

            # overwrite selected_projectID if user has selected a different project
            selected_projectID = st.selectbox(
                "Existing projects",
                options=options,
                index=selected_index,
                label_visibility="collapsed"
            )

        # popover to add a new project
        with col2:
            add_new_project_popover("New")

        # adjust the selected project
        settings, _ = load_settings()
        if settings["vars"]["analyse_advanced"]["selected_projectID"] != selected_projectID:
            settings["vars"]["analyse_advanced"]["selected_projectID"] = selected_projectID
            with open(settings_file, "w") as file:
                json.dump(settings, file, indent=2)
            st.rerun()

        # return
        return selected_projectID

# this widget lets the user select a location from a dropdown menu
# or add a new location via popover


def location_selector_widget():

    # load settings
    coords_found_in_exif, exif_lat, exif_lng = fetch_vars(section="analyse_advanced",  # SESSION
                                                          requested_vars=["coords_found_in_exif", "exif_lat", "exif_lng"])

    # check what is already known and selected
    locations, location = fetch_known_locations()

    # # calculate distance to closest known locations if coordinates are found in metadata
    # if "closest_location" not in st.session_state:
    #     st.session_state.closest_location = None
    # if st.session_state.coords_found_in_exif:
    #     st.session_state.closest_location = match_locations((st.session_state.exif_lat, st.session_state.exif_lng), locations)
    if coords_found_in_exif:  # SESSION
        closest_location = match_locations((exif_lat, exif_lng), locations)

    # if first location, show only button and no dropdown
    if locations == {}:
        add_new_location_popover("Define your first location")

        # # show info box if coordinates are found in metadata
        # if st.session_state.coords_found_in_exif:
        #     info_box(f"Coordinates ({st.session_state.exif_lat:.5f}, {st.session_state.exif_lng:.5f}) were automatically extracted from the image metadata. They will be pre-filled when adding the new location.")
        if coords_found_in_exif:  # SESSION
            info_box(
                f"Coordinates ({exif_lat:.5f}, {exif_lng:.5f}) were automatically extracted from the image metadata. They will be pre-filled when adding the new location.")

    # if there are locations, show dropdown and button
    else:
        col1, col2 = st.columns([3, 1])

        # dropdown for existing locations
        with col1:
            options = list(locations.keys())

            # set to last selected location if it exists
            selected_index = options.index(
                location) if location in options else 0

            # # if coordinates are found in metadata, pre-select the closest location
            # if st.session_state.coords_found_in_exif and st.session_state.closest_location is not None:
            #     closes_location_name, _ = st.session_state.closest_location
            #     selected_index = options.index(closes_location_name) if closes_location_name in options else 0

            if coords_found_in_exif and closest_location is not None:
                # if coordinates are found in metadata, pre-select the closest location
                closest_location_name, _ = closest_location
                selected_index = options.index(
                    closest_location_name) if closest_location_name in options else 0

            # create the selectbox
            location = st.selectbox(
                "Choose a location ID",
                options=options,
                index=selected_index,
                label_visibility="collapsed"
            )

        # popover to add a new location
        with col2:
            add_new_location_popover("New")

        # # info box if coordinates are found in metadata
        # if st.session_state.coords_found_in_exif:

        #     # define message based on whether a closest location was found
        #     message = f"Coordinates extracted from image metadata: ({st.session_state.exif_lat:.5f}, {st.session_state.exif_lng:.5f}). "
        #     if st.session_state.closest_location is not None:
        #         name, dist = st.session_state.closest_location
        #         message += f"Matches known location <i>{name}</i>, about {dist} meters away."
        #     else:
        #         message += f"No known location found within 50 meters."
        #     info_box(message)

        # info box if coordinates are found in metadata
        if coords_found_in_exif:

            # define message based on whether a closest location was found
            message = f"Coordinates extracted from image metadata: ({exif_lat:.5f}, {exif_lng:.5f}). "
            if closest_location is not None:
                name, dist = closest_location
                message += f"Matches known location <i>{name}</i>, about {dist} meters away."
            else:
                message += f"No known location found within 50 meters."
            info_box(message)

        # return
        return location


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance in meters between two lat/lng points."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * \
        math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def match_locations(known_point, locations, max_distance_meters=50):
    """
    Find the closest known location within a max distance (default 50 meters).

    Parameters:
    - known_point: tuple (lat, lon)
    - locations: dict of known locations with lat/lon
    - max_distance_meters: float, max distance to consider a match

    Returns:
    - Tuple (location_name, distance_in_meters_rounded) or None if no match
    """
    lat_known, lon_known = known_point
    candidates = []

    for name, data in locations.items():
        lat = data["lat"]
        lon = data["lon"]
        dist = haversine_distance(lat_known, lon_known, lat, lon)
        if dist <= max_distance_meters:
            candidates.append((name, dist))

    if not candidates:
        return None

    # Get closest match
    closest = min(candidates, key=lambda x: x[1])
    return (closest[0], round(closest[1]))


def datetime_selector_widget():

    # Initialize the session state for min_datetime_found if not set
    if "min_datetime_found" not in st.session_state:
        st.session_state.min_datetime_found = None
        # if present, it will be of format "datetime.datetime(2013, 1, 17, 13, 5, 21)"

    # Pre-fill defaults
    default_date = None
    default_hour = "--"
    default_minute = "--"
    default_second = "--"

    if st.session_state.min_datetime_found:
        # # In case it's stored as a string like "datetime.datetime(2013, 1, 17, 13, 5, 21)"
        # if isinstance(st.session_state.min_datetime_found, str):
        #     st.session_state.min_datetime_found = eval(st.session_state.min_datetime_found)

        dt = st.session_state.min_datetime_found
        default_date = dt.date()
        default_hour = f"{dt.hour:02d}"
        default_minute = f"{dt.minute:02d}"
        default_second = f"{dt.second:02d}"

    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        selected_date = st.date_input("Date", value=default_date)

    # Format options as zero-padded strings
    hour_options = ["--"] + [f"{i:02d}" for i in range(24)]
    minute_options = ["--"] + [f"{i:02d}" for i in range(60)]
    second_options = ["--"] + [f"{i:02d}" for i in range(60)]

    with col2:
        selected_hour = st.selectbox(
            "Hour", options=hour_options, index=hour_options.index(default_hour))

    with col3:
        selected_minute = st.selectbox(
            "Minute", options=minute_options, index=minute_options.index(default_minute))

    with col4:
        selected_second = st.selectbox(
            "Second", options=second_options, index=second_options.index(default_second))

    if st.session_state.min_datetime_found:
        info_box(
            f"Prefilled with the earliest datetime found in the metadata. If adjusted, the other datetimes will update automatically.",
            icon=":material/info:")

    # Check if all values are selected properly
    if (
        selected_date
        and selected_hour != "--"
        and selected_minute != "--"
        and selected_second != "--"
    ):
        selected_time = time(
            hour=int(selected_hour),
            minute=int(selected_minute),
            second=int(selected_second)
        )
        selected_datetime = datetime.combine(selected_date, selected_time)
        # st.write("Selected datetime:", selected_datetime)

        # deployment will only be added once the user has pressed the "ANALYSE" button

        return selected_datetime


def fetch_known_projects():
    settings, _ = load_settings()
    projects = settings["projects"]
    selected_projectID = settings["vars"]["analyse_advanced"].get(
        "selected_projectID")
    return projects, selected_projectID


def fetch_known_locations():
    settings, _ = load_settings()
    selected_projectID = settings["vars"]["analyse_advanced"].get(
        "selected_projectID")
    project = settings["projects"][selected_projectID]
    selected_location = settings["vars"]["analyse_advanced"].get(
        "selected_location")
    locations = project["locations"]
    return locations, selected_location


def fetch_known_deployments():
    settings, _ = load_settings()
    selected_projectID = settings["vars"]["analyse_advanced"].get(
        "selected_projectID")
    project = settings["projects"][selected_projectID]
    location = project["location"]
    location = project["locations"][location]
    deployments = location["deployments"]
    selected_deployment = location["selected_deployment"]
    return deployments, selected_deployment


def generate_deployment_id():

    # Create a consistent 5-char hash from the datetime and some randomness
    rand_str_1 = ''.join(random.choices(
        string.ascii_uppercase + string.digits, k=5))
    rand_str_2 = ''.join(random.choices(
        string.ascii_uppercase + string.digits, k=5))

    # Combine into deployment ID
    return f"dep-{rand_str_1}-{rand_str_2}"

###### WORKING STEPPER BAR ######


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


def fetch_step(section):
    # load
    # settings, _ = load_settings()
    # selected_projectID = settings["vars"]["analyse_advanced"].get("selected_projectID")
    # project_vars = settings["projects"][project]["vars"]
    # step = project_vars.get("step", 0)
    # return step

    settings, _ = load_settings()
    vars = settings["vars"].get(section)
    step = vars.get("step", 0)
    return step


# def update_vars(section, updates):

#     # load
#     settings, _ = load_settings()
#     vars = settings["vars"][section]

#     # st.write("DEBUG - Project Variables")
#     # st.write(project_vars)

#     for key, value in updates.items():
#         if isinstance(value, datetime):
#             # If the value is a datetime, convert it to ISO format
#             value = value.isoformat()
#         vars[key] = value

#     vars.update(updates)

#     # Save updated settings
#     with open(settings_file, "w") as file:
#         json.dump(settings, file, indent=2)

def default_converter(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(
        f"Object of type {obj.__class__.__name__} is not JSON serializable")


def update_vars(section, updates):
    settings, settings_file = load_settings()

    if "vars" not in settings:
        settings["vars"] = {}
    if section not in settings["vars"]:
        settings["vars"][section] = {}

    # Update only the section with any type of values
    settings["vars"][section].update(updates)

    # Use `default=default_converter` to catch any lingering datetime objects
    with open(settings_file, "w") as file:
        json.dump(settings, file, indent=2, default=default_converter)


def add_deployment(datetime):

    settings, _ = load_settings()
    selected_folder = settings["selected_folder"]
    selected_projectID = settings["vars"]["analyse_advanced"].get(
        "selected_projectID")
    project = settings["projects"][selected_projectID]
    location = project["location"]
    location = project["locations"][location]
    deployments = location["deployments"]

    # generate a unique deployment ID
    deployment_id = generate_deployment_id()

    # Add new deployment
    deployments[deployment_id] = {
        "deploymentStart": datetime.isoformat(),
        "deploymentEnd": None,  # initially set to None
        "path": selected_folder,
    }

    # # Sort deployments (optional: dicts don't preserve order unless using OrderedDict or Python 3.7+)
    # sorted_deployments = dict(sorted(deployments.items(), key=lambda item: item[0].lower()))
    # settings["projects"][project]["locations"][location]["deployments"] = sorted_deployments
    # settings["projects"][project]["locations"][location]["selected_deployment"] = deployment_id

    # Save updated settings
    with open(settings_file, "w") as file:
        json.dump(settings, file, indent=2)

    # Return list of deployments and index of the new one
    deployment_list = list(deployments.values())
    selected_index = deployment_list.index(deployments[deployment_id])

    return selected_index, deployment_list


def add_location(location_id, lat, lon):

    settings, _ = load_settings()
    selected_projectID = settings["vars"]["analyse_advanced"].get(
        "selected_projectID")
    project = settings["projects"][selected_projectID]
    locations = project["locations"]

    # Check if location_id is unique
    if location_id in locations.keys():
        raise ValueError(
            f"Location ID '{location_id}' already exists. Please choose a unique ID, or select existing project from dropdown menu.")

    # Add new location
    locations[location_id] = {
        "lat": lat,
        "lon": lon,
        "selected_deployment": None,
        "deployments": {},
    }

    # Sort locations (optional: dicts don't preserve order unless using OrderedDict or Python 3.7+)
    sorted_locations = dict(
        sorted(locations.items(), key=lambda item: item[0].lower()))
    settings["projects"][project]["locations"] = sorted_locations
    settings["projects"][project]["location"] = location_id

    # Save updated settings
    with open(settings_file, "w") as file:
        json.dump(settings, file, indent=2)

    # Return list of locations and index of the new one
    location_list = list(sorted_locations.values())
    selected_index = location_list.index(sorted_locations[location_id])

    return selected_index, location_list


# # show popup with model information
def add_project(projectID, comments):

    settings, settings_file = load_settings()
    projects = settings["projects"]
    projectIDs = projects.keys()

    # st.write(projectIDs)

    # Check if project_id is unique
    if projectID in projectIDs:
        raise ValueError(
            f"project ID '{projectID}' already exists. Please choose a unique ID, or select existing project from dropdown menu.")

    # Add new project
    projects[projectID] = {
        "comments": comments,
        # "location": None,
        # "vars": {},
        "locations": {},
    }

    settings["projects"] = projects  # add project
    # update selected project
    settings["vars"]["analyse_advanced"]["selected_projectID"] = projectID

    # Save updated settings
    with open(settings_file, "w") as file:
        json.dump(settings, file, indent=2)

    # Return list of projects and index of the new one
    project_list = list(projects.values())
    selected_index = project_list.index(projects[projectID])

    return selected_index, project_list


def add_new_project_popover(txt):
    # use st.empty to create a popover container
    # so that it can be closed on button click
    # and the popover can be reused
    popover_container = st.empty()
    with popover_container.container():
        with st.popover(f":material/add_circle: {txt}",
                        help="Define a new project",
                        use_container_width=True):

            # fetch known projects IDs
            known_projects, _ = fetch_known_projects()

            # input for project ID
            print_widget_label("Unique project ID",
                               help_text="This ID will be used to identify the project in the system.")
            project_id = st.text_input(
                "project ID", max_chars=50, label_visibility="collapsed")
            project_id = project_id.strip()

            # input for optional comments
            print_widget_label("Optionally add any comments or notes",
                               help_text="This is a free text field where you can add any comments or notes about the project.")
            comments = st.text_area(
                "Comments", height=150, label_visibility="collapsed")
            comments = comments.strip()

            # button to save project
            if st.button(":material/save: Save project", use_container_width=True):

                # check validity
                if not project_id.strip():
                    st.error("project ID cannot be empty.")
                elif project_id in list(known_projects.keys()):
                    st.error(
                        f"Error: The ID '{project_id}' is already taken. Please choose a unique ID, or select the existing project from dropdown menu.")
                else:

                    # if all good, add project
                    add_project(project_id, comments)

                    # reset session state variables before reloading
                    st.session_state.clear()
                    popover_container.empty()
                    st.rerun()


def add_new_location_popover(txt):

    # use st.empty to create a popover container
    # so that it can be closed on button click
    # and the popover can be reused
    popover_container = st.empty()
    with popover_container.container():
        with st.popover(f":material/add_circle: {txt}",
                        help="Define a new location",
                        use_container_width=True):

            # init vars
            lat_selected, lng_selected, exif_set, coords_found_in_exif = fetch_vars(section="analyse_advanced",  # SESSION`
                                                                                    requested_vars=["lat_selected", "lng_selected", "exif_set", "coords_found_in_exif"])

            # # init session state vars
            # if "lat_selected" not in st.session_state:
            #     st.session_state.lat_selected = None
            # if "lng_selected" not in st.session_state:
            #     st.session_state.lng_selected = None
            # if "exif_set" not in st.session_state:
            #     st.session_state.exif_set = False
            # if "coords_found_in_exif" not in st.session_state:
            #     st.session_state.coords_found_in_exif = False

            # update values if coordinates found in metadata
            if st.session_state.coords_found_in_exif:
                info_box(
                    f"Coordinates from metadata have been preselected ({st.session_state.exif_lat:.6f}, {st.session_state.exif_lng:.6f}).")
                if not st.session_state.exif_set:
                    st.session_state.lat_selected = st.session_state.exif_lat
                    st.session_state.lng_selected = st.session_state.exif_lng
                    st.session_state.exif_set = True
                    st.rerun()

            # fetch known locations
            known_locations, _ = fetch_known_locations()

            # base map
            m = fl.Map(
                location=[0, 0],
                zoom_start=1,
                control_scale=True
            )

            # terrain layer
            fl.TileLayer(
                tiles='https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.jpg',
                attr='© Stamen, © OpenStreetMap',
                name='Stamen Terrain',
                overlay=False,
                control=True
            ).add_to(m)

            # satellite layer
            fl.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='© Esri',
                name='Esri Satellite',
                overlay=False,
                control=True
            ).add_to(m)

            # layer control
            fl.LayerControl().add_to(m)

            # add markers
            bounds = []
            if known_locations:

                # add the selected location
                if st.session_state.lat_selected and st.session_state.lng_selected:
                    fl.Marker(
                        [st.session_state.lat_selected,
                            st.session_state.lng_selected],
                        title="Selected location",
                        tooltip="Selected location",
                        icon=fl.Icon(icon="camera", prefix="fa",
                                     color="darkred")
                    ).add_to(m)
                    bounds.append([st.session_state.lat_selected,
                                   st.session_state.lng_selected])

                # add the other known locations
                for location_id, location_info in known_locations.items():
                    coords = [location_info["lat"], location_info["lon"]]
                    fl.Marker(
                        coords,
                        tooltip=location_id,
                        icon=fl.Icon(icon="camera", prefix="fa",
                                     color="darkblue")
                    ).add_to(m)
                    bounds.append(coords)
                    m.fit_bounds(bounds, padding=(75, 75))

            else:

                # add the selected location
                if st.session_state.lat_selected and st.session_state.lng_selected:
                    fl.Marker(
                        [st.session_state.lat_selected,
                            st.session_state.lng_selected],
                        title="Selected location",
                        tooltip="Selected location",
                        icon=fl.Icon(icon="camera", prefix="fa",
                                     color="darkred")
                    ).add_to(m)

                    # only one marker so set bounds to the selected location
                    buffer = 0.001
                    bounds = [
                        [st.session_state.lat_selected - buffer,
                            st.session_state.lng_selected - buffer],
                        [st.session_state.lat_selected + buffer,
                            st.session_state.lng_selected + buffer]
                    ]
                    m.fit_bounds(bounds)

            # fit map to markers with some extra padding
            if bounds:
                m.fit_bounds(bounds, padding=(75, 75))

            # add brief lat lng popup on mouse click
            m.add_child(fl.LatLngPopup())

            # render map in center
            _, map_col, _ = st.columns([0.025, 0.95, 0.025])
            with map_col:
                map_data = st_folium(m, height=300, width=700)
            # map_data = st_folium(m, height=300, width=700)

            # update lat lng widgets when clicking on map
            if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
                st.session_state.lat_selected = map_data["last_clicked"]["lat"]
                st.session_state.lng_selected = map_data["last_clicked"]["lng"]
                fl.Marker(
                    [st.session_state.lat_selected, st.session_state.lng_selected],
                    title="Selected location",
                    tooltip="Selected location",
                    icon=fl.Icon(icon="camera", prefix="fa", color="green")
                ).add_to(m)
                st.rerun()

            # user input
            col1, col2 = st.columns([1, 1])

            # lat
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

            # lng
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

            # location ID
            print_widget_label("Enter unique location ID",
                               help_text="This ID will be used to identify the location in the system.")
            new_location_id = st.text_input(
                "Enter new Location ID",
                label_visibility="collapsed",
            )
            new_location_id = new_location_id.strip()

            # button to save location
            if st.button(":material/save: Save location", use_container_width=True):

                # check validity
                if new_location_id == "":
                    st.error("Location ID cannot be empty.")
                elif new_location_id in known_locations.keys():
                    st.error(
                        f"Error: The ID '{new_location_id}' is already taken. Please choose a unique ID or select the required location ID from the dropdown menu.")
                elif st.session_state.lat_selected == 0.0 and st.session_state.lng_selected == 0.0:
                    st.error(
                        "Error: Latitude and Longitude cannot be (0, 0). Please select a valid location.")
                else:

                    # if all good, add location
                    add_location(
                        new_location_id, st.session_state.lat_selected, st.session_state.lng_selected)
                    new_location_id = None

                    # reset session state variables before reloading
                    st.session_state.clear()
                    st.session_state.coords_found_in_exif = False
                    popover_container.empty()
                    st.rerun()


def browse_directory_widget(selected_folder):
    col1, col2 = st.columns([1, 3], vertical_alignment="center")
    with col1:
        if st.button(":material/folder: Browse", key="folder_select_button", use_container_width=True):
            selected_folder = select_folder()
            # update_vars("selected_folder", selected_folder)
            # save_global_vars({"selected_folder": selected_folder})
            update_vars(section="analyse_advanced",
                        updates={"selected_folder": selected_folder})

            # reset session state variables
            st.session_state.clear()

    if not selected_folder:
        with col2:
            st.write('<span style="color: grey;"> None selected...</span>',
                     unsafe_allow_html=True)
    else:
        with col2:
            folder_short = "..." + \
                selected_folder[-45:] if len(
                    selected_folder) > 45 else selected_folder
            st.markdown(
                f'Selected folder <code style="color:#086164; font-family:monospace;">{folder_short}</code>', unsafe_allow_html=True)
    return selected_folder


def select_folder():
    result = subprocess.run([sys.executable, os.path.join(
        AddaxAI_files, "AddaxAI", "streamlit-AddaxAI", "frontend", "folder_selector.py")], capture_output=True, text=True)
    folder_path = result.stdout.strip()
    if folder_path != "" and result.returncode == 0:
        return folder_path
    else:
        return None


#######################
### MODEL UTILITIES ###
#######################

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


def get_image_datetime(file_path):
    try:
        image = Image.open(file_path)
        exif_data = image._getexif()
        if not exif_data:
            return None
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'DateTimeOriginal':
                return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
    except Exception:
        pass
    return None


def get_video_datetime(file_path):
    try:
        parser = createParser(str(file_path))
        if not parser:
            return None
        metadata = extractMetadata(parser)
        if metadata and metadata.has("creation_date"):
            return metadata.get("creation_date")
    except Exception:
        pass
    return None


def get_file_datetime(file_path):
    # Try image EXIF
    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        dt = get_image_datetime(file_path)
        if dt:
            return dt

    # Try video metadata
    if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        dt = get_video_datetime(file_path)
        if dt:
            return dt

    # Fallback: file modified time
    return datetime.fromtimestamp(file_path.stat().st_mtime)


def get_image_gps(file_path):
    try:
        img = Image.open(file_path)
        exif_data = piexif.load(img.info.get("exif", b""))
        gps_data = exif_data.get("GPS", {})

        if gps_data:
            def to_degrees(value):
                d = value[0][0] / value[0][1]
                m = value[1][0] / value[1][1]
                s = value[2][0] / value[2][1]
                return d + (m / 60.0) + (s / 3600.0)

            lat = to_degrees(gps_data[piexif.GPSIFD.GPSLatitude])
            if gps_data[piexif.GPSIFD.GPSLatitudeRef] == b'S':
                lat = -lat

            lon = to_degrees(gps_data[piexif.GPSIFD.GPSLongitude])
            if gps_data[piexif.GPSIFD.GPSLongitudeRef] == b'W':
                lon = -lon

            return (lat, lon)
    except Exception as e:
        # st.error(f"Error reading GPS data from {file_path}. Please check the file format and EXIF data.")
        # st.error(e)
        return None


def get_video_gps(file_path):
    try:
        parser = createParser(str(file_path))
        if not parser:
            return None

        with parser:
            metadata = extractMetadata(parser)
            if not metadata:
                return None

            # hachoir stores GPS sometimes under 'location' or 'latitude/longitude'
            location = metadata.get('location')
            if location:
                # Might return something like "+52.379189+004.899431/"
                parts = location.strip("/").split("+")
                parts = [p for p in parts if p]
                if len(parts) >= 2:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    return (lat, lon)

            # Try direct latitude/longitude keys
            lat = metadata.get('latitude')
            lon = metadata.get('longitude')
            if lat and lon:
                return (float(lat), float(lon))
    except Exception:
        return None


def get_file_gps(file_path):
    suffix = file_path.suffix.lower()

    # Image formats
    if suffix in ['.jpg', '.jpeg', '.png']:
        return get_image_gps(file_path)

    # Video formats
    if suffix in ['.mp4', '.avi', '.mov', '.mkv']:
        return get_video_gps(file_path)

    return None


def check_folder_metadata():
    with st.spinner("Checking data..."):
        settings, _ = load_settings()
        # selected_folder = Path(settings["selected_folder"])
        selected_folder = Path(
            settings["vars"]["analyse_advanced"]["selected_folder"])

        datetimes = []
        gps_coords = []

        # get video and image counts
        image_files = []
        video_files = []

        for f in selected_folder.rglob("*"):
            if f.is_file():
                ext = f.suffix.lower()
                if ext in IMG_EXTENSIONS:
                    image_files.append(f)
                elif ext in VIDEO_EXTENSIONS:
                    video_files.append(f)

        # Limit GPS extraction to reduce processing time (GPS parsing is expensive)
        # Sample GPS every Nth file to spread out checks and catch delayed fixes
        # Stop checking once we have enough valid GPS points to avoid redundant work
        # Most cameras lack GPS, so early exit saves time when no coordinates found
        # Initial GPS readings can be noisy; averaging multiple points improves accuracy
        gps_checked = 0
        max_gps_checks = 100
        check_every_nth = 10
        sufficient_gps_coords = 5

        # images
        for i, file in enumerate(image_files):
            dt = get_image_datetime(file)
            if dt:
                datetimes.append(dt)

            # spread GPS checks across files and early exit
            if i % check_every_nth == 0 and gps_checked < max_gps_checks and len(gps_coords) < sufficient_gps_coords:
                gps = get_image_gps(file)
                if gps:
                    gps_coords.append(gps)
                gps_checked += 1

        # videos
        for i, file in enumerate(video_files):
            dt = get_video_datetime(file)
            if dt:
                datetimes.append(dt)

            # spread GPS checks across files and early exit
            if i % check_every_nth == 0 and gps_checked < max_gps_checks and len(gps_coords) < sufficient_gps_coords:
                gps = get_video_gps(file)
                if gps:
                    gps_coords.append(gps)
                gps_checked += 1

        # min_datetime = min(datetimes) if datetimes else None
        # max_datetime = max(datetimes) if datetimes else None
        exif_min_datetime = min(datetimes) if datetimes else None
        exif_max_datetime = max(datetimes) if datetimes else None

        # Initialize session state for lat/lon if not set
        # if "coords_found_in_exif" not in st.session_state:
        #     st.session_state.coords_found_in_exif = False
        # if "exif_lat" not in st.session_state:
        #     st.session_state.exif_lat = None
        # if "exif_lng" not in st.session_state:
        #     st.session_state.exif_lng = None
        # if "min_datetime_found" not in st.session_state:
        #     st.session_state.min_datetime_found = None
        # if "max_datetime_found" not in st.session_state:
        #     st.session_state.max_datetime_found = None

        # Initialize variables
        coords_found_in_exif = False
        exif_lat = None
        exif_lng = None
        # exif_min_datetime = None
        # exif_max_datetime = None

        if gps_coords:
            lats, lons = zip(*gps_coords)
            ave_lat = statistics.mean(lats)
            ave_lon = statistics.mean(lons)
            # st.session_state.exif_lat = ave_lat
            # st.session_state.exif_lng = ave_lon
            # st.session_state.coords_found_in_exif = True
            exif_lat = ave_lat
            exif_lng = ave_lon
            coords_found_in_exif = True

        # set session state values
        # st.session_state.min_datetime_found = min_datetime
        # st.session_state.max_datetime_found = max_datetime
        # min_datetime_found = min_datetime
        # max_datetime_found = max_datetime

        # SESSION
        update_vars(section="analyse_advanced",
                    updates={
                        "coords_found_in_exif": coords_found_in_exif,
                        "exif_lat": exif_lat,
                        "exif_lng": exif_lng,
                        "exif_min_datetime": exif_min_datetime,
                        "exif_max_datetime": exif_max_datetime,
                    })

        # write results to the app
        info_txt = f"Found {len(image_files)} images and {len(video_files)} videos in the selected folder."
        info_box(info_txt, icon=":material/info:")

        # st.write(st.session_state)


def info_box(msg, icon=":material/info:"):
    flexible_callout(msg,
                     icon=icon,
                     background_color="#d9e3e7af",
                     font_color="#086164",
                     icon_size=23)


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


# def save_project_vars(new_data):
#     if not isinstance(new_data, dict):
#         raise ValueError("Expected a dictionary as input")

#     # fpath = os.path.join("/Users/peter/Desktop/streamlit_app/frontend", "vars.json")
#     fpath = os.path.join(AddaxAI_files, "AddaxAI",
#                          "streamlit-AddaxAI", "frontend", "vars.json")
#     temp_fpath = fpath + ".tmp"  # Temporary file for atomic write

#     # Load existing data or start with an empty dictionary
#     try:
#         if os.path.exists(fpath):
#             with open(fpath, "r", encoding="utf-8") as file:
#                 data = json.load(file)
#         else:
#             data = {}
#     except (json.JSONDecodeError, IOError):
#         data = {}  # Handle corrupt file or read errors

#     # Update data with new values
#     data.update(new_data)

#     # Write updated data to a temporary file first (atomic write)
#     try:
#         with open(temp_fpath, "w", encoding="utf-8") as file:
#             json.dump(data, file, indent=4)
#         # Replace old file only after successful write
#         os.replace(temp_fpath, fpath)
#     except IOError as e:
#         st.warning(f"Error writing to file: {e}")

def save_global_vars(new_data):
    global_settings, settings_file = load_settings()
    temp_file = settings_file + ".tmp"

    # Update with new data
    global_settings.update(new_data)

    # Atomic write to prevent file corruption
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(global_settings, f, indent=2)
        os.replace(temp_file, settings_file)
    except IOError as e:
        raise RuntimeError(f"Error writing to settings file: {e}")


def save_project_vars(new_data):
    # """
    # Update or add key-value pairs to a specific project in a settings JSON file.

    # Parameters:
    # - new_data (dict): Dictionary of new or updated variables.
    # - project_name (str): The name of the project to update (e.g., "VeluweProject").
    # - settings_file (str): Full path to the settings JSON file.
    # """
    # if not isinstance(new_data, dict):
    #     raise ValueError("Expected new_data to be a dictionary")

    # Load full settings or initialize
    # try:
    #     if os.path.exists(settings_file):
    #         with open(settings_file, "r", encoding="utf-8") as f:
    #             settings = json.load(f)
    #     else:
    #         settings = {}
    # except (json.JSONDecodeError, IOError):
    #     settings = {}

    settings, settings_file = load_settings()
    # current_project = settings["global_vars"]["current_project"]
    temp_file = settings_file + ".tmp"

    # Get project section, or initialize if missing
    current_project = settings["global_vars"]["current_project"]
    project_vars = settings["projects"][current_project]
    # st.write(f"project_vars: {project_vars}")

    # Update with new data
    project_vars.update(new_data)
    # st.write(f"project_vars: {project_vars}")

    # Save back into settings
    settings["projects"][current_project] = project_vars

    # Atomic write to prevent file corruption
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        os.replace(temp_file, settings_file)
    except IOError as e:
        raise RuntimeError(f"Error writing to settings file: {e}")


def load_global_vars():
    """Reads the global variables from the JSON file and returns them as a dictionary."""

    # Load full settings or initialize
    try:
        if os.path.exists(settings_file):
            with open(settings_file, "r", encoding="utf-8") as f:
                settings = json.load(f)
        else:
            settings = {}
    except (json.JSONDecodeError, IOError):
        settings = {}

    return settings.get("global_vars", {})


def fetch_vars(section, requested_vars):
    settings, _ = load_settings()
    section_vars = settings["vars"][section]
    return {var: section_vars.get(var, None) for var in requested_vars}.values()


def load_settings():
    """Reads the data from the JSON file and returns it as a dictionary."""

    # Load full settings or initialize
    try:
        if os.path.exists(settings_file):
            with open(settings_file, "r", encoding="utf-8") as f:
                settings = json.load(f)
        else:
            settings = {}
    except (json.JSONDecodeError, IOError):
        settings = {}

    return settings, settings_file

# def load_project_vars():
#     settings, _ = load_settings()
#     project = settings["vars"]["analyse_advanced"].get("project")
#     current_project = settings["projects"][selected_projectID]
#     return current_project


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


# import streamlit as st


# SPECIES SELECTOR


def species_selector():

    nodes = [
        {
            "label": "Class Mammalia", "value": "class_mammalia", "children": [
                {
                    "label": "Order Rodentia", "value": "order_rodentia", "children": [
                        {
                            "label": "Family Sciuridae (squirrels)", "value": "family_sciuridae", "children": [
                                {
                                    "label": "Genus Tamias (chipmunks)", "value": "genus_tamias", "children": [
                                        {"label": "Tamias striatus (eastern chipmunk)",
                                         "value": "tamias_striatus"},
                                        # This one is a species in genus Tamiasciurus, might move later
                                        {"label": "Tamiasciurus hudsonicus (red squirrel)", "value": "tamiasciurus_hudsonicus"}
                                    ]
                                },
                                {
                                    "label": "Genus Sciurus (tree squirrels)", "value": "genus_sciurus", "children": [
                                        {"label": "Sciurus niger (eastern fox squirrel)",
                                         "value": "sciurus_niger"},
                                        {"label": "Sciurus carolinensis (eastern gray squirrel)",
                                         "value": "sciurus_carolinensis"},
                                    ]
                                },
                                {
                                    "label": "Genus Marmota (marmots)", "value": "genus_marmota", "children": [
                                        {"label": "Marmota monax (groundhog)",
                                         "value": "marmota_monax"},
                                        {"label": "Marmota flaviventris (yellow-bellied marmot)",
                                         "value": "marmota_flaviventris"},
                                    ]
                                },
                                {"label": "Otospermophilus beecheyi (california ground squirrel)",
                                 "value": "otospermophilus_beecheyi"}
                            ]
                        },
                        {
                            "label": "Family Muridae (gerbils and relatives)", "value": "family_muridae", "children": [
                                # add species/genus here if any
                            ]
                        },
                        {
                            "label": "Family Geomyidae (pocket gophers)", "value": "family_geomyidae", "children": [
                                # add species/genus here if any
                            ]
                        },
                        {
                            "label": "Family Erethizontidae (new world porcupines)", "value": "family_erethizontidae", "children": [
                                {"label": "Erethizon dorsatus (north american porcupine)",
                                 "value": "erethizon_dorsatus"}
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "label": "Class Squamata", "value": "class_squamata", "children": [
                {
                    "label": "Order Squamata (squamates)", "value": "order_squamata"
                    # could add families/genera/species here if you have them
                }
            ]
        }
    ]

    # Initialize state
    if "selected_nodes" not in st.session_state:
        st.session_state.selected_nodes = []
    if "expanded_nodes" not in st.session_state:
        st.session_state.expanded_nodes = []
    if "last_selected" not in st.session_state:
        st.session_state.last_selected = {}

    # UI
    with st.popover("Select from tree", use_container_width=True):
        selected = tree_select(
            nodes,
            check_model="leaf",
            checked=st.session_state.selected_nodes,
            expanded=st.session_state.expanded_nodes,
            show_expand_all=True,
            half_check_color="#086164",
            check_color="#086164",
            key="tree_select2"
        )

    # If the selection is new, update and rerun
    if selected is not None:
        new_checked = selected.get("checked", [])
        new_expanded = selected.get("expanded", [])
        last_checked = st.session_state.last_selected.get("checked", [])
        last_expanded = st.session_state.last_selected.get("expanded", [])

        if new_checked != last_checked or new_expanded != last_expanded:
            st.session_state.selected_nodes = new_checked
            st.session_state.expanded_nodes = new_expanded
            st.session_state.last_selected = selected
            st.rerun()  # 🔁 Force a rerun so the component picks up the change

    # Feedback

    def count_leaf_nodes(nodes):
        count = 0
        for node in nodes:
            if "children" in node and node["children"]:
                count += count_leaf_nodes(node["children"])
            else:
                count += 1
        return count

    # Example usage
    leaf_count = count_leaf_nodes(nodes)
    # st.write(f"Number of leaf nodes: {leaf_count}")

    st.write("You selected:", len(st.session_state.selected_nodes),
             " of ", leaf_count, "classes")
