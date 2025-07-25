from utils import init_paths

from streamlit_tree_select import tree_select
import os
import json
import streamlit as st
import sys
import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from appdirs import user_config_dir
# import pandas as pd
import statistics
# from collections import defaultdict
import subprocess
import re
import csv
import string
import math
import time as sleep_time
from datetime import datetime, time  # , timedelta
# from datetime import datetime
import os
from pathlib import Path
from datetime import datetime
import tarfile
import requests

from PIL import Image
# from st_flexible_callout_elements import flexible_callout
import random
from PIL.ExifTags import TAGS
from hachoir.metadata import extractMetadata
from hachoir.parser import createParser
import piexif
from tqdm import tqdm
from streamlit_modal import Modal

# st.write("sys.path:", sys.path)
# st.write("length of sys.path:", len(sys.path))
# st.write("This is the module its looking for: /Applications/AddaxAI_files/cameratraps/megadetector/detection/video_utils.py")


# sys.path.insert(0, '/Applications/AddaxAI_files/cameratraps')


# local imports
# from megadetector.detection.video_utils import VIDEO_EXTENSIONS
# VIDEO_EXTENSIONS = []
# # from megadetector.utils.path_utils import IMG_EXTENSIONS
# IMG_EXTENSIONS = []
from utils.common import load_vars, update_vars, replace_vars, info_box, load_map, print_widget_label, clear_vars, requires_addaxai_update, MultiProgressBars


# set global variables
# AddaxAI_files = os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from utils.config import *
# AddaxAI_files = os.path.join(
#     AddaxAI_files, "AddaxAI", "streamlit-AddaxAI")
CLS_DIR = os.path.join(ADDAXAI_FILES, "models", "cls")
DET_DIR = os.path.join(ADDAXAI_FILES, "models", "det")

# load camera IDs
config_dir = user_config_dir("AddaxAI")
map_file = os.path.join(config_dir, "map.json")

# set versions
with open(os.path.join(ADDAXAI_FILES, 'AddaxAI', 'version.txt'), 'r') as file:
    current_AA_version = file.read().strip()




def install_env(
    modal: Modal,
    env_name: str,
):
    
    # modal.close()
    
    info_box(
        "The queue is currently being processed. Do not refresh the page or close the app, as this will interrupt the processing."
    )
    
    if st.button("Cancel", use_container_width=True):
        st.warning("Installation cancelled. You can try again later.")
        sleep_time.sleep(2)
        modal.close()
        return


    command = (
        f"{MICROMAMBA} create -p {ADDAXAI_FILES_ST}/envs/env-{env_name} python=3.11 -y && "
        f"{MICROMAMBA} run -p {ADDAXAI_FILES_ST}/envs/env-{env_name} pip install -r {ADDAXAI_FILES_ST}/envs/reqs/env-{env_name}/macos/requirements.txt"
    )
    
    
    status_placeholder = st.empty()
    
    # st.write("Running MegaDetector...")
    with st.spinner(f"Installing virtual environment '{env_name}'..."):
        with st.expander("Show details", expanded=False):
            with st.container(border=True, height=250):
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    shell=True,
                    universal_newlines=True
                )

                output_placeholder = st.empty()
                output_lines = "Booting up micromamba installation...\n\n"
                output_placeholder.code(output_lines, language="bash")
                for line in process.stdout:
                    output_lines += line
                    output_placeholder.code(output_lines, language="bash")

                process.stdout.close()
                process.wait()

    # ✅ Show result message above
    if process.returncode == 0:
        status_placeholder.success("Environment installed successfully!")
        sleep_time.sleep(2)
        modal.close()
    else:
        status_placeholder.error(f"Installation failed with exit code {process.returncode}.")
        if st.button("Close window", use_container_width=True):
            modal.close()
        
        





# this one is old (when I still thought I needed to download the env)
# but i left it here since it clearly shows how to use the progress bars
def DEMO_PBARS(
    modal: Modal,
    env_name: str,
):
    
    # modal = Modal(f"Installing ENV", key="installing-env", show_close_button=False)
    # modal.open()
    # if modal.is_open():
    #     with modal.container():
            
        info_box(
            "The queue is currently being processed. Do not refresh the page or close the app, as this will interrupt the processing."
            "It is recommended to avoid using your computer for other tasks, as the processing requires significant system resources."
        )
        
        if st.button("Cancel", use_container_width=True):
            st.warning("Installation cancelled. You can try again later.")
            sleep_time.sleep(2)
            modal.close()
            return





        # url = f"https://addaxaipremiumstorage.blob.core.windows.net/github-zips/latest/macos/envs/{env_name}.tar.xz"
        url = f"https://addaxaipremiumstorage.blob.core.windows.net/github-zips/latest/macos/envs/env-{env_name}.tar.xz"
        local_filename = f"envs/env-{env_name}.tar.xz"

        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        # show progress bars
        pbars = MultiProgressBars("Installing virual environment")
        pbars.add_pbar("download", "Waiting to download...", "Downloading...", "Download complete!", max_value=total_size)
        pbars.add_pbar("extract", "Waiting to extract...", "Extracting...", "Extraction complete!", max_value=None)
        # pbars.add_status("install", "Waiting to install...", "Installing...", "Installation complete!")

        # download progress bar
        block_size = 1024
        pbar = tqdm(total=total_size / (1024 * 1024), unit='MB', unit_scale=False, unit_divisor=1)
        with open(local_filename, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                mb = len(data) / (1024 * 1024)
                pbar.update(mb)
                label = pbars.generate_label_from_tqdm(pbar)
                pbars.update("download", n=len(data), text=label)
        pbar.close()
        
        # Extract progress bar
        with tarfile.open(local_filename, mode="r:xz") as tar:
            members = tar.getmembers()
            pbars.set_max_value("extract", len(members))  # ✅ This is clean and intuitive
            pbar = tqdm(total=len(members), unit="files", unit_scale=False)
            for member in members:
                tar.extract(member, path="envs/")
                pbar.update(1)
                label = pbars.generate_label_from_tqdm(pbar)
                pbars.update("extract", n=1, text=label)
            pbar.close()
        os.remove(local_filename)
        
        # pip install requirements
        
        
        
        # # install_placeholder.write("")

        # pip_cmd =                 [
        #     os.path.join(AddaxAI_streamlit_files, "envs", f"env-{env_name}", "bin", "python"),
        #             "-m", "pip", "install", "-r",
        #             os.path.join(AddaxAI_streamlit_files, "envs", "reqs", f"env-{env_name}", "macos", "requirements.txt")
        #         ]
        
        
        # # Trigger pip install

        # status = pbars.update_status("install", phase="mid")

        # with status:
        #     with st.container(border=True, height=300):
        #         output_placeholder = st.empty()
        #         live_output = "Booting up pip installation...\n\n"
        #         output_placeholder.code(live_output)

        #         process = subprocess.Popen(
        #             pip_cmd,
        #             stdout=subprocess.PIPE,
        #             stderr=subprocess.STDOUT,
        #             text=True,
        #             bufsize=1
        #         )

        #         for line in process.stdout:
        #             live_output += line
        #             output_placeholder.code(live_output)

        #         process.wait()

        # pbars.update_status("install", phase="post")
        
        modal.close()
    
    
    # modal.close()







def project_selector_widget():

    # check what is already known and selected
    projects, selected_projectID = load_known_projects()

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
        # map, _ = load_map()
        analyse_advanced_vars = load_vars(section="analyse_advanced")
        previous_projectID = analyse_advanced_vars.get(
            "selected_projectID", None)
        if previous_projectID != selected_projectID:
            # analyse_advanced_vars["selected_projectID"] = selected_projectID
            update_vars("analyse_advanced", {
                "selected_projectID": selected_projectID
            })
            # with open(map_file, "w") as file:
            #     json.dump(map, file, indent=2)
            st.rerun()

        # return
        return selected_projectID


def location_selector_widget():

    # load settings
    # coords_found_in_exif, exif_lat, exif_lng = load_vars(section="analyse_advanced",  # SESSION
    #                                                       requested_vars=["coords_found_in_exif", "exif_lat", "exif_lng"])

    vars = load_vars(section="analyse_advanced")
    coords_found_in_exif = vars.get("coords_found_in_exif", False)
    exif_lat = vars.get("exif_lat", 0.0)
    exif_lng = vars.get("exif_lng", 0.0)

    # check what is already known and selected
    locations, location = load_known_locations()

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
                if dist > 0:
                    message += f"Matches known location <i>{name}</i>, about {dist} meters away."
                else:
                    message += f"Matches known location <i>{name}</i>."
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

    # init vars
    vars = load_vars(section="analyse_advanced")
    exif_min_datetime_str = vars.get("exif_min_datetime", None)
    exif_min_datetime = (
        datetime.fromisoformat(exif_min_datetime_str)
        if exif_min_datetime_str is not None
        else None
    )

    # Initialize the session state for exif_min_datetime if not set
    # if "exif_min_datetime" not in st.session_state:
    #     st.session_state.exif_min_datetime = None
    # if present, it will be of format "datetime.datetime(2013, 1, 17, 13, 5, 21)"

    # Pre-fill defaults
    default_date = None
    default_hour = "--"
    default_minute = "--"
    default_second = "--"

    if exif_min_datetime:
        # # In case it's stored as a string like "datetime.datetime(2013, 1, 17, 13, 5, 21)"
        # if isinstance(st.session_state.exif_min_datetime, str):
        #     st.session_state.exif_min_datetime = eval(st.session_state.exif_min_datetime)

        dt = exif_min_datetime
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

    if exif_min_datetime:
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


def load_known_projects():
    map, _ = load_map()
    analyse_advanced_vars = load_vars(section="analyse_advanced")
    projects = map["projects"]
    selected_projectID = analyse_advanced_vars.get(
        "selected_projectID")
    return projects, selected_projectID


def load_known_locations():
    map, _ = load_map()
    analyse_advanced_vars = load_vars(section="analyse_advanced")
    selected_projectID = analyse_advanced_vars.get(
        "selected_projectID")
    project = map["projects"][selected_projectID]
    selected_locationID = analyse_advanced_vars.get(
        "selected_locationID")
    locations = project["locations"]
    return locations, selected_locationID


def load_known_deployments():
    settings, _ = load_map()
    analyse_advanced_vars = load_vars(section="analyse_advanced")
    selected_projectID = analyse_advanced_vars.get(
        "selected_projectID")
    project = settings["projects"][selected_projectID]
    selected_locationID = analyse_advanced_vars.get(
        "selected_locationID")
    location = project["locations"][selected_locationID]
    deployments = location["deployments"]
    selected_deploymentID = analyse_advanced_vars.get(
        "selected_deploymentID")
    return deployments, selected_deploymentID


def generate_deployment_id():

    # Create a consistent 5-char hash from the datetime and some randomness
    rand_str_1 = ''.join(random.choices(
        string.ascii_uppercase + string.digits, k=5))
    rand_str_2 = ''.join(random.choices(
        string.ascii_uppercase + string.digits, k=5))

    # Combine into deployment ID
    return f"dep-{rand_str_1}-{rand_str_2}"


def add_deployment(selected_min_datetime):

    # settings, _ = load_settings()
    # selected_folder = settings["vars"]["analyse_advanced"].get(
    #     "selected_folder")
    # selected_projectID = settings["vars"]["analyse_advanced"].get(
    #     "selected_projectID")
    # project = settings["projects"][selected_projectID]
    # location = project["location"]
    # location = project["locations"][location]
    # deployments = location["deployments"]

    map, _ = load_map()
    analyse_advanced_vars = load_vars(section="analyse_advanced")
    selected_folder = analyse_advanced_vars.get(
        "selected_folder")
    selected_projectID = analyse_advanced_vars.get(
        "selected_projectID")
    project = map["projects"][selected_projectID]
    selected_locationID = analyse_advanced_vars.get(
        "selected_locationID")
    location = project["locations"][selected_locationID]
    deployments = location["deployments"]

    # check what the exif datetime is
    exif_min_datetime_str = analyse_advanced_vars.get(
        "exif_min_datetime", None)
    exif_min_datetime = (
        datetime.fromisoformat(exif_min_datetime_str)
        if exif_min_datetime_str is not None
        else None
    )
    exif_max_datetime_str = analyse_advanced_vars.get(
        "exif_max_datetime", None)
    exif_max_datetime = (
        datetime.fromisoformat(exif_max_datetime_str)
        if exif_max_datetime_str is not None
        else None
    )

    # then calculate the difference between the selected datetime and the exif datetime
    diff_min_datetime = selected_min_datetime - exif_min_datetime
    # TODO: if the exif_min_datetime is None, it errors. fix that.

    # Adjust exif_max_datetime if selected_min_datetime is later than exif_min_datetime
    selected_max_datetime = exif_max_datetime + diff_min_datetime

    # generate a unique deployment ID
    deployment_id = generate_deployment_id()

    update_vars("analyse_advanced", {
        "selected_deploymentID": deployment_id,
    })

    # Add new deployment
    deployments[deployment_id] = {
        "deploymentStart": datetime.isoformat(selected_min_datetime),
        # this is not ctually selected, but calculated from the exif metadata
        "deploymentEnd": datetime.isoformat(selected_max_datetime),
        "path": selected_folder,
        "datetimeDiffSeconds": diff_min_datetime.total_seconds()
    }

    # Save updated settings
    with open(map_file, "w") as file:
        json.dump(map, file, indent=2)

    # Return list of deployments and index of the new one
    deployment_list = list(deployments.values())
    selected_index = deployment_list.index(deployments[deployment_id])

    return selected_index, deployment_list


def add_location(location_id, lat, lon):

    settings, _ = load_map()
    analyse_advanced_vars = load_vars(section="analyse_advanced")
    selected_projectID = analyse_advanced_vars.get(
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
        # "selected_deployment": None,
        "deployments": {},
    }

    # add the selected location ID to the vars
    update_vars("analyse_advanced", {
        "selected_locationID": location_id,
    })
    # analyse_advanced_vars = load_vars(section="analyse_advanced")
    # analyse_advanced_vars["selected_locationID"] = location_id

    # # Sort locations (optional: dicts don't preserve order unless using OrderedDict or Python 3.7+)
    # sorted_locations = dict(
    #     sorted(locations.items(), key=lambda item: item[0].lower()))
    # settings["projects"][selected_projectID]["locations"] = sorted_locations
    # settings["projects"][selected_projectID]["location"] = location_id

    # Save updated settings
    with open(map_file, "w") as file:
        json.dump(settings, file, indent=2)

    # Return list of locations and index of the new one
    location_list = list(locations.values())
    selected_index = location_list.index(locations[location_id])

    return selected_index, location_list


# # show popup with model information
def add_project(projectID, comments):

    map, map_file = load_map()
    projects = map["projects"]
    projectIDs = projects.keys()

    # analyse_advanced_vars = load_vars(section="analyse_advanced")
    # selected_projectID = analyse_advanced_vars["selected_projectID"]

    # st.write(projectIDs)

    # Check if project_id is unique
    if projectID in projectIDs:
        raise ValueError(
            f"project ID '{projectID}' already exists. Please choose a unique ID, or select existing project from dropdown menu.")

    # Add new project
    projects[projectID] = {
        "comments": comments,
        "locations": {},
    }

    map["projects"] = projects  # add project
    # update selected project
    update_vars("analyse_advanced", {
        "selected_projectID": projectID,
        "selected_locationID": None,  # reset location selection
        "selected_deploymentID": None,  # reset deployment selection
    })
    # map["vars"]["analyse_advanced"]["selected_projectID"] = projectID

    # Save updated settings
    with open(map_file, "w") as file:
        json.dump(map, file, indent=2)

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

            # load known projects IDs
            known_projects, _ = load_known_projects()

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
                    # st.session_state.clear()
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
            vars = load_vars(section="analyse_advanced")
            selected_lat = vars.get("selected_lat", None)
            selected_lng = vars.get("selected_lng", None)
            exif_set = vars.get("exif_set", False)
            coords_found_in_exif = vars.get("coords_found_in_exif", False)
            exif_lat = vars.get("exif_lat", None)
            exif_lng = vars.get("exif_lng", None)

            # # init session state vars
            # if "selected_lat" not in st.session_state:
            #     st.session_state.selected_lat = None
            # if "selected_lng" not in st.session_state:
            #     st.session_state.selected_lng = None
            # if "exif_set" not in st.session_state:
            #     st.session_state.exif_set = False
            # if "coords_found_in_exif" not in st.session_state:
            #     st.session_state.coords_found_in_exif = False

            # update values if coordinates found in metadata
            if coords_found_in_exif:
                info_box(
                    f"Coordinates from metadata have been preselected ({exif_lat:.6f}, {exif_lng:.6f}).")
                if not exif_set:
                    # selected_lat = exif_lat
                    # selected_lng = exif_lng
                    # exif_set = True
                    update_vars("analyse_advanced", {
                        "selected_lat": exif_lat,
                        "selected_lng": exif_lng,
                        "exif_set": True,
                    })
                    st.rerun()

            # load known locations
            known_locations, _ = load_known_locations()

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
                if selected_lat and selected_lng:
                    fl.Marker(
                        [selected_lat,
                            selected_lng],
                        title="Selected location",
                        tooltip="Selected location",
                        icon=fl.Icon(icon="camera", prefix="fa",
                                     color="darkred")
                    ).add_to(m)
                    bounds.append([selected_lat,
                                   selected_lng])

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
                if selected_lat and selected_lng:
                    fl.Marker(
                        [selected_lat,
                            selected_lng],
                        title="Selected location",
                        tooltip="Selected location",
                        icon=fl.Icon(icon="camera", prefix="fa",
                                     color="darkred")
                    ).add_to(m)

                    # only one marker so set bounds to the selected location
                    buffer = 0.001
                    bounds = [
                        [selected_lat - buffer,
                            selected_lng - buffer],
                        [selected_lat + buffer,
                            selected_lng + buffer]
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
                map_data = st_folium(m, height=325, width=700)
            # map_data = st_folium(m, height=300, width=700)

            # update lat lng widgets when clicking on map
            if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
                selected_lat = map_data["last_clicked"]["lat"]
                selected_lng = map_data["last_clicked"]["lng"]
                # fl.Marker(
                #     [selected_lat, selected_lng],
                #     title="Selected location",
                #     tooltip="Selected location",
                #     icon=fl.Icon(icon="camera", prefix="fa", color="green")
                # ).add_to(m)
                update_vars("analyse_advanced", {
                    "selected_lat": selected_lat,
                    "selected_lng": selected_lng,
                })
                st.rerun()

            # user input
            col1, col2 = st.columns([1, 1])

            # lat
            with col1:
                print_widget_label("Enter latitude or click on the map",
                                   help_text="Enter the latitude of the location.")
                old_lat = selected_lat if not None else 0.0
                new_lat = st.number_input(
                    "Enter latitude or click on the map",
                    value=selected_lat,
                    format="%.6f",
                    step=0.000001,
                    min_value=-90.0,
                    max_value=90.0,
                    label_visibility="collapsed",
                )
                selected_lat = new_lat
                if new_lat != old_lat:
                    st.rerun()

            # lng
            with col2:
                print_widget_label("Enter longitude or click on the map",
                                   help_text="Enter the longitude of the location.")
                old_lng = selected_lng if not None else 0.0
                new_lng = st.number_input(
                    "Enter longitude or click on the map",
                    value=selected_lng,
                    format="%.6f",
                    step=0.000001,
                    min_value=-180.0,
                    max_value=180.0,
                    label_visibility="collapsed",
                )
                selected_lng = new_lng
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
                elif selected_lat == 0.0 and selected_lng == 0.0:
                    st.error(
                        "Error: Latitude and Longitude cannot be (0, 0). Please select a valid location.")
                elif selected_lat is None or selected_lng is None:
                    st.error(
                        "Error: Latitude and Longitude cannot be empty. Please select a valid location.")
                else:

                    # if all good, add location
                    add_location(
                        new_location_id, selected_lat, selected_lng)
                    new_location_id = None

                    # reset session state variables before reloading
                    update_vars("analyse_advanced", {
                        "coords_found_in_exif": False,
                        "exif_set": False,
                        "exif_lat": None,
                        "exif_lng": None,
                        "selected_lat": None,
                        "selected_lng": None
                    })
                    popover_container.empty()
                    st.rerun()


def browse_directory_widget():

    analyse_advanced_vars = load_vars(section="analyse_advanced")
    selected_folder = analyse_advanced_vars.get("selected_folder")

    col1, col2 = st.columns([1, 3])#, vertical_alignment="center")
    with col1:
        if st.button(":material/folder: Browse", key="folder_select_button", use_container_width=True):
            selected_folder = select_folder()
            # update_vars("selected_folder", selected_folder)
            # save_global_vars({"selected_folder": selected_folder})
            clear_vars(section="analyse_advanced")
            update_vars(section="analyse_advanced",
                        updates={"selected_folder": selected_folder})

            # # reset session state variables
            # st.session_state.clear()

    if not selected_folder:
        with col2:
            # st.write('<span style="color: grey;"> None selected...</span>',
            #          unsafe_allow_html=True)
            text = f'<span style="color: grey;"> None selected...</span>'
            st.markdown(
                f"""
                    <div style="background-color: #f0f2f6; padding: 7px; border-radius: 8px;">
                        &nbsp;&nbsp;{text}
                    </div>
                    """,
                unsafe_allow_html=True
            )
    else:
        with col2:
            folder_short = "..." + \
                selected_folder[-45:] if len(
                    selected_folder) > 45 else selected_folder
            # st.markdown(
            #     f'Selected folder <code style="color:#086164; font-family:monospace;">{folder_short}</code>', unsafe_allow_html=True)
            
            text = f"Selected &nbsp;&nbsp;<code style='color:#086164; font-family:monospace;'>{folder_short}</code>"
            st.markdown(
                f"""
                    <div style="background-color: #f0f2f6; padding: 7px; border-radius: 8px;">
                        &nbsp;&nbsp;{text}
                    </div>
                    """,
                unsafe_allow_html=True
            )
    return selected_folder


def select_folder():
    result = subprocess.run([sys.executable, os.path.join(
        ADDAXAI_FILES_ST, "utils", "folder_selector.py")], capture_output=True, text=True)
    folder_path = result.stdout.strip()
    if folder_path != "" and result.returncode == 0:
        return folder_path
    else:
        return None


#######################
### MODEL UTILITIES ###
#######################

def load_model_metadata():
    model_info_json = os.path.join(
        ADDAXAI_FILES_ST, "assets", "model_meta", "model_meta.json")
    with open(model_info_json, "r") as file:
        model_info = json.load(file)
    return model_info


def det_model_selector_widget(model_meta):
    det_model_meta = model_meta["det"]

    # Build model info tuples: (emoji + friendly_name for display, modelID, friendly_name for sorting)
    model_items = [
        (f"{meta.get('emoji', '')}\u00A0\u00A0{meta['friendly_name']}",
         modelID, meta["friendly_name"])
        for modelID, meta in det_model_meta.items()
    ]

    # Sort by the friendly_name (3rd element of the tuple)
    model_choices = sorted(model_items, key=lambda x: x[2].lower())
    display_names = [item[0] for item in model_choices]
    modelID_lookup = {**{item[0]: item[1] for item in model_choices}}

    # Load previously selected model ID
    general_settings_vars = load_vars(section="general_settings")
    previously_selected_det_modelID = general_settings_vars.get(
        "previously_selected_det_modelID", "MD5A")

    # Resolve previously selected modelID to display name
    previously_selected_display_name = next(
        (name for name, ID in modelID_lookup.items()
         if ID == previously_selected_det_modelID),
        None
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_display_name = st.selectbox(
            "Select a model for detection",
            options=display_names,
            index=display_names.index(previously_selected_display_name),
            label_visibility="collapsed"
        )

        selected_modelID = modelID_lookup[selected_display_name]

    with col2:
        show_cls_model_info_popover(det_model_meta[selected_modelID])

    return selected_modelID


def cls_model_selector_widget(model_meta):

    cls_model_meta = model_meta["cls"]

    # Build model info tuples: (emoji + friendly_name for display, modelID, friendly_name for sorting)
    model_items = [
        (f"{meta.get('emoji', '')}\u00A0\u00A0{meta['friendly_name']}",
         modelID, meta["friendly_name"])
        for modelID, meta in cls_model_meta.items()
    ]

    # Sort by the friendly_name (3rd element of the tuple)
    model_choices = sorted(model_items, key=lambda x: x[2].lower())

    # Define the "NONE" entry for generic animal detection
    none_display = "🐾  Generic animal detection (no identification)"
    display_names = [none_display] + [item[0] for item in model_choices]
    modelID_lookup = {none_display: "NONE", **
                      {item[0]: item[1] for item in model_choices}}

    # Load previously selected model ID
    general_settings_vars = load_vars(section="general_settings")
    previously_selected_modelID = general_settings_vars.get(
        "selected_modelID", "SAH-DRY-ADS-v1")

    # Resolve previously selected modelID to display name
    previously_selected_display_name = next(
        (name for name, ID in modelID_lookup.items()
         if ID == previously_selected_modelID),
        "NONE"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_display_name = st.selectbox(
            "Select a model for classification",
            options=display_names,
            index=display_names.index(previously_selected_display_name),
            label_visibility="collapsed"
        )

        selected_modelID = modelID_lookup[selected_display_name]

    with col2:
        if selected_modelID != "NONE":
            show_cls_model_info_popover(cls_model_meta[selected_modelID])
        else:
            show_none_model_info_popover()

    return selected_modelID


def select_model_widget(model_type, prev_selected_model):
    # prepare radio button options
    model_info = load_all_model_info(model_type)
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


def load_all_model_info(type):

    # load
    model_info_json = os.path.join(
        ADDAXAI_FILES_ST, "model_info.json")
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
        # settings, _ = load_map()
        # selected_folder = Path(settings["selected_folder"])
        # selected_folder = Path(
        #     settings["vars"]["analyse_advanced"]["selected_folder"])
        analyse_advanced_vars = load_vars(section="analyse_advanced")
        selected_folder = Path(analyse_advanced_vars.get("selected_folder"))

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


def show_none_model_info_popover():

    popover_container = st.empty()
    with popover_container.container():
        with st.popover(f":material/info: Info",
                        help="Model information",
                        use_container_width=True):
            st.write(
                "This model is used for generic animal detection, without identifying specific species or classes.")
            st.write(
                "It is useful for detecting animals in images or videos without the need for specific classification.")


def show_none_model_info_popover():
    # # use st.empty to create a popover container
    # # so that it can be closed on button click
    # # and the popover can be reused
    # popover_container = st.empty()
    # with popover_container.container():
    with st.popover(
        ":material/info: Info",
        help="Model information",
        use_container_width=True
    ):
        st.markdown(
            """
            **Generic animal detection (no identification)**

            Selecting this option means the system will use a general-purpose detection model that locates and labels objects only as:
            - *animal*
            - *vehicle*
            - *person*

            No species-level identification will be performed.

            This option is helpful if:
            - There is no suitable species identification model available.
            - You want to filter out empty images.
            - You want to ID the animals manually without using a idnetification model.
            
            If you want to use a specific species identification model, please select one from the dropdown menu.
            """
        )

# format the class name for display


def format_class_name(s):
    if "http" in s:
        return s  # leave as is
    else:
        s = s.replace('_', ' ')
        s = s.strip()
        s = s.lower()
        return s


def show_cls_model_info_popover(model_info):
    # use st.empty to create a popover container
    # so that it can be closed on button click
    # and the popover can be reused
    popover_container = st.empty()
    with popover_container.container():
        with st.popover(f":material/info: Info",
                        help="Model information",
                        use_container_width=True):

            friendly_name = model_info.get('friendly_name', None)
            if friendly_name and friendly_name != "":
                st.write("")
                print_widget_label("Name", "rocket_launch")
                st.write(friendly_name)

            description = model_info.get('description', None)
            if description and description != "":
                st.write("")
                print_widget_label("Description", "history_edu")
                st.write(description)

            all_classes = model_info.get('all_classes', None)
            if all_classes and all_classes != []:
                st.write("")
                print_widget_label("Classes", "pets")
                formatted_classes = [format_class_name(
                    cls) for cls in all_classes]
                if len(formatted_classes) == 1:
                    string = formatted_classes[0] + "."
                else:
                    string = ', '.join(
                        formatted_classes[:-1]) + ', and ' + formatted_classes[-1] + "."
                st.write(string.capitalize())

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
                needs_EA_update_bool = requires_addaxai_update(min_version)
                if needs_EA_update_bool:
                    st.write(
                        f"This model requires AddaxAI version {min_version}. Your current AddaxAI version {current_AA_version} will not be able to run this model. An update is required. Update via the [Addax Data Science website](https://addaxdatascience.com/addaxai/).")
                else:
                    st.write(
                        f"Current version of AddaxAI (v{current_AA_version}) is able to use this model. No update required.")

# @st.dialog("Model information", width="large")
# def show_model_info(model_info):
#     friendly_name = model_info.get('friendly_name', None)
#     if friendly_name and friendly_name != "":
#         st.write("")
#         print_widget_label("Name", "rocket_launch")
#         st.write(friendly_name)
#     # print_widget_label("Name", "rocket_launch")
#     # st.write(model_name)

#     description = model_info.get('description', None)
#     if description and description != "":
#         st.write("")
#         print_widget_label("Description", "history_edu")
#         st.write(description)

#     all_classes = model_info.get('all_classes', None)
#     if all_classes and all_classes != []:
#         st.write("")
#         print_widget_label("Classes", "pets")
#         formatted_classes = [all_classes[0].replace('_', ' ').capitalize(
#         )] + [cls.replace('_', ' ').lower() for cls in all_classes[1:]]
#         output = ', '.join(
#             formatted_classes[:-1]) + ', and ' + formatted_classes[-1] + "."
#         st.write(output)

#     developer = model_info.get('developer', None)
#     if developer and developer != "":
#         st.write("")
#         print_widget_label("Developer", "code")
#         st.write(developer)

#     owner = model_info.get('owner', None)
#     if owner and owner != "":
#         st.write("")
#         print_widget_label("Owner", "account_circle")
#         st.write(owner)

#     info_url = model_info.get('info_url', None)
#     if info_url and info_url != "":
#         st.write("")
#         print_widget_label("More information", "info")
#         st.write(info_url)

#     citation = model_info.get('citation', None)
#     if citation and citation != "":
#         st.write("")
#         print_widget_label("Citation", "article")
#         st.write(citation)

#     license = model_info.get('license', None)
#     if license and license != "":
#         st.write("")
#         print_widget_label("License", "copyright")
#         st.write(license)

#     min_version = model_info.get('min_version', None)
#     if min_version and min_version != "":
#         st.write("")
#         print_widget_label("Required AddaxAI version", "verified")
#         needs_EA_update_bool = requires_addaxai_update(min_version)
#         if needs_EA_update_bool:
#             st.write(
#                 f"This model requires AddaxAI version {min_version}. Your current AddaxAI version {current_AA_version} will not be able to run this model. An update is required. Update via the [Addax Data Science website](https://addaxdatascience.com/addaxai/).")
#         else:
#             st.write(
#                 f"Current version of AddaxAI (v{current_AA_version}) is able to use this model. No update required.")


def load_model_info(model_name):
    return json.load(open(os.path.join(CLS_DIR, model_name, "variables.json"), "r"))


def save_cls_classes(cls_model_key, slected_classes):
    # load
    model_info_json = os.path.join(
        ADDAXAI_FILES_ST, "model_info.json")
    with open(model_info_json, "r") as file:
        model_info = json.load(file)
    model_info['cls'][cls_model_key]['selected_classes'] = slected_classes

    # save
    with open(model_info_json, "w") as file:
        json.dump(model_info, file, indent=4)


def load_taxon_mapping(cls_model_ID):
    taxon_mapping_csv = os.path.join(
        ADDAXAI_FILES_ST, "models", "cls", cls_model_ID, "taxon-mapping.csv")

    taxon_mapping = []
    with open(taxon_mapping_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            taxon_mapping.append(row)

    return taxon_mapping


# def slugify(text):
#     """Create a slug-friendly string for the value keys."""
#     text = text.lower()
#     text = re.sub(r'\s+', '_', text)
#     text = re.sub(r'[^a-z0-9_]+', '', text)
#     return text

# def flatten_single_child_nodes(nodes):
#     flattened = []
#     for node in nodes:
#         if "children" in node and len(node["children"]) == 1:
#             child = node["children"][0]
#             grandchildren = child.get("children", [])

#             # If child is leaf (no children), use child's label and value (likely model_class)
#             if not grandchildren:
#                 merged_label = child['label']
#                 merged_value = child['value']
#                 merged_children = []
#             else:
#                 # Child has children, keep parent's label/value
#                 merged_label = node['label']
#                 merged_value = node['value']
#                 merged_children = flatten_single_child_nodes(grandchildren)

#             merged_node = {
#                 "label": merged_label,
#                 "value": merged_value
#             }
#             if merged_children:
#                 merged_node["children"] = merged_children

#             flattened.append(merged_node)

#         else:
#             new_node = node.copy()
#             if "children" in node and node["children"]:
#                 new_node["children"] = flatten_single_child_nodes(node["children"])
#             flattened.append(new_node)
#     return flattened

def sort_leaf_first(nodes):
    leaves = []
    parents = []

    for node in nodes:
        # Determine if it's a leaf or has children
        if "children" in node and node["children"]:
            # Recurse into children first
            node["children"] = sort_leaf_first(node["children"])
            parents.append(node)
        else:
            leaves.append(node)

    # Sort both groups alphabetically by label (case-insensitive)
    leaves.sort(key=lambda x: x["label"].lower())
    parents.sort(key=lambda x: x["label"].lower())

    return leaves + parents

def merge_single_redundant_nodes(nodes):
    merged = []
    for node in nodes:
        if "children" in node and len(node["children"]) == 1:
            child = node["children"][0]

            # Compare prefixes to see if they're redundant
            parent_prefix = node["label"].split(" ")[0]
            child_prefix = child["label"].split(" ")[0]

            if parent_prefix == child_prefix:
                # Replace parent node with child's label and value
                node["label"] = child["label"]
                node["value"] = child["value"]

                # If the child has children, adopt them; else remove children entirely
                grandkids = child.get("children", [])
                if grandkids:
                    node["children"] = merge_single_redundant_nodes(grandkids)
                else:
                    node.pop("children", None)  # make it a real leaf node

        # If still has children (not replaced), recurse into them
        if "children" in node and node["children"]:
            node["children"] = merge_single_redundant_nodes(node["children"])

        merged.append(node)
    return merged

def build_taxon_tree(taxon_mapping):
    root = {}
    levels = ["level_class", "level_order",
              "level_family", "level_genus", "level_species"]

    for entry in taxon_mapping:
        
        # If no proper class level, place under "Unknown taxonomy"
        if not entry.get("level_class", "").startswith("class "):
            unknown_key = "<i>Unknown taxonomy</i>"
            if unknown_key not in root:
                root[unknown_key] = {
                    "label": unknown_key,
                    "value": unknown_key,
                    "children": {}
                }
            current_level = root[unknown_key]["children"]

            model_class = entry["model_class"].strip()
            label = f"<b>{model_class}</b>"
            value = model_class
            if value not in current_level:
                current_level[value] = {
                    "label": label,
                    "value": value,
                    "children": {}
                }
            continue  # Skip the normal taxonomic loop
        
        current_level = root
        last_taxon_name = None
        for i, level in enumerate(levels):
            taxon_name = entry.get(level)
            if not taxon_name or taxon_name == "":
                continue

            is_last_level = (i == len(levels) - 1)

            if not is_last_level:
                if taxon_name == last_taxon_name:
                    continue
                label = taxon_name
                value = taxon_name

                if value not in current_level:
                    current_level[value] = {
                        "label": label,
                        "value": value,
                        "children": {}
                    }
                current_level = current_level[value]["children"]
                last_taxon_name = taxon_name

            else:
                model_class = entry["model_class"].strip()
                if taxon_name.startswith("class ") or \
                    taxon_name.startswith("order ") or \
                        taxon_name.startswith("family ") or \
                            taxon_name.startswith("genus "):
                    label = f"{taxon_name} (<b>{model_class}</b>, <i>unspecified)</i>"
                else:
                    label = f"{taxon_name} (<b>{model_class}</b>)"
                value = model_class
                if value not in current_level:
                    current_level[value] = {
                        "label": label,
                        "value": value,
                        "children": {}
                    }
                # species is a leaf: do not update current_level

    def dict_to_list(d):
        result = []
        for node_key, node_val in d.items():
            children_list = dict_to_list(
                node_val["children"]) if node_val["children"] else []
            node = {
                "label": node_val["label"],
                "value": node_val["value"]
            }
            if children_list:
                node["children"] = children_list
            result.append(node)
        return result

    raw_tree = dict_to_list(root)
    merged_tree = merge_single_redundant_nodes(raw_tree)
    sorted_tree = sort_leaf_first(merged_tree)
    return sorted_tree

def get_all_leaf_values(nodes):
    leaf_values = []
    for node in nodes:
        if "children" in node and node["children"]:
            # Recurse into children
            leaf_values.extend(get_all_leaf_values(node["children"]))
        else:
            # Leaf node
            leaf_values.append(node["value"])
    return leaf_values

selected_species = ["cow", "dog", "cat"]
# the json is here: /Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/models/cls/SAH-DRY-ADS-v1/variables.json
# it looks like this:
# {
#     "model_fname": "sub_saharan_drylands_v1.pt",
#     "description": "The Sub-Saharan Drylands model is a deep learning image classifier trained on 13 million camera trap images from diverse ecosystems across eastern and southern Africa. Covering 328 categories, primarily at the species level, it supports taxonomic fallback, predicting higher-level taxa (e.g., genus or family) when species-level certainty is low. The model is designed for wildlife identification across savannas, dry forests, arid shrublands, and semi-desert habitats. Training data includes images from South Africa, Tanzania, Kenya, Mozambique, Botswana, Namibia, Rwanda, Madagascar, and Uganda. All training images are open-source and available via LILA BC (https://lila.science/).",
#     "developer": "Addax Data Science",
#     "env": "pytorch",
#     "type": "addax-sdzwa-pt",
#     "download_info": [
#     ],
#     "citation": "https://joss.theoj.org/papers/10.21105/joss.05581",
#     "license": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
#     "total_download_size": "215 MB",
#     "info_url": "https://addaxdatascience.com/",
#     "all_classes": [
#     ],
#     "selected_classes": [
#     ],
#     "var_cls_detec_thresh": "0.40",
#     "var_cls_detec_thresh_default": "0.40",
#     "var_cls_class_thresh": "0.50",
#     "var_cls_class_thresh_default": "0.50",
#     "var_smooth_cls_animal": false,
#     "var_tax_levels_idx": 0,
#     "var_tax_fallback": true,
#     "min_version": "6.16"
# }
# the selected_species should go into the "selected_classes" field of the json file

def add_deployment_to_queue():
    
    # todo: this all needs to be st.session_state based, 
    
    analyse_advanced_vars = load_vars(section="analyse_advanced")
    process_queue = analyse_advanced_vars.get("process_queue", [])
    # previous_process_queue = analyse_advanced_vars.get("process_queue", [])
    # process_queue = analyse_advanced_vars.get("process_queue", []).copy() # copy to avoid mutating the original list in session state

    
    selected_folder = analyse_advanced_vars["selected_folder"]
    selected_projectID = analyse_advanced_vars["selected_projectID"]
    selected_locationID = analyse_advanced_vars["selected_locationID"]
    # selected_lat = analyse_advanced_vars["selected_lat"]
    # selected_lng = analyse_advanced_vars["selected_lng"]
    selected_min_datetime = analyse_advanced_vars["selected_min_datetime"]
    # selected_deploymentID = analyse_advanced_vars["selected_deploymentID"]
    selected_det_modelID = analyse_advanced_vars["selected_det_modelID"]
    selected_cls_modelID = analyse_advanced_vars["selected_cls_modelID"]
    # deployment_start_file = analyse_advanced_vars["deployment_start_file"]
    # deployment_start_datetime = analyse_advanced_vars["deployment_start_datetime"]
    
    # Create a new deployment entry
    new_deployment = {
        "selected_folder": selected_folder,
        "selected_projectID": selected_projectID,
        "selected_locationID": selected_locationID,
        # "selected_lat": selected_lat,
        # "selected_lng": selected_lng,
        "selected_min_datetime": selected_min_datetime,
        # "selected_deploymentID": selected_deploymentID,
        "selected_det_modelID": selected_det_modelID,
        "selected_cls_modelID": selected_cls_modelID,
        # "deployment_start_file": deployment_start_file,
        # "deployment_start_datetime": deployment_start_datetime
    }
    
    # Add the new deployment to the queue
    # st.write(f"previous_process_queue: {previous_process_queue}")
    
    # updated_process_queue = [new_deployment] + previous_process_queue
    # st.write(f"updated_process_queue: {updated_process_queue}")
    
    # sleep_time.sleep(10)  # simulate processing time
    
    process_queue.append(new_deployment)

    # write back to the vars file
    replace_vars(section="analyse_advanced", new_vars = {"process_queue": process_queue})
    
    # return
    
    
def write_selected_species(selected_species, cls_model_ID):
    # Construct the path to the JSON file
    json_path = os.path.join(ADDAXAI_FILES_ST, "models", "cls", cls_model_ID, "variables.json")
    
    # Load the existing JSON content
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # # test
    # all_classes = data["all_classes"]
    
    # missing = set(all_classes) - set(selected_species)  # in all_classes but not in selected_species
    # extra = set(selected_species) - set(all_classes)    # in selected_species but not in all_classes

    # st.write("Missing species:", len(missing))
    # st.write("Missing species:", sorted(missing))
    # st.write("Extra species:", len(extra))
    # st.write("Extra species:", sorted(extra))
    
    # st.write("are the lists the same:", sorted(all_classes) == sorted(selected_species))
        
    # sleep_time.sleep(5)
    
    # Update the selected_classes field
    data["selected_classes"] = selected_species
    
    # Write the updated content back to the file
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)



def species_selector_widget(taxon_mapping):
    nodes = build_taxon_tree(taxon_mapping)

    # st.write(nodes)

    # Initialize state
    if "selected_nodes" not in st.session_state:
        st.session_state.selected_nodes = []
    if "expanded_nodes" not in st.session_state:
        st.session_state.expanded_nodes = []
    if "last_selected" not in st.session_state:
        st.session_state.last_selected = {}



    col1, col2 = st.columns([1, 3])
    with col1:



        # UI - assuming tree_select is your widget for tree picking
        with st.popover(":material/pets: Select", use_container_width=True):
            
            butn_col1, butn_col2 = st.columns([1, 1])
            with butn_col1:
                if st.button(":material/select_check_box: Select all", key="expand_all_button", use_container_width=True):
                    # st.session_state.selected_nodes = [node["value"] for node in nodes]
                    st.session_state.selected_nodes = get_all_leaf_values(nodes)
                    # st.rerun()  # Force rerun to update the tree
            with butn_col2:
                if st.button(":material/check_box_outline_blank: Select none", key="collapse_all_button", use_container_width=True):
                    st.session_state.selected_nodes = []
                    # st.rerun()
                    
            with st.container(border=True):
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

        # Handle selection update and rerun
        if selected is not None:
            new_checked = selected.get("checked", [])
            new_expanded = selected.get("expanded", [])
            last_checked = st.session_state.last_selected.get("checked", [])
            last_expanded = st.session_state.last_selected.get("expanded", [])

            if new_checked != last_checked or new_expanded != last_expanded:
                st.session_state.selected_nodes = new_checked
                st.session_state.expanded_nodes = new_expanded
                st.session_state.last_selected = selected
                st.rerun()  # Force rerun

    # Count leaf nodes
    def count_leaf_nodes(nodes):
        count = 0
        for node in nodes:
            if "children" in node and node["children"]:
                count += count_leaf_nodes(node["children"])
            else:
                count += 1
        return count

    with col2:
        leaf_count = count_leaf_nodes(nodes)
        text = f"You have selected <code style='color:#086164; font-family:monospace;'>{len(st.session_state.selected_nodes)}</code> of <code style='color:#086164; font-family:monospace;'>{leaf_count}</code> classes. "
        st.markdown(
            f"""
                <div style="background-color: #f0f2f6; padding: 7px; border-radius: 8px;">
                    &nbsp;&nbsp;{text}
                </div>
                """,
            unsafe_allow_html=True
        )
    
    return st.session_state.selected_nodes
    # st.write("Selected nodes:", st.session_state.selected_nodes)


# def species_selector_widget(taxon_mapping):

#     nodes = [
#         {
#             "label": "Class Mammalia", "value": "class_mammalia", "children": [
#                 {
#                     "label": "Order Rodentia", "value": "order_rodentia", "children": [
#                         {
#                             "label": "Family Sciuridae (squirrels)", "value": "family_sciuridae", "children": [
#                                 {
#                                     "label": "Genus Tamias (chipmunks)", "value": "genus_tamias", "children": [
#                                         {"label": "Tamias striatus (eastern chipmunk)",
#                                          "value": "tamias_striatus"},
#                                         # This one is a species in genus Tamiasciurus, might move later
#                                         {"label": "Tamiasciurus hudsonicus (red squirrel)", "value": "tamiasciurus_hudsonicus"}
#                                     ]
#                                 },
#                                 {
#                                     "label": "Genus Sciurus (tree squirrels)", "value": "genus_sciurus", "children": [
#                                         {"label": "Sciurus niger (eastern fox squirrel)",
#                                          "value": "sciurus_niger"},
#                                         {"label": "Sciurus carolinensis (eastern gray squirrel)",
#                                          "value": "sciurus_carolinensis"},
#                                     ]
#                                 },
#                                 {
#                                     "label": "Genus Marmota (marmots)", "value": "genus_marmota", "children": [
#                                         {"label": "Marmota monax (groundhog)",
#                                          "value": "marmota_monax"},
#                                         {"label": "Marmota flaviventris (yellow-bellied marmot)",
#                                          "value": "marmota_flaviventris"},
#                                     ]
#                                 },
#                                 {"label": "Otospermophilus beecheyi (california ground squirrel)",
#                                  "value": "otospermophilus_beecheyi"}
#                             ]
#                         },
#                         {
#                             "label": "Family Muridae (gerbils and relatives)", "value": "family_muridae", "children": [
#                                 # add species/genus here if any
#                             ]
#                         },
#                         {
#                             "label": "Family Geomyidae (pocket gophers)", "value": "family_geomyidae", "children": [
#                                 # add species/genus here if any
#                             ]
#                         },
#                         {
#                             "label": "Family Erethizontidae (new world porcupines)", "value": "family_erethizontidae", "children": [
#                                 {"label": "Erethizon dorsatus (north american porcupine)",
#                                  "value": "erethizon_dorsatus"}
#                             ]
#                         }
#                     ]
#                 }
#             ]
#         },
#         {
#             "label": "Class Squamata", "value": "class_squamata", "children": [
#                 {
#                     "label": "Order Squamata (squamates)", "value": "order_squamata"
#                     # could add families/genera/species here if you have them
#                 }
#             ]
#         }
#     ]

#     # Initialize state
#     if "selected_nodes" not in st.session_state:
#         st.session_state.selected_nodes = []
#     if "expanded_nodes" not in st.session_state:
#         st.session_state.expanded_nodes = []
#     if "last_selected" not in st.session_state:
#         st.session_state.last_selected = {}

#     # UI
#     with st.popover("Select from tree", use_container_width=True):
#         selected = tree_select(
#             nodes,
#             check_model="leaf",
#             checked=st.session_state.selected_nodes,
#             expanded=st.session_state.expanded_nodes,
#             show_expand_all=True,
#             half_check_color="#086164",
#             check_color="#086164",
#             key="tree_select2"
#         )

#     # If the selection is new, update and rerun
#     if selected is not None:
#         new_checked = selected.get("checked", [])
#         new_expanded = selected.get("expanded", [])
#         last_checked = st.session_state.last_selected.get("checked", [])
#         last_expanded = st.session_state.last_selected.get("expanded", [])

#         if new_checked != last_checked or new_expanded != last_expanded:
#             st.session_state.selected_nodes = new_checked
#             st.session_state.expanded_nodes = new_expanded
#             st.session_state.last_selected = selected
#             st.rerun()  # 🔁 Force a rerun so the component picks up the change

#     # Feedback

#     def count_leaf_nodes(nodes):
#         count = 0
#         for node in nodes:
#             if "children" in node and node["children"]:
#                 count += count_leaf_nodes(node["children"])
#             else:
#                 count += 1
#         return count

#     # Example usage
#     leaf_count = count_leaf_nodes(nodes)
#     # st.write(f"Number of leaf nodes: {leaf_count}")

#     st.write("You selected:", len(st.session_state.selected_nodes),
#              " of ", leaf_count, "classes")
