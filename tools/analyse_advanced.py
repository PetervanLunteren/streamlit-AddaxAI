import streamlit as st
import os
from datetime import datetime
import tarfile
import requests
from tqdm import tqdm 
import subprocess
import time as sleep_time
from utils import init_paths
import sys
from streamlit_modal import Modal
import streamlit.components.v1 as components

from utils.config import *

# todo: only read the vars files here, not in the utils module
# todo: make project select in the sidebar, all tools need a project no need to select it every time
# todo: revert everything back to st.session state, no need to use vars files, only write the vars to file if added to the queue

# todo: do the working, selected_lat, selected_lon, selected_cls_modelID, selected_det_modelID.
# todo: also save the image or video that had the min_datetime, so that we can calculate the diff every time we need it "deployment_start_file". Then it can read the exif from the path. No need to read all exifs of all images.  searc h for deployment_start_file, deployment_start_datetime
# todo: download the models too if needed


# import local modules
from utils.common import (load_lang_txts, load_vars, StepperBar, print_widget_label, 
                         update_vars, clear_vars, info_box, MultiProgressBars,
                         init_session_state, get_session_var, set_session_var, update_session_vars, warning_box)
from utils.analyse_advanced import (browse_directory_widget,
                                        check_folder_metadata,
                                        project_selector_widget,
                                        datetime_selector_widget,
                                        location_selector_widget,
                                        # add_deployment,
                                        cls_model_selector_widget,
                                        load_model_metadata,
                                        det_model_selector_widget,
                                        species_selector_widget,
                                        load_taxon_mapping_cached,  # Use cached version
                                        add_deployment_to_queue,
                                        install_env,
                                        run_process_queue,
                                        download_model,
                                        add_project_modal,
                                        add_location_modal,
                                        show_cls_model_info_modal,
                                        show_none_model_info_modal,
                                        species_selector_modal
                                        )


# st.write(AddaxAI_files)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED RESOURCE LOADING - Use cached session state values from main.py
# ═══════════════════════════════════════════════════════════════════════════════

# ✅ OPTIMIZATION 1: Use cached resources from main.py session state 
# Eliminates 3 expensive file reads per rerun:
# - load_lang_txts() -> txts (language JSON file)
# - load_model_metadata() -> model_meta (large model metadata JSON) 
# - load_vars("general_settings") -> lang/mode (settings JSON file)
txts = st.session_state["txts"]
model_meta = st.session_state["model_meta"]
lang = st.session_state["shared"]["lang"]
mode = st.session_state["shared"]["mode"]

# Load persistent vars (needs fresh data for queue management)
analyse_advanced_vars = load_vars(section="analyse_advanced")

# Initialize session state for this tool
init_session_state("analyse_advanced")

# Initialize step from session state
step = get_session_var("analyse_advanced", "step", 0)

from utils.config import *

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL AVAILABILITY CHECKING
# ═══════════════════════════════════════════════════════════════════════════════

# Simple filesystem checks for model and environment availability
# Lightweight operations that run on every check - no caching complexity

def check_model_availability(model_type, model_id, model_meta):
    """
    Check model and environment availability.
    Simple filesystem checks - no caching complexity.
    
    Args: 
        model_type: 'cls' or 'det'
        model_id: Model identifier
        model_meta: Model metadata dictionary
    
    Returns:
        dict: {
            'env_exists': bool,
            'model_exists': bool,
            'env_name': str,
            'model_fname': str,
            'friendly_name': str
        }
    """
    model_info = model_meta[model_type][model_id]
    env_name = model_info["env"]
    model_fname = model_info["model_fname"]
    friendly_name = model_info["friendly_name"]
    
    env_path = os.path.join(ADDAXAI_FILES_ST, "envs", f"env-{env_name}")
    model_path = os.path.join(ADDAXAI_FILES_ST, "models", model_type, model_id, model_fname)
    
    return {
        'env_exists': os.path.exists(env_path),
        'model_exists': os.path.exists(model_path),
        'env_name': env_name,
        'model_fname': model_fname,
        'friendly_name': friendly_name
    }


# # get from st.session_state
# ADDAXAI_FILES = st.session_state["shared"]["ADDAXAI_FILES"]
# ADDAXAI_FILES_ST = st.session_state["shared"]["ADDAXAI_FILES_ST"]

# st.write(f"TEMP_DIR: {TEMP_DIR}")
# st.write(f"CONFIG_DIR: {CONFIG_DIR}")

# st.write(st.session_state)

# # from utils.hf_downloader import HuggingFaceRepoDownloader
# if st.button("DEBUG: download repo HF"):
#     generate_wildlife_id()
    
#     # Initialize your UI progress bars
#     ui_pbars = MultiProgressBars("Download Progress")
#     ui_pbars.add_pbar("download", "Preparing download...", "Downloading...", "Download complete!")
    

#     downloader = HuggingFaceRepoDownloader()
    
#     # Test with the provided URL
#     test_repo = "https://huggingface.co/Addax-Data-Science/SAH-DRY-ADS-v1/tree/main"
    
#     model_ID = "MD5B-0-0"
    
#     success = downloader.download_repo(
#         model_ID=model_ID,
#         local_dir=os.path.join(ADDAXAI_FILES_ST, "models", "det", model_ID),
#         ui_pbars=ui_pbars,
#         pbar_id="download"
#     )
    
#     if success:
#         print("🎉 Test completed successfully!")
#     else:
#         print("😞 Test completed with some failures.")
    
    
    


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY MODALS - Now optimized to only create when needed using session state
# ═══════════════════════════════════════════════════════════════════════════════

# modal for installing environment - only create when needed
if get_session_var("analyse_advanced", "show_modal_install_env", False):
    modal_install_env = Modal(f"Installing virtual environment", key="installing-env", show_close_button=False)
    with modal_install_env.container():
        install_env(modal_install_env, get_session_var("analyse_advanced", "required_env_name"))

# modal for processing queue - only create when needed
if get_session_var("analyse_advanced", "show_modal_process_queue", False):
    modal_process_queue = Modal(f"Processing queue...", key="process_queue", show_close_button=False)
    with modal_process_queue.container():
        # Process queue should always be loaded from persistent storage
        process_queue = analyse_advanced_vars.get("process_queue", [])
        run_process_queue(modal_process_queue, process_queue)

# modal for downloading models - only create when needed
if get_session_var("analyse_advanced", "show_modal_download_model", False):
    modal_download_model = Modal(f"Downloading model...", key="download_model", show_close_button=False)
    with modal_download_model.container():
        download_model(modal_download_model, get_session_var("analyse_advanced", "download_modelID"), model_meta)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED MODAL MANAGEMENT - Only create modals when needed using session state
# ═══════════════════════════════════════════════════════════════════════════════
# This approach prevents expensive modal creation on every rerun by using session
# state flags to conditionally create modals only when they should be displayed.

# modal for adding new project - only create when needed
if get_session_var("analyse_advanced", "show_modal_add_project", False):
    modal_add_project = Modal(f"Adding project...", key="add_project", show_close_button=False)
    with modal_add_project.container(): 
        add_project_modal(modal_add_project)

# modal for adding new location - only create when needed
if get_session_var("analyse_advanced", "show_modal_add_location", False):
    modal_add_location = Modal(f"Adding location...", key="add_location", show_close_button=False)
    with modal_add_location.container():
        add_location_modal(modal_add_location)

# modal for showing classification model info - only create when needed
if get_session_var("analyse_advanced", "show_modal_cls_model_info", False):
    modal_show_cls_model_info = Modal(f"Model information", key="show_cls_model_info", show_close_button=False)
    with modal_show_cls_model_info.container():
        # Get model info from session state when modal is open 
        model_info = get_session_var("analyse_advanced", "modal_cls_model_info_data", {})
        show_cls_model_info_modal(modal_show_cls_model_info, model_info)

# modal for showing none model info - only create when needed
if get_session_var("analyse_advanced", "show_modal_none_model_info", False):
    modal_show_none_model_info = Modal(f"Model information", key="show_none_model_info", show_close_button=False)
    with modal_show_none_model_info.container():
        show_none_model_info_modal(modal_show_none_model_info)

# modal for species selector - only create when needed
if get_session_var("analyse_advanced", "show_modal_species_selector", False):
    modal_species_selector = Modal(f"Select species", key="species_selector", show_close_button=False)
    with modal_species_selector.container():
        # Get cached data from session state
        nodes = get_session_var("analyse_advanced", "modal_species_nodes", [])
        all_leaf_values = get_session_var("analyse_advanced", "modal_species_leaf_values", [])
        species_selector_modal(modal_species_selector, nodes, all_leaf_values)
        



# if st.button(":material/help: Help", use_container_width=True):
#     os.system("/Applications/AddaxAI_files/envs/env-base/bin/python" "/Applications/AddaxAI_files/cameratraps/megadetector/detection/run_detector_batch.py" "/Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/models/det/MD5A/md_v5a.0.0.pt" "/Users/peter/Downloads/example-projects-small/project_Kenya/location_001/deployment_001" "/Users/peter/Downloads/example-projects-small/project_Kenya/location_001/deployment_001/test_output.json")

st.markdown("*This is where the AI detection happens. Peter will figure this out as this is mainly a task of rearrangin the previous code.*")

# header
st.header(":material/rocket_launch: Add deployment to database", divider="grey")
st.write(
    "You can analyze one deployment at a time using AI models. "
    "A deployment refers to all the images and videos stored on a single SD card retrieved from the field. "
    "This typically corresponds to one physical camera at one location during a specific period. "
    "The analysis results are saved to a recognition file, which can then be used by other tools in the platform."
)

st.write("")
st.subheader(":material/sd_card: Deployment information", divider="grey")
st.write("Fill in the information related to this deployment. A deployment refers to all the images and videos stored on a single SD card retrieved from the field.")

###### STEPPER BAR ######

st.write("Current step:", step)

# --- Create stepper
stepper = StepperBar(
    steps=["Folder", "Deployment", "Model", "Species"],
    orientation="horizontal",
    active_color="#086164",
    completed_color="#0861647D",
    inactive_color="#dadfeb"
)
stepper.set_step(step)

# this is the stepper bar that will be used to navigate through the steps of the deployment creation process
with st.container(border=True):

    # stepper bar progress
    st.write("")
    st.markdown(stepper.display(), unsafe_allow_html=True)
    st.divider()

    # folder selection
    if step == 0:

        st.write("Here you can select the folder where your deployment is located. ")

        # select folder
        with st.container(border=True):
            print_widget_label("Folder",
                               help_text="Select the folder where your deployment is located.")
            selected_folder = browse_directory_widget()

            if selected_folder and os.path.isdir(selected_folder):
                check_folder_metadata()
                # st.write(st.session_state)

            if selected_folder and not os.path.isdir(selected_folder):
                st.error(
                    "The selected folder does not exist. Please select a valid folder.")
                selected_folder = None

        # place the buttons
        col_btn_prev, col_btn_next = st.columns([1, 1])

        with col_btn_next:
            if selected_folder and os.path.isdir(selected_folder):
                if st.button(":material/arrow_forward: Next", use_container_width=True):
                    # Store selected folder temporarily and advance step
                    set_session_var("analyse_advanced", "selected_folder", selected_folder)
                    set_session_var("analyse_advanced", "step", 1)
                    st.rerun()
            else:
                st.button(":material/arrow_forward: Next",
                          use_container_width=True,
                          disabled=True,
                          key="project_next_button_dummy")

    elif step == 1:

        with st.container(border=True):
            print_widget_label(
                "Project", help_text="help text")

            selected_projectID = project_selector_widget()

        if selected_projectID:

            # location metadata
            with st.container(border=True):
                print_widget_label(
                    "Location", help_text="help text")
                selected_locationID = location_selector_widget()
            # st.write("")

            # camera ID metadata
            if selected_locationID:
                with st.container(border=True):
                    print_widget_label(
                        "Start", help_text="help text")
                    selected_min_datetime = datetime_selector_widget()

        # place the buttons
        col_btn_prev, col_btn_next = st.columns([1, 1])

        # the previous button is always enabled
        with col_btn_prev:
            if st.button(":material/replay: Start over", use_container_width=True):
                # Clear only temporary session state, preserve persistent queue
                clear_vars("analyse_advanced")
                st.rerun()

        if selected_projectID and selected_locationID and selected_min_datetime:
            with col_btn_next:
                if selected_min_datetime:
                    if st.button(":material/arrow_forward: Next", use_container_width=True):
                        # Store selections temporarily and advance step
                        update_session_vars("analyse_advanced", {
                            "step": 2,
                            "selected_projectID": selected_projectID,
                            "selected_locationID": selected_locationID,
                            "selected_min_datetime": selected_min_datetime
                        })
                        # Update persistent general settings with committed projectID
                        update_vars(section="general_settings",
                                    updates={"selected_projectID": selected_projectID})
                        st.rerun()
                else:
                    st.button(":material/arrow_forward: Next",
                              use_container_width=True, disabled=True)

    elif step == 2:
        st.write("MODEL STUFF!")

        needs_installing = False
        selected_cls_modelID = None
        selected_det_modelID = None

        # # load model metadata
        # model_meta = load_model_metadata()

        # select cls model
        with st.container(border=True):
            print_widget_label("Species identification model",
                               help_text="Here you can select the model of your choosing.")
            selected_cls_modelID = cls_model_selector_widget(model_meta)

            if selected_cls_modelID and selected_cls_modelID != "NONE":
                
                # Check model availability (simple filesystem checks)
                availability = check_model_availability('cls', selected_cls_modelID, model_meta)
                
                # Check environment availability
                if not availability['env_exists']:
                    needs_installing = True
                    warning_box(title = "Virtual environment required",
                        msg = f"The selected model needs a specific virtual environment that is not yet installed. Please install it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.",
                        icon = ":material/warning:")
                    if st.button(f"Install virtual environment", key = "install_cls_virtual_env", use_container_width=False):
                        set_session_var("analyse_advanced", "required_env_name", availability['env_name'])
                        # Set session state flag to show modal on next rerun
                        set_session_var("analyse_advanced", "show_modal_install_env", True)
                        st.rerun()
                        
                # Check model file availability
                if not availability['model_exists']:
                    needs_installing = True
                    warning_box(
                        title= "Model download required",
                        msg = f"The selected classification model still needs to be downloaded. Please download it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.")
                    if st.button(f"Download model files", use_container_width=False, key="download_cls_model_button"):
                        set_session_var("analyse_advanced", "download_modelID", selected_cls_modelID)
                        # Set session state flag to show modal on next rerun
                        set_session_var("analyse_advanced", "show_modal_download_model", True)
                        st.rerun()
        # st.write("")

        # select detection model
        if selected_cls_modelID or selected_cls_modelID != "NONE":
            with st.container(border=True):
                print_widget_label("Animal detection model",
                                   help_text="The species identification model you selected above requires a detection model to locate the animals in the images. Here you can select the model of your choosing.")
                selected_det_modelID = det_model_selector_widget(model_meta)
                if selected_det_modelID:
                    
                    # Check model availability (simple filesystem checks)
                    availability = check_model_availability('det', selected_det_modelID, model_meta)
                    
                    # Check environment availability
                    if not availability['env_exists']:
                        needs_installing = True
                        warning_box(title = "Virtual environment required",
                            msg = f"The selected model needs a specific virtual environment that is not yet installed. Please install it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.",
                            icon = ":material/warning:")
                        if st.button(f"Install virtual environment", key="install_det_virtual_env", use_container_width=False):
                            set_session_var("analyse_advanced", "required_env_name", availability['env_name'])
                            # Set session state flag to show modal on next rerun
                            set_session_var("analyse_advanced", "show_modal_install_env", True)
                            st.rerun()
                    
                    # Check model file availability
                    if not availability['model_exists']:
                        needs_installing = True
                        warning_box(
                            title= "Model download required",
                            msg = f"The selected classification model still needs to be downloaded. Please download it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.")
                        if st.button(f"Download model files", use_container_width=False, key="download_det_model_button"):
                            set_session_var("analyse_advanced", "download_modelID", selected_det_modelID)
                            # Set session state flag to show modal on next rerun
                            set_session_var("analyse_advanced", "show_modal_download_model", True)
                            st.rerun()
                    

        # place the buttons
        col_btn_prev, col_btn_next = st.columns([1, 1])

        # the previous button is always enabled
        with col_btn_prev:
            if st.button(":material/replay: Start over", use_container_width=True):
                clear_vars(section="analyse_advanced")
                st.rerun()

        with col_btn_next:
            if (selected_cls_modelID and selected_det_modelID) or \
                    (selected_cls_modelID == "NONE" and selected_det_modelID):
                if not needs_installing:
                    if st.button(":material/arrow_forward: Next", use_container_width=True):
                        # Store model selections temporarily and advance step
                        update_session_vars("analyse_advanced", {
                            "step": 3,
                            "selected_cls_modelID": selected_cls_modelID,
                            "selected_det_modelID": selected_det_modelID
                        })
                        st.rerun()
                else:
                    st.button(":material/arrow_forward: Next",
                            use_container_width=True, disabled=True,
                            key="model_next_button_dummy", help="You need to install the required virtual environment or download the model files for the selected models before proceeding. ")
            else:
                st.button(":material/arrow_forward: Next",
                        use_container_width=True, disabled=True,
                        key="model_next_button_dummy", help="You need to select both a species identification model and an animal detection model before proceeding.")

    elif step == 3:

        st.write("Species Selection!")

        selected_cls_modelID = get_session_var("analyse_advanced", "selected_cls_modelID")
        # ✅ OPTIMIZATION 3: Cached taxon mapping loading
        # Only loads CSV file when classification model changes
        # Previous: CSV parsing on every step 3 visit
        # Now: Cached in session state by model ID
        taxon_mapping = load_taxon_mapping_cached(selected_cls_modelID)
        # st.write(taxon_mapping)
        with st.container(border=True):
            print_widget_label("Species presence",
                               help_text="Here you can select the model of your choosing.")
            selected_species = species_selector_widget(taxon_mapping, selected_cls_modelID)

        st.write("Selected species:", selected_species)

        # place the buttons
        col_btn_prev, col_btn_next = st.columns([1, 1])

        # the previous button is always enabled
        with col_btn_prev:
            if st.button(":material/replay: Start over", use_container_width=True):
                clear_vars("analyse_advanced") 
                st.rerun()

        with col_btn_next:
            if st.button(":material/playlist_add: Add to queue", use_container_width=True, type="primary"):
                # Store selected species temporarily, then commit all to queue
                set_session_var("analyse_advanced", "selected_species", selected_species)
                # Reset step to beginning
                set_session_var("analyse_advanced", "step", 0)
                # Add deployment to persistent queue
                add_deployment_to_queue()
                st.rerun()


# TODO: this shouldnt be in the same var as the other step vars etc., but step etc should be in the session state
# for now its fine, but rename it also that collabroators know that this is not the same as the step vars


st.write("")
st.subheader(":material/traffic_jam: Process queue", divider="grey")
process_queue = analyse_advanced_vars.get("process_queue", [])
if len(process_queue) == 0:
    st.write("You currently have no deployments in the queue. Please add a deployment to the queue to start processing.")
    st.button(":material/rocket_launch: Process queue", use_container_width=True, type="primary", disabled=True,
              help="You need to add a deployment to the queue first.")
else:

    st.write(
        f"You currently have {len(process_queue)} deployments in the queue.")
    # col1, _ = st.columns([1, 1])
    # with col1:

    with st.expander(":material/visibility: View queue details", expanded=False):
        with st.container(border=True, height=320):
            for i, deployment in enumerate(process_queue):
                with st.container(border=True):
                    selected_folder = deployment['selected_folder']
                    selected_projectID = deployment['selected_projectID']
                    selected_locationID = deployment['selected_locationID']
                    selected_min_datetime = deployment['selected_min_datetime']
                    selected_det_modelID = deployment['selected_det_modelID']
                    selected_cls_modelID = deployment['selected_cls_modelID']
                    col1, col2, col3 = st.columns([6, 1, 1])

                    with col1:
                        folder_short = "..." + \
                            selected_folder[-45:] if len(
                                selected_folder) > 45 else selected_folder
                        text = f"Folder &nbsp;&nbsp;<code style='color:#086164; font-family:monospace;'>{folder_short}</code>"
                        st.markdown(
                            f"""
                                <div style="background-color: #f0f2f6; padding: 7px; border-radius: 8px;">
                                    &nbsp;&nbsp;{text}
                                </div>
                                """,
                            unsafe_allow_html=True
                        )

                    with col2:
                        st.button(":material/delete:", help="Remove from queue", key=f"remove_{i}",
                                  use_container_width=True)
                    with col3:
                        # st.popover(f"Process {deployment['selected_folder']}")
                        with st.popover(":material/visibility:", help="Show details", use_container_width=True):
                            st.write(
                                f"**Project**: {deployment['selected_projectID']}")
                            st.write(
                                f"**Location**: {deployment['selected_locationID']}")
                            st.write(
                                f"**Species identification model**: {deployment['selected_cls_modelID']}")
                            st.write(
                                f"**Animal detection model**: {deployment['selected_det_modelID']}")

    if st.button(":material/rocket_launch: Process queue", use_container_width=True, type="primary"):
        # Set session state flag to show modal on next rerun
        set_session_var("analyse_advanced", "show_modal_process_queue", True)
        st.rerun()
