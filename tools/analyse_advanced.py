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
                         init_session_state, get_session_var, set_session_var, update_session_vars)
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
                                        load_taxon_mapping,
                                        add_deployment_to_queue,
                                        install_env,
                                        run_process_queue,
                                        download_model
                                        )


# st.write(AddaxAI_files)

# load files
txts = load_lang_txts()
general_settings_vars = load_vars(section="general_settings")
analyse_advanced_vars = load_vars(section="analyse_advanced")
model_meta = load_model_metadata()

# init session state for this tool
init_session_state("analyse_advanced")

# init vars
step = get_session_var("analyse_advanced", "step", 0)
lang = general_settings_vars["lang"]
mode = general_settings_vars["mode"]

# get from st.session_state
ADDAXAI_FILES = st.session_state["shared"]["ADDAXAI_FILES"]
ADDAXAI_FILES_ST = st.session_state["shared"]["ADDAXAI_FILES_ST"]

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
    
    
    


# the modals need to be defined before they are used
modal_install_env = Modal(f"Installing virtual environment",
                          key="installing-env", show_close_button=False)
if modal_install_env.is_open():
    with modal_install_env.container():
        install_env(modal_install_env, get_session_var("analyse_advanced", "required_env_name"))

# modal for processing queue
modal_process_queue = Modal(f"Processing queue...", key="process_queue",
                show_close_button=False)
if modal_process_queue.is_open():
    with modal_process_queue.container():
        # Process queue should always be loaded from persistent storage
        process_queue = analyse_advanced_vars.get("process_queue", [])
        run_process_queue(modal_process_queue, process_queue)

# modal for downloading models
modal_download_model = Modal(f"Downloading model...", key="download_model",
                show_close_button=False)
if modal_download_model.is_open():
    with modal_download_model.container():
        download_model(modal_download_model, get_session_var("analyse_advanced", "download_modelID"), model_meta)
        



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
                
                # download the env if needed
                req_env = model_meta['cls'][selected_cls_modelID]["env"]
                if not os.path.exists(os.path.join(ADDAXAI_FILES_ST, "envs", f"env-{req_env}")):
                    needs_installing = True
                    st.warning(
                        f"The selected classification model needs the virtual environment {req_env}. Please install it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.")
                    if st.button(f"Install {req_env}", use_container_width=False):
                        set_session_var("analyse_advanced", "required_env_name", req_env)
                        modal_install_env.open()
                        
                # download the model if needed
                model_fname = model_meta['cls'][selected_cls_modelID]["model_fname"]
                friendly_model_name = model_meta['cls'][selected_cls_modelID]["friendly_name"]
                if not os.path.exists(os.path.join(ADDAXAI_FILES_ST, "models", "cls", selected_cls_modelID, model_fname)):
                    needs_installing = True
                    st.warning(
                        f"The selectedclassification model __{friendly_model_name}__ still needs to be downloaded. Please download it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.")
                    if st.button(f"Download model", use_container_width=False, key="download_cls_model_button"):
                        set_session_var("analyse_advanced", "download_modelID", selected_cls_modelID)
                        modal_download_model.open()
        # st.write("")

        # select detection model
        if selected_cls_modelID or selected_cls_modelID != "NONE":
            with st.container(border=True):
                print_widget_label("Animal detection model",
                                   help_text="The species identification model you selected above requires a detection model to locate the animals in the images. Here you can select the model of your choosing.")
                selected_det_modelID = det_model_selector_widget(model_meta)
                if selected_det_modelID:
                    
                    # install the env if needed
                    req_env = model_meta['det'][selected_det_modelID]["env"]
                    if not os.path.exists(os.path.join(ADDAXAI_FILES_ST, "envs", f"env-{req_env}")):
                        needs_installing = True
                        st.warning(
                            f"The selected detection model needs the virtual environment {req_env}. Please install it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.")
                        if st.button(f"Install virtual environment *{req_env}*", use_container_width=False):
                            set_session_var("analyse_advanced", "required_env_name", req_env)
                            modal_install_env.open()
                    
                    # download the model if needed
                    model_fname = model_meta['det'][selected_det_modelID]["model_fname"]
                    friendly_model_name = model_meta['det'][selected_det_modelID]["friendly_name"]
                    if not os.path.exists(os.path.join(ADDAXAI_FILES_ST, "models", "det", selected_det_modelID, model_fname)):
                        needs_installing = True
                        st.warning(
                            f"The selected detection model {friendly_model_name} still needs to be downloaded. Please download it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.")
                        if st.button(f"Download model", use_container_width=False, key = "download_det_model_button"):
                            set_session_var("analyse_advanced", "download_modelID", selected_det_modelID)
                            modal_download_model.open()
                    

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
                            key="model_next_button_dummy", help="You need to install the required virtual environment for the selected models before proceeding. ")
            else:
                st.button(":material/arrow_forward: Next",
                        use_container_width=True, disabled=True,
                        key="model_next_button_dummy", help="You need to select both a species identification model and an animal detection model before proceeding.")

    elif step == 3:

        st.write("Species Selection!")

        selected_cls_modelID = get_session_var("analyse_advanced", "selected_cls_modelID")
        taxon_mapping = load_taxon_mapping(selected_cls_modelID)
        # st.write(taxon_mapping)
        with st.container(border=True):
            print_widget_label("Species presence",
                               help_text="Here you can select the model of your choosing.")
            selected_species = species_selector_widget(taxon_mapping)

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
        # Process queue is always persistent, no need to copy to session state
        modal_process_queue.open()
