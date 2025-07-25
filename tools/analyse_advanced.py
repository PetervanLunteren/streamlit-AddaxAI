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

# import local modules
from utils.common import load_lang_txts, load_vars, StepperBar, print_widget_label, update_vars, clear_vars, info_box, MultiProgressBars
from utils.analyse_advanced import (browse_directory_widget,
                                        check_folder_metadata,
                                        project_selector_widget,
                                        datetime_selector_widget,
                                        location_selector_widget,
                                        add_deployment,
                                        cls_model_selector_widget,
                                        load_model_metadata,
                                        det_model_selector_widget,
                                        species_selector_widget,
                                        load_taxon_mapping,
                                        add_deployment_to_queue,
                                        install_env
                                        )


# st.write(AddaxAI_files)

# load files
txts = load_lang_txts()
general_settings_vars = load_vars(section="general_settings")
analyse_advanced_vars = load_vars(section="analyse_advanced")
model_meta = load_model_metadata()

# init vars
step = analyse_advanced_vars.get("step", 0)
lang = general_settings_vars["lang"]
mode = general_settings_vars["mode"]


# the modals need to be defined before they are used
modal_install_env = Modal(f"Installing virtual environment",
                          key="installing-env", show_close_button=False)
if modal_install_env.is_open():
    with modal_install_env.container():
        install_env(modal_install_env, st.session_state["required_env_name"])

# if st.button("DEBUG - run MD", use_container_width=True):


def run_md(deployment_folder, pbars):

    model_file = "/Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/models/det/MD5A/md_v5a.0.0.pt"
    output_file = os.path.join(deployment_folder, "addaxai-deployment.json")
    command = [
        f"{ADDAXAI_FILES_ST}/envs/env-megadetector/bin/python",
        "-m", "megadetector.detection.run_detector_batch",
        model_file,
        deployment_folder,
        output_file
    ]



    

    status_placeholder = st.empty()

    # st.write("Running MegaDetector...")
    # with st.spinner(f"Running MegaDetector..."):
        # with st.expander("Show details", expanded=False):
            # with st.container(border=True, height=250):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=False,
        universal_newlines=True
    )

    # output_placeholder = st.empty()
    # output_lines = "Booting up MegaDetector...\n\n"
    # output_placeholder.code(output_lines, language="bash")
    for line in process.stdout:
        line = line.strip()
        pbars.update_from_tqdm_string("detector", line)
        # output_lines += line
        # output_placeholder.code(output_lines, language="bash")

    process.stdout.close()
    process.wait()

    # ✅ Show result message above
    if process.returncode == 0:
        status_placeholder.success("MD ran successfully!")
        # sleep_time.sleep(2)
        # modal.close()
    else:
        status_placeholder.error(
            f"Failed with exit code {process.returncode}.")
        # if st.button("Close window", use_container_width=True):
        #     modal.close()


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
                    update_vars(section="analyse_advanced",
                                updates={"step": 1})  # 0 indexed
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
                # this is messing with the queue. it is deleting the queue too...
                clear_vars(section="analyse_advanced")
                st.rerun()

        if selected_projectID and selected_locationID and selected_min_datetime:
            with col_btn_next:
                if selected_min_datetime:
                    if st.button(":material/arrow_forward: Next", use_container_width=True):

                        update_vars(section="analyse_advanced",
                                    updates={"step": 2,  # 0 indexed
                                             "selected_projectID": selected_projectID,
                                             "selected_locationID": selected_locationID,
                                             "selected_min_datetime": selected_min_datetime})
                        add_deployment(
                            selected_min_datetime=selected_min_datetime)
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
                req_env = model_meta['cls'][selected_cls_modelID]["env"]
                # if not os.path.exists(f"/Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/envs/{req_env}"):
                if not os.path.exists(os.path.join(ADDAXAI_FILES_ST, "envs", f"env-{req_env}")):
                    needs_installing = True
                    st.warning(
                        f"The selected classification model needs the virtual environment {req_env}. Please install it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.")

                    if st.button(f"Install {req_env}", use_container_width=False):
                        st.session_state["required_env_name"] = req_env
                        modal_install_env.open()
        # st.write("")

        # select detection model
        if selected_cls_modelID or selected_cls_modelID != "NONE":
            with st.container(border=True):
                print_widget_label("Animal detection model",
                                   help_text="The species identification model you selected above requires a detection model to locate the animals in the images. Here you can select the model of your choosing.")
                selected_det_modelID = det_model_selector_widget(model_meta)
                if selected_det_modelID:
                    req_env = model_meta['det'][selected_det_modelID]["env"]
                    if not os.path.exists(os.path.join(ADDAXAI_FILES_ST, "envs", f"env-{req_env}")):
                        needs_installing = True
                        st.warning(
                            f"The selected detection model needs the virtual environment {req_env}. Please install it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.")
                        if st.button(f"Install {req_env}", use_container_width=False):
                            st.session_state["required_env_name"] = req_env
                            modal_install_env.open()

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
                        update_vars(section="analyse_advanced",
                                    updates={"step": 3,  # 0 indexed
                                             "selected_cls_modelID": selected_cls_modelID,
                                             "selected_det_modelID": selected_det_modelID})
                        st.rerun()
            else:
                st.button(":material/arrow_forward: Next",
                          use_container_width=True, disabled=True,
                          key="model_next_button_dummy", help="You need to install the required virtual environment for the selected models before proceeding. ")

    elif step == 3:

        st.write("Species Selection!")

        selected_cls_modelID = analyse_advanced_vars["selected_cls_modelID"]
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
                clear_vars(section="analyse_advanced")
                st.rerun()

        with col_btn_next:
            if st.button(":material/playlist_add: Add to queue", use_container_width=True, type="primary"):
                update_vars(section="analyse_advanced",
                            updates={"step": 0})  # 0 indexed
                add_deployment_to_queue()

                # write_selected_species(selected_species = selected_species,
                #                        cls_model_ID = analyse_advanced_vars["selected_cls_modelID"])

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

    if st.button(":material/rocket_launch: Process queue 1", use_container_width=True, type="primary"):

        # overall_progress = st.empty()
        pbars = MultiProgressBars(container_label="Processing queue...",)
        
        pbars.add_pbar("detector", "Loading images...", "Detecting...", "Finished detection!", max_value=47)

        for idx, deployment in enumerate(process_queue):
            selected_folder = deployment['selected_folder']
            selected_projectID = deployment['selected_projectID']
            selected_locationID = deployment['selected_locationID']
            selected_min_datetime = deployment['selected_min_datetime']
            selected_det_modelID = deployment['selected_det_modelID']
            selected_cls_modelID = deployment['selected_cls_modelID']

            # run the MegaDetector
            pbars.update_label(f"Processing deployment: {idx} of {len(process_queue)} ...{selected_folder[-15:]}")
            # overall_progress.write(f"Processing deployment: {idx} of {len(process_queue)} ...{selected_folder[-15:]}")
            run_md(selected_folder, pbars)
        # pbars.update_label("")

    modal = Modal(f"Processing queue...", key="process_queue",
                  show_close_button=False)
    if st.button(":material/rocket_launch: Process queue 2", use_container_width=True, type="primary"):
        modal.open()
        st.write("RUN!")

    if modal.is_open():
        with modal.container():

            info_box(
                "The queue is currently being processed. Do not refresh the page or close the app, as this will interrupt the processing."
                "It is recommended to avoid using your computer for other tasks, as the processing requires significant system resources."
            )

            # st.progress(0, "Processing queue...")  # TODO: this should be a real progress bar
            _, cancel_col, _ = st.columns([1, 2, 1])

            with cancel_col:
                if st.button(":material/cancel: Cancel", use_container_width=True):
                    modal.close()

            # Overall queue progress
            st.write("Overall queue progress")
            queue_pbar = st.progress(0, text="Initializing queue...")

            # Deployment container with border
            with st.expander(":material/visibility: Show details", expanded=False):
                deployment_container = st.container(border=True)

                # Create placeholders inside the container
                deployment_title_placeholder = deployment_container.empty()
                spinner1_placeholder = deployment_container.empty()
                subtask1_pbar = deployment_container.progress(
                    0, text="Subtask 1")
                subtask2_pbar = deployment_container.progress(
                    0, text="Subtask 2")
                subtask3_pbar = deployment_container.progress(
                    0, text="Subtask 3")
                spinner2_placeholder = deployment_container.empty()

                for idx, deployment in enumerate(process_queue):
                    # Title
                    deployment_title_placeholder.write(
                        f":material/sd_card: Deployment {idx + 1}")

                    # Reset all progress bars
                    subtask1_pbar.progress(0, text="Starting deployment...")
                    subtask2_pbar.progress(0, text="Preparing files")
                    subtask3_pbar.progress(0, text="Uploading to server")

                    # Spinner (e.g., initialization or file checks)
                    with spinner1_placeholder.status("Initializing...", expanded=True):
                        sleep_time.sleep(0.5)  # Simulate some setup work

                    # Main deployment progress
                    for i in range(100):
                        sleep_time.sleep(0.001)
                        subtask1_pbar.progress(
                            i + 1, text=f"Deployment {idx + 1} ({i + 1}%)")

                    # Subtask 1
                    for i in range(100):
                        sleep_time.sleep(0.001)
                        subtask2_pbar.progress(
                            i + 1, text=f"Preparing files ({i + 1}%)")

                    # Subtask 2
                    for i in range(100):
                        sleep_time.sleep(0.001)
                        subtask3_pbar.progress(
                            i + 1, text=f"Uploading to server ({i + 1}%)")

                    # Spinner (e.g., initialization or file checks)
                    with spinner1_placeholder.status("Finalizing...", expanded=True):
                        sleep_time.sleep(0.5)  # Simulate some setup work

                    # Update overall queue progress
                    queue_pbar.progress(int((idx + 1) / len(process_queue) * 100),
                                        text=f"Completed {idx + 1} of {len(process_queue)} deployments")

            # info_box("You're done!!!!! It took you X minutes, and you should know that the recognition files should not be altered, etc....")

            modal.close()
