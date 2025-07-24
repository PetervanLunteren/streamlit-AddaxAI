import streamlit as st
import os
from datetime import datetime
import tarfile
import requests
from tqdm import tqdm
import subprocess
import time as sleep_time
from ads_utils import init_paths
from streamlit_modal import Modal
import streamlit.components.v1 as components

# todo: only read the vars files here, not in the ads_utils module
# todo: make project select in the sidebar, all tools need a project no need to select it every time
# todo: revert everything back to st.session state, no need to use vars files, only write the vars to file if added to the queue

# todo: do the working, selected_lat, selected_lon, selected_cls_modelID, selected_det_modelID. 
# todo: also save the image or video that had the min_datetime, so that we can calculate the diff every time we need it "deployment_start_file". Then it can read the exif from the path. No need to read all exifs of all images.  searc h for deployment_start_file, deployment_start_datetime

# import local modules
from ads_utils.common import load_lang_txts, load_vars, StepperBar, print_widget_label, update_vars, clear_vars, info_box, MultiProgressBars
from ads_utils.analyse_advanced import (browse_directory_widget,
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
                                        run_env_installer
                                        )


# load files
txts = load_lang_txts()
general_settings_vars = load_vars(section="general_settings")
analyse_advanced_vars = load_vars(section="analyse_advanced")
model_meta = load_model_metadata()

# init vars
step = analyse_advanced_vars.get("step", 0)
lang = general_settings_vars["lang"]
mode = general_settings_vars["mode"]







# DEBUG
modal_1 = Modal(f"Installing ENV", key="installing-env", show_close_button=False)

if st.button("Show installation modal", use_container_width=True):
    modal_1.open()

if modal_1.is_open():
    with modal_1.container():
        # st.write("Ready to install.")
        # st.write("This may take a few minutes. Do not refresh the page.")

        # if st.button("Run installer now", key="run-installer"):
        run_env_installer(modal_1, "env-empty-debug")





# if st.button("Download and extract1", use_container_width=True):
#     modal_1.open()
#     run_env_installer(modal_1, "env-empty-debug")

    
                    
# if modal_1.is_open():
#     with modal_1.container():
        
#         info_box(
#             "The queue is currently being processed. Do not refresh the page or close the app, as this will interrupt the processing."
#             "It is recommended to avoid using your computer for other tasks, as the processing requires significant system resources."
#         )

#         url = "https://addaxaipremiumstorage.blob.core.windows.net/github-zips/latest/macos/envs/env-empty-debug.tar.xz"
#         local_filename = "envs/env-empty-debug.tar.xz"

#         response = requests.get(url, stream=True)
#         response.raise_for_status()
#         total_size = int(response.headers.get('content-length', 0))

#         # show progress bars
#         pbars = MultiProgressBars("Installing virual environment")
#         pbars.add_pbar("download", "Waiting to download...", "Downloading...", "Download complete!", max_value=total_size)
#         pbars.add_pbar("extract", "Waiting to extract...", "Extracting...", "Extraction complete!", max_value=None)
#         pbars.add_status("install", "Waiting to install...", "Installing...", "Installation complete!")

#         # download progress bar
#         block_size = 1024
#         pbar = tqdm(total=total_size / (1024 * 1024), unit='MB', unit_scale=False, unit_divisor=1)
#         with open(local_filename, 'wb') as f:
#             for data in response.iter_content(block_size):
#                 f.write(data)
#                 mb = len(data) / (1024 * 1024)
#                 pbar.update(mb)
#                 label = pbars.generate_label_from_tqdm(pbar)
#                 pbars.update("download", n=len(data), text=label)
#         pbar.close()
        
#         # Extract progress bar
#         with tarfile.open(local_filename, mode="r:xz") as tar:
#             members = tar.getmembers()
#             pbars.set_max_value("extract", len(members))  # ✅ This is clean and intuitive
#             pbar = tqdm(total=len(members), unit="files", unit_scale=False)
#             for member in members:
#                 tar.extract(member, path="envs/")
#                 pbar.update(1)
#                 label = pbars.generate_label_from_tqdm(pbar)
#                 pbars.update("extract", n=1, text=label)
#             pbar.close()
        
#         # pip install requirements
        
        
        
#         # install_placeholder.write("")

#         pip_cmd =                 [
#                     "/Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/envs/env-empty-debug/bin/python",
#                     "-m", "pip", "install",
#                     "-r", "/Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/envs/reqs/env-debug/macos/requirements.txt"
#                 ]
        
        
#         # Trigger pip install

#         status = pbars.update_status("install", phase="mid")

#         with status:
#             with st.container(border=True, height=300):
#                 output_placeholder = st.empty()
#                 live_output = "Booting up pip installation...\n\n"
#                 output_placeholder.code(live_output)

#                 process = subprocess.Popen(
#                     pip_cmd,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.STDOUT,
#                     text=True,
#                     bufsize=1
#                 )

#                 for line in process.stdout:
#                     live_output += line
#                     output_placeholder.code(live_output)

#                 process.wait()

#         pbars.update_status("install", phase="post")
    
    
    
#     modal_1.close()
    
    
    
    
    
    
    
    
    # status = pbars.update_status("install", phase="mid")
    
    
    # pip_cmd =                 [
    #                 "/Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/envs/env-empty-debug/bin/python",
    #                 "-m", "pip", "install",
    #                 "-r", "/Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/envs/reqs/env-debug/macos/requirements.txt"
    #             ]
    
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
    #         pbars.update_status("install", phase="post")
    
    
    

    # with st.status("Installing required Python packages...", expanded=False) as status:
    #     with st.container(border=True, height=300):
    #         output_placeholder = st.empty()
    #         live_output = "Booting up pip installation...\n\n"
    #         output_placeholder.code(live_output)

    #         process = subprocess.Popen(
    #             [
    #                 "/Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/envs/env-empty-debug/bin/python",
    #                 "-m", "pip", "install",
    #                 "-r", "/Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/envs/reqs/env-debug/macos/requirements.txt"
    #             ],
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.STDOUT,
    #             text=True,
    #             bufsize=1  # Line-buffered
    #         )

    #         # Read pip output line by line and update display
    #         for line in process.stdout:
    #             live_output += line
    #             output_placeholder.code(live_output)

    #         return_code = process.wait()

    #         if return_code == 0:
    #             status.update(label="Packages installed successfully!", state="complete")
    #         else:
    #             status.update(label="Package installation failed", state="error")


            
    # os.system("/Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/envs/env-empty-debug/bin/python -m pip install -r /Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI/envs/reqs/env-debug/macos/requirements.txt")
    



# import tarfile
# import os

    # # Open the archive
    # with tarfile.open(local_filename, mode="r:xz") as tar:
    #     members = tar.getmembers()
    #     total_files = len(members)

    #     # Reset tqdm and Streamlit progress bar for extraction
    #     pbar = tqdm(total=total_files, unit="file", unit_scale=False)

    #     for i, member in enumerate(members):
    #         tar.extract(member, path="envs/")  # extract to a directory
    #         pbar.update(1)

    #         label = pbars.generate_label_from_tqdm(pbar)
    #         pbars.update("extract", n=1, text=label)
    #         sleep_time.sleep(0.1)  # Simulate some processing time
            
    #         # st.write(label)

    #     pbar.close()


    # pbar = tqdm(total=100)
    # for _ in range(100):
    #     sleep_time.sleep(0.2)
    #     pbar.update(1)
    #     label = pbars.generate_label_from_tqdm(pbar)
    #     pbars.update("extract", n=1, text=label)

    # pbar.close()
    # pbars.set_description("download", "Done!")
    # pbars.set_description("extract", "Done!")













import os
import tarfile
import urllib.request

def download_with_progress(url, filename):
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = downloaded / total_size * 100
            if percent > 100:
                percent = 100
            st.write(f"Downloading: {int(percent)}%")
    
    urllib.request.urlretrieve(url, filename, reporthook=reporthook)
    st.write(f"Downloaded to {filename}")

if st.button("Download and extract env-empty-debug.tar.xz", use_container_width=True):

    # URL of the .tar.xz file
    url = "https://addaxaipremiumstorage.blob.core.windows.net/github-zips/latest/macos/envs/env-empty-debug.tar.xz"

    # Destination to extract the archive
    extract_to = "/Applications/AddaxAI_files/envs" 

    # Temporary filename to store the downloaded archive
    download_path = "/Applications/AddaxAI_files/envs/env-empty-debug.tar.xz"

    with st.expander("Downloading data...", expanded=True):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = downloaded / total_size
                if percent > 1:
                    percent = 1
                progress_bar.progress(percent)
                progress_text.text(f"Downloading... {percent*100:.2f}%")

        urllib.request.urlretrieve(url, download_path, reporthook=reporthook)

        progress_text.text("Download complete!")

    # with st.status("Downloading data...", expanded=True) as status:
    #     # Create a placeholder for substep progress
    #     progress_line = st.empty()

    #     def reporthook(block_num, block_size, total_size):
    #         if total_size > 0:
    #             downloaded = block_num * block_size
    #             percent = downloaded / total_size * 100
    #             if percent > 100:
    #                 percent = 100
    #             # Update just the progress line (substep)
    #             progress_line.text(f"Downloading... {percent:.2f}%")

    #     st.write("Preparing to download...")
    #     # time.sleep(1)

    #     urllib.request.urlretrieve(url, download_path, reporthook=reporthook)

    #     progress_line.text("Download complete!")
    #     status.update(label="Download complete!", state="complete", expanded=False)

    # with st.status("Starting download...", expanded=True) as status:

    #     def reporthook(block_num, block_size, total_size):
    #         if total_size > 0:
    #             downloaded = block_num * block_size
    #             percent = downloaded / total_size * 100
    #             if percent > 100:
    #                 percent = 100
    #             # Update the status label with percentage
    #             status.update(label=f"Downloading... {percent:.2f}%")

    #     st.write("Searching for data...")
    #     # time.sleep(2)
    #     st.write("Found URL.")
    #     # time.sleep(1)
    #     st.write("Downloading data...")
    #     # time.sleep(1)

    #     urllib.request.urlretrieve(url, download_path, reporthook=reporthook)

    #     status.update(label="Download complete!", state="complete", expanded=False)



    # with st.status("Downloading data..."):




    #     # Step 1: Download the file
    #     st.write("Downloading archive...")
    #     download_with_progress(url, download_path)
    #     st.write(f"Downloaded to {download_path}")

    #     # Step 2: Extract the .tar.xz archive
    #     st.write(f"Extracting to {extract_to}...")
    #     os.makedirs(extract_to, exist_ok=True)
    #     with tarfile.open(download_path, "r:xz") as tar:
    #         tar.extractall(path=extract_to)
        
    #     # Step 3: Clean up the downloaded archive
    #     os.remove(download_path)
    #     st.write(f"Removed downloaded archive: {download_path}")

    #     st.write("Done.")








# DEBUG













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
                clear_vars(section="analyse_advanced") # this is messing with the queue. it is deleting the queue too...
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

        # # load model metadata
        # model_meta = load_model_metadata()

        # select cls model
        with st.container(border=True):
            print_widget_label("Species identification model",
                            help_text="Here you can select the model of your choosing.")
            selected_cls_modelID = cls_model_selector_widget(model_meta)
        # st.write("")

        # select detection model

        with st.container(border=True):
            print_widget_label("Animal detection model",
                            help_text="The species identification model you selected above requires a detection model to locate the animals in the images. Here you can select the model of your choosing.")
            selected_det_modelID = det_model_selector_widget(model_meta)
        # st.write("")

        # place the buttons
        col_btn_prev, col_btn_next = st.columns([1, 1])

        # the previous button is always enabled
        with col_btn_prev:
            if st.button(":material/replay: Start over", use_container_width=True):
                clear_vars(section="analyse_advanced")
                st.rerun()

        with col_btn_next:
            if st.button(":material/arrow_forward: Next", use_container_width=True):

                update_vars(section="analyse_advanced",
                            updates={"step": 3,  # 0 indexed
                                    "selected_cls_modelID": selected_cls_modelID,
                                    "selected_det_modelID": selected_det_modelID})
                st.rerun()

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
    
    
    
    st.write(f"You currently have {len(process_queue)} deployments in the queue.")
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
                        folder_short = "..." + selected_folder[-45:] if len(selected_folder) > 45 else selected_folder
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
                        st.button(":material/delete:", help= "Remove from queue", key=f"remove_{i}",
                                use_container_width=True)
                    with col3:
                        # st.popover(f"Process {deployment['selected_folder']}")
                        with st.popover(":material/visibility:", help = "Show details", use_container_width=True):                        
                            st.write(f"**Project**: {deployment['selected_projectID']}")
                            st.write(f"**Location**: {deployment['selected_locationID']}")
                            st.write(f"**Species identification model**: {deployment['selected_cls_modelID']}")
                            st.write(f"**Animal detection model**: {deployment['selected_det_modelID']}")
                        
                        
    
    modal = Modal(f"Processing queue...", key="process_queue", show_close_button=False)
    if st.button(":material/rocket_launch: Process queue", use_container_width=True, type="primary"):
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
                subtask1_pbar = deployment_container.progress(0, text="Subtask 1")
                subtask2_pbar = deployment_container.progress(0, text="Subtask 2")
                subtask3_pbar = deployment_container.progress(0, text="Subtask 3")
                spinner2_placeholder = deployment_container.empty()
                

                for idx, deployment in enumerate(process_queue):
                    # Title
                    deployment_title_placeholder.write(f":material/sd_card: Deployment {idx + 1}")

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
                        subtask1_pbar.progress(i + 1, text=f"Deployment {idx + 1} ({i + 1}%)")

                    # Subtask 1
                    for i in range(100):
                        sleep_time.sleep(0.001)
                        subtask2_pbar.progress(i + 1, text=f"Preparing files ({i + 1}%)")

                    # Subtask 2
                    for i in range(100):
                        sleep_time.sleep(0.001)
                        subtask3_pbar.progress(i + 1, text=f"Uploading to server ({i + 1}%)")

                    # Spinner (e.g., initialization or file checks)
                    with spinner1_placeholder.status("Finalizing...", expanded=True):
                        sleep_time.sleep(0.5)  # Simulate some setup work

                    # Update overall queue progress
                    queue_pbar.progress(int((idx + 1) / len(process_queue) * 100),
                                        text=f"Completed {idx + 1} of {len(process_queue)} deployments")
                    
            
            # info_box("You're done!!!!! It took you X minutes, and you should know that the recognition files should not be altered, etc....")
            
            modal.close()
                        
                            
# if processing_bool:
#     st.warning("The queue is currently being processed. Please wait until the processing is finished before adding new deployments to the queue.")
#     # st.progress(0, "Processing queue...")  # TODO: this should be a real progress bar
    
#     progress_bar = st.progress(0)
#     # st.markdown("<br>" * 1000, unsafe_allow_html=True) # hacky tacky clear the screen

#     for i in range(100):
#         sleep_time.sleep(0.03)  # adjust speed
#         progress_bar.progress(i + 1)

#     # st.success("Done!")
#     update_vars(section="analyse_advanced", updates={"processing": False})
#     st.rerun()  # this should rerun the page to show the updated state
    
#     # st.stop()           
    
    


            
