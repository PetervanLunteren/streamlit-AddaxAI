import streamlit as st
import os
from datetime import datetime
import time as sleep_time
from ads_utils import init_paths

# todo: only read the vars files here, not in the ads_utils module
# todo: make project select in the sidebar, all tools need a project no need to select it every time
# todo: revert everything back to st.session state, no need to use vars files, only write the vars to file if added to the queue

# todo: do the working, selected_lat, selected_lon, selected_cls_modelID, selected_det_modelID. 
# todo: also save the image or video that had the min_datetime, so that we can calculate the diff every time we need it "deployment_start_file". Then it can read the exif from the path. No need to read all exifs of all images.  searc h for deployment_start_file, deployment_start_datetime

# import local modules
from ads_utils.common import load_lang_txts, load_vars, StepperBar, print_widget_label, update_vars, clear_vars
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
                                        add_deployment_to_queue
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

# 
processing_bool = analyse_advanced_vars.get("processing", False)
if processing_bool:
    st.warning("The queue is currently being processed. Please wait until the processing is finished before adding new deployments to the queue.")
    # st.progress(0, "Processing queue...")  # TODO: this should be a real progress bar
    
    progress_bar = st.progress(0)

    for i in range(100):
        sleep_time.sleep(0.03)  # adjust speed
        progress_bar.progress(i + 1)

    # st.success("Done!")
    update_vars(section="analyse_advanced", updates={"processing": False})
    st.rerun()  # this should rerun the page to show the updated state
    
    # st.stop()
    
if not processing_bool:

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


    # def start_processing_queue():
        # update_vars(section = "analyse_advanced", updates = {"processing": True})
        

        
        
        # st.rerun()


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
        if st.button(":material/rocket_launch: Process queue", use_container_width=True, type="primary"):
            # st.write("Processing queue...") # TODO
            
            
            update_vars(section="analyse_advanced",
                        updates={"processing": True}) 
            st.rerun()
            
            # st.rerun()  # this should start the processing of the queue, but for now it just reruns the page
        # with st.expander("View queue details", expanded=True):
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