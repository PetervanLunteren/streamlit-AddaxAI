import streamlit as st
import pandas as pd
from datetime import datetime, timedelta




from ads_utils.common import *


# cd /Applications/AddaxAI_files/AddaxAI && conda activate env-streamlit-addaxai && cd streamlit-AddaxAI/frontend && streamlit run main.py >> streamlit_log.txt 2>&1 &

# load language settings
txts = load_lang_txts()
# settings, _ = load_map()

general_settings_vars = load_vars(section = "general_settings")
lang = general_settings_vars["lang"]
mode = general_settings_vars["mode"]

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

step = load_step(section="analyse_advanced")
st.write("Current step:", step)

# --- Create stepper
stepper = StepperBar(
    steps=["Folder", "Deployment", "Model", "Settings", "Run"],
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
                                updates={"step": 1}) # 0 indexed
                    st.rerun()
            else:
                st.button(":material/arrow_forward: Next",
                          use_container_width=True,
                          disabled=True,
                          key = "project_next_button_dummy")

    # # folder selection
    # if step == 1:

    #     st.write("HELP TEXT")

    #     with st.container(border=True):
    #         print_widget_label(
    #             "Project", help_text="help text")

    #         # Store selection in session state
    #         if "projectID" not in st.session_state:
    #             st.session_state.projectID = None

    #         projectID = project_selector_widget()

    #     if projectID:

    #         # location metadata
    #         with st.container(border=True):
    #             print_widget_label(
    #                 "Location", help_text="help text")
    #             selected_locationID = location_selector_widget()
    #         # st.write("")

    #         # camera ID metadata
    #         if selected_locationID:
    #             with st.container(border=True):
    #                 print_widget_label(
    #                     "Start", help_text="help text")
    #                 selected_min_datetime = datetime_selector_widget()

    #             if selected_min_datetime:

    #                 if st.button("DEBUG"):
    #                     add_deployment(selected_min_datetime)

    #     # place the buttons
    #     col_btn_prev, col_btn_next = st.columns([1, 1])

    #     # the previous button is always enabled
    #     with col_btn_prev:
    #         if st.button(":material/replay: Start over", use_container_width=True):
    #             clear_vars(section="analyse_advanced")
    #             st.rerun()

    #     with col_btn_next:
    #         if projectID:
    #             if st.button(":material/arrow_forward: Next", use_container_width=True):
    #                 update_vars(section="analyse_advanced",
    #                             updates={"step": 2})
    #                 st.rerun()
    #         else:
    #             st.button(":material/arrow_forward: Next",
    #                       use_container_width=True, disabled=True)

    elif step == 1:

        with st.container(border=True):
            print_widget_label(
                "Project", help_text="help text")

            # # Store selection in session state
            # if "projectID" not in st.session_state:
            #     st.session_state.projectID = None

            selected_projectID = project_selector_widget()



        



        if selected_projectID:
            # # Only update session state and save settings if value changed
            # if st.session_state.get("projectID") != selected_projectID:
            #     st.session_state.projectID = selected_projectID

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
                    


                st.write("Selected datetime:", selected_min_datetime)
                st.write("Selected location ID:", selected_locationID)




                    # if st.button("DEBUG"):
                    #     add_deployment(selected_min_datetime)

        # place the buttons
        col_btn_prev, col_btn_next = st.columns([1, 1])

        # the previous button is always enabled
        with col_btn_prev:
            if st.button(":material/replay: Start over", use_container_width=True):
                clear_vars(section="analyse_advanced")
                st.rerun()

        if selected_projectID and selected_locationID and selected_min_datetime:
                with col_btn_next:
                    if selected_min_datetime:
                        if st.button(":material/arrow_forward: Next", use_container_width=True):
                            
                            update_vars(section="analyse_advanced",
                                        updates={
                                            # "step": 2,  # 0 indexed # DEBUG
                                                 "step" : 1,
                                                 "selected_projectID": selected_projectID,
                                                 "selected_locationID": selected_locationID,
                                                 "selected_min_datetime": selected_min_datetime})
                            add_deployment(selected_min_datetime = selected_min_datetime)
                            st.rerun()
                    else:
                        st.button(":material/arrow_forward: Next",
                                use_container_width=True, disabled=True)

    elif step == 2:
        st.write("This is where you can choose the mdoel stuff!")
        st.text_input("Location", key="location_input",
                      placeholder="Specify the location for your deployment")
    elif step == 3:
        st.write("In this step, you can select the camera used for the deployment.")
        st.text_input("Camera", key="camera_input",
                      placeholder="Select a camera for your deployment")
    elif step == 4:
        st.write("Finally, you can set the start date and time for the deployment. This is important for tracking when the data was collected.")
        st.date_input("Start Date", key="start_date_input",
                      value=datetime.now().date())


st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

##### END WORKING STEPPER BAR ######

# # select folder
# # st.write("")
# with st.container(border=True):
#     print_widget_label("Folder",
#                        help_text="Select the folder where your deployment is located.")
#     folder = browse_directory_widget(settings["folder"])

#     if folder and os.path.isdir(folder):
#         check_folder_metadata()
#         # st.write(st.session_state)

#     if folder and not os.path.isdir(folder):
#         st.error("The selected folder does not exist. Please select a valid folder.")
#         folder = None

# st.write("")

# # exit()

# # prev_folder = project_vars.get("folder", None)


# # if a folder is selected, show the model selection
# if folder and os.path.isdir(folder):

#     with st.container(border=True):
#         print_widget_label(
#             "Project", help_text="help text")

#         # Store selection in session state
#         if "projectID" not in st.session_state:
#             st.session_state.projectID = None

#         projectID = project_selector_widget()

#     # # Update settings only if the selection has changed
#     # if projectID and projectID != st.session_state.projectID:
#     #     st.session_state.projectID = projectID

#     if projectID:
#         # Only update session state and save settings if value changed
#         if st.session_state.get("projectID") != projectID:
#             st.session_state.projectID = projectID

#             # adjust the selected project
#             settings, _ = load_settings()
#             settings["project"] = projectID
#             with open(settings_file, "w") as file:
#                 json.dump(settings, file, indent=2)

#         # # write the selected project ID to the settings
#         # settings["projectID"] = projectID

#         # location metadata
#         with st.container(border=True):
#             print_widget_label(
#                 "Location", help_text="help text")
#             selected_locationID = location_selector_widget()
#         # st.write("")

#         # camera ID metadata
#         if selected_locationID:
#             # with st.container(border=True):
#             # print_widget_label(
#             #     "Which camera was used?", help_text="This is a unique identifier for the physical camera device itself—not the location. Since cameras can be moved between different locations over time, this ID helps track the specific camera regardless of where it’s deployed. It’s important for camtrapDP export and allows you to filter or analyze data by camera device. You can use any naming convention you like, as long as you can clearly identify which specific device it refers to.\n\nExamples: Browning51, Reconyx-A3, Peter's backup cam, etc.")
#             # selected_cameraID = camera_selector_widget()
#             # st.write("")

#             # datetime metadata
#             # if selected_cameraID:
#             with st.container(border=True):
#                 print_widget_label(
#                     "Start", help_text="help text")
#                 selected_min_datetime = datetime_selector_widget()
#                 # st.write("")
#                 # st.write("Selected datetime:", selected_min_datetime)
#                 # st.write("")

#             if selected_min_datetime:

#                 if st.button("DEBUG"):
#                     add_deployment(selected_min_datetime)


# #     # model section
# #     st.write("")
# #     st.subheader(":material/smart_toy: Model settings", divider="gray")


# # # prev_selected_cls_model = project_vars.get("selected_cls_model", "EUR-DF-v1.3")
# # # prev_selected_det_model = project_vars.get("selected_det_model", "MD5A")
# # # prev_selected_model_type = project_vars.get("selected_model_type", 'IDENTIFY')


# #     # widget to select the model type
# #     with st.container(border=True):
# #         print_widget_label("Would you like to only locate where the animals are or also identify their species?",
# #                            help_text="Choose whether you want the model to simply show where animals are, or also classify them by species.")
# #         selected_model_type = radio_buttons_with_captions(
# #             option_caption_dict={
# #                 "IDENTIFY": {"option": "Identify",
# #                              "caption": "Detect animals and automatically identify their species."},
# #                 "LOCATE": {"option": "Locate",
# #                            "caption": "Just detect animals and show where they are — I’ll identify them myself."}
# #             },
# #             key="model_type",
# #             scrollable=False,
# #             default_option=prev_selected_model_type)
# #     st.write("")

# #     # if the user just want to locate animals, then only select the detection model
# #     if selected_model_type == 'LOCATE':

# #         # select detection model
# #         with st.container(border=True):
# #             print_widget_label("Which model do you want to use to locate the animals?",
# #                                help_text="Here you can select the model of your choosing.")
# #             selected_det_model = select_model_widget(
# #                 "det", prev_selected_det_model)
# #         st.write("")

# #     # if the user want to indentify the species, then select both detection and classification models
# #     elif selected_model_type == 'IDENTIFY':

# #         # select detection model
# #         with st.container(border=True):
# #             print_widget_label("Which model do you want to use to locate the animals?",
# #                                help_text="Here you can select the model of your choosing.")
# #             selected_det_model = select_model_widget(
# #                 "det", prev_selected_det_model)
# #         st.write("")

# #         # select classification model
# #         if selected_det_model:
# #             with st.container(border=True):
# #                 print_widget_label("Which model do you want to use to locate the animals?",
# #                                    help_text="Here you can select the model of your choosing.")
# #                 selected_cls_model = select_model_widget(
# #                     "cls", prev_selected_cls_model)
# #             st.write("")

# #             # select species presence
# #             if selected_cls_model:

# #                 # load info about the selected model
# #                 slected_model_info = load_all_model_info(
# #                     "cls")[selected_cls_model]

# #                 # select classes
# #                 with st.container(border=True):
# #                     print_widget_label("Which species are present in your project area?",
# #                                        help_text="Select the species that are present in your project area.")
# #                     selected_classes = multiselect_checkboxes(slected_model_info['all_classes'],
# #                                                               slected_model_info['selected_classes'])
# #                 st.write("")

# #     # deployment metadata section is always shown, regardless of the model type
# #     if selected_model_type:
# #         print("dummy print to avoid streamlit warning")


# # # st.write(check_start_datetime())

# # #
# # # from streamlit_datetime_picker import date_time_picker, date_range_picker

# # # dt = date_time_picker(placeholder='Enter your birthday')

# # # Get date


# # # # Get time (hour and minute only)
# # # time = st.time_input("Select a time", step = 60)

# # # # Combine
# # # dt = datetime.combine(date, time)

# # # st.write("Selected datetime (no seconds):", dt)


# # st.write("")
# # st.write("")
# # st.write("")
# # if st.button(":material/rocket_launch: Let's go!", key="save_vars_button", use_container_width=True, type="primary"):

# #     if selected_model_type == 'IDENTIFY':
# #         save_project_vars({"folder": folder,
# #                           "selected_model_type": selected_model_type,
# #                           "selected_det_model": selected_det_model,
# #                           "selected_cls_model": selected_cls_model})

# #         st.write("DEBUG: selected_cls_model", selected_cls_model)

# #         # TODO: this is saving it to the wrong place, it should be saved to models/cls/<model_name>/variables.json
# #         save_cls_classes(selected_cls_model, selected_classes)
# #     elif selected_model_type == 'LOCATE':
# #         save_project_vars({"folder": folder,
# #                           "selected_model_type": selected_model_type,
# #                           "selected_det_model": selected_det_model})

# #     st.success("Variables saved successfully!")
