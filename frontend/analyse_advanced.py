import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from backend.utils import *

# cd /Applications/AddaxAI_files/AddaxAI && conda activate env-streamlit-addaxai && cd streamlit-AddaxAI/frontend && streamlit run main.py >> streamlit_log.txt 2>&1 &

# load language settings
txts = load_txts()
settings, _ = load_settings()
# project_vars = load_project_vars()

# st.write("project_vars:", project_vars)

# exit()

lang = settings["lang"]
mode = settings["mode"]


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

# select folder
# st.write("")
with st.container(border=True):
    print_widget_label("Folder",
                       help_text="Select the folder where your deployment is located.")
    selected_folder = browse_directory_widget(settings["selected_folder"])
    
    if selected_folder and os.path.isdir(selected_folder):
        check_folder_metadata()
        # st.write(st.session_state)
    
    if selected_folder and not os.path.isdir(selected_folder):
        st.error("The selected folder does not exist. Please select a valid folder.")
        selected_folder = None
    
# st.write("")

# exit()

# prev_selected_folder = project_vars.get("selected_folder", None)



# if a folder is selected, show the model selection
if selected_folder and os.path.isdir(selected_folder):
    
    with st.container(border=True):
        print_widget_label(
            "Project", help_text="help text")
        
        # Store selection in session state
        if "selected_projectID" not in st.session_state:
            st.session_state.selected_projectID = None
        
        selected_projectID = project_selector_widget()
    
    # # Update settings only if the selection has changed
    # if selected_projectID and selected_projectID != st.session_state.selected_projectID:
    #     st.session_state.selected_projectID = selected_projectID
    
    if selected_projectID:
        # Only update session state and save settings if value changed
        if st.session_state.get("selected_projectID") != selected_projectID:
            st.session_state.selected_projectID = selected_projectID
        
            # adjust the selected project
            settings, _ = load_settings()
            settings["selected_project"] = selected_projectID
            with open(settings_file, "w") as file:
                json.dump(settings, file, indent=2)
        
        # # write the selected project ID to the settings
        # settings["selected_projectID"] = selected_projectID
    
        # location metadata
        with st.container(border=True):
            print_widget_label(
                "Location", help_text="help text")
            selected_locationID = location_selector_widget()
        # st.write("")

        # camera ID metadata
        if selected_locationID:
            # with st.container(border=True):
                # print_widget_label(
                #     "Which camera was used?", help_text="This is a unique identifier for the physical camera device itself—not the location. Since cameras can be moved between different locations over time, this ID helps track the specific camera regardless of where it’s deployed. It’s important for camtrapDP export and allows you to filter or analyze data by camera device. You can use any naming convention you like, as long as you can clearly identify which specific device it refers to.\n\nExamples: Browning51, Reconyx-A3, Peter's backup cam, etc.")
                # selected_cameraID = camera_selector_widget()
            # st.write("")
            
            # datetime metadata
            # if selected_cameraID:
            with st.container(border=True):
                print_widget_label(
                    "Start", help_text="help text")
                selected_datetime = datetime_selector_widget()
                st.write("")
                st.write("Selected datetime:", selected_datetime)
                # st.write("")
                
            if selected_datetime:
                
                if st.button("DEBUG"):
                    add_deployment(selected_datetime)
                    
                    
    
    
    
    
    
    
    
    
    
#     # model section
#     st.write("")
#     st.subheader(":material/smart_toy: Model settings", divider="gray")
    
    
    
    
    
# # prev_selected_cls_model = project_vars.get("selected_cls_model", "EUR-DF-v1.3")
# # prev_selected_det_model = project_vars.get("selected_det_model", "MD5A")
# # prev_selected_model_type = project_vars.get("selected_model_type", 'IDENTIFY')
    
    
    
#     # widget to select the model type
#     with st.container(border=True):
#         print_widget_label("Would you like to only locate where the animals are or also identify their species?",
#                            help_text="Choose whether you want the model to simply show where animals are, or also classify them by species.")
#         selected_model_type = radio_buttons_with_captions(
#             option_caption_dict={
#                 "IDENTIFY": {"option": "Identify",
#                              "caption": "Detect animals and automatically identify their species."},
#                 "LOCATE": {"option": "Locate",
#                            "caption": "Just detect animals and show where they are — I’ll identify them myself."}
#             },
#             key="model_type",
#             scrollable=False,
#             default_option=prev_selected_model_type)
#     st.write("")

#     # if the user just want to locate animals, then only select the detection model
#     if selected_model_type == 'LOCATE':

#         # select detection model
#         with st.container(border=True):
#             print_widget_label("Which model do you want to use to locate the animals?",
#                                help_text="Here you can select the model of your choosing.")
#             selected_det_model = select_model_widget(
#                 "det", prev_selected_det_model)
#         st.write("")

#     # if the user want to indentify the species, then select both detection and classification models
#     elif selected_model_type == 'IDENTIFY':

#         # select detection model
#         with st.container(border=True):
#             print_widget_label("Which model do you want to use to locate the animals?",
#                                help_text="Here you can select the model of your choosing.")
#             selected_det_model = select_model_widget(
#                 "det", prev_selected_det_model)
#         st.write("")

#         # select classification model
#         if selected_det_model:
#             with st.container(border=True):
#                 print_widget_label("Which model do you want to use to locate the animals?",
#                                    help_text="Here you can select the model of your choosing.")
#                 selected_cls_model = select_model_widget(
#                     "cls", prev_selected_cls_model)
#             st.write("")

#             # select species presence
#             if selected_cls_model:

#                 # fetch info about the selected model
#                 slected_model_info = fetch_all_model_info(
#                     "cls")[selected_cls_model]

#                 # select classes
#                 with st.container(border=True):
#                     print_widget_label("Which species are present in your project area?",
#                                        help_text="Select the species that are present in your project area.")
#                     selected_classes = multiselect_checkboxes(slected_model_info['all_classes'],
#                                                               slected_model_info['selected_classes'])
#                 st.write("")

#     # deployment metadata section is always shown, regardless of the model type
#     if selected_model_type:
#         print("dummy print to avoid streamlit warning")




# # st.write(check_start_datetime())

# # 
# # from streamlit_datetime_picker import date_time_picker, date_range_picker

# # dt = date_time_picker(placeholder='Enter your birthday')

# # Get date


# # # Get time (hour and minute only)
# # time = st.time_input("Select a time", step = 60)

# # # Combine
# # dt = datetime.combine(date, time)

# # st.write("Selected datetime (no seconds):", dt)
                


# st.write("")
# st.write("")
# st.write("")
# if st.button(":material/rocket_launch: Let's go!", key="save_vars_button", use_container_width=True, type="primary"):

#     if selected_model_type == 'IDENTIFY':
#         save_project_vars({"selected_folder": selected_folder,
#                           "selected_model_type": selected_model_type,
#                           "selected_det_model": selected_det_model,
#                           "selected_cls_model": selected_cls_model})

#         st.write("DEBUG: selected_cls_model", selected_cls_model)

#         # TODO: this is saving it to the wrong place, it should be saved to models/cls/<model_name>/variables.json
#         save_cls_classes(selected_cls_model, selected_classes)
#     elif selected_model_type == 'LOCATE':
#         save_project_vars({"selected_folder": selected_folder,
#                           "selected_model_type": selected_model_type,
#                           "selected_det_model": selected_det_model})

#     st.success("Variables saved successfully!")
