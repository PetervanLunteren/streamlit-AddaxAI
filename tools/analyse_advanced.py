import streamlit as st
import os
from datetime import datetime

# todo: only read the vars files here, not in the ads_utils module

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
                                        write_selected_species
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
    steps=["Folder", "Deployment", "Model", "Species", "Run"],
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

        # # load model metadata
        # model_meta = load_model_metadata()

        # select cls model
        with st.container(border=True):
            print_widget_label("Species identification model",
                               help_text="Here you can select the model of your choosing.")
            selected_cls_model = cls_model_selector_widget(model_meta)
        # st.write("")

        # select detection model

        with st.container(border=True):
            print_widget_label("Animal detection model",
                               help_text="The species identification model you selected above requires a detection model to locate the animals in the images. Here you can select the model of your choosing.")
            selected_det_model = det_model_selector_widget(model_meta)
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
                                     "selected_cls_model": selected_cls_model,
                                     "selected_det_model": selected_det_model})
                st.rerun()

    elif step == 3:

        st.write("Species Selection!")

        selected_cls_model = analyse_advanced_vars["selected_cls_model"]
        taxon_mapping = load_taxon_mapping(selected_cls_model)
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
            if st.button(":material/arrow_forward: Next", use_container_width=True):
                update_vars(section="analyse_advanced",
                            updates={"step": 3})  # 0 indexed
                write_selected_species(selected_species = selected_species,
                                       cls_model_ID = analyse_advanced_vars["selected_cls_model"])

                
                
                st.rerun()

        # elif step == 4:
        #     st.write("Finally, you can set the start date and time for the deployment. This is important for tracking when the data was collected.")
        # st.date_input("Start Date", key="start_date_input",
        # value = datetime.now().date())


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

            # # select detection model
            # with st.container(border=True):
            #     print_widget_label("Which model do you want to use to locate the animals?",
            #                        help_text="Here you can select the model of your choosing.")
            #     selected_det_model = select_model_widget(
            #         "det", prev_selected_det_model)
            # st.write("")

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
