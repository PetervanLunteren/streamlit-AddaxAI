import streamlit as st
import pandas as pd
from backend.utils import load_txts, load_vars, save_global_vars, location_selector_widget, save_cls_classes, print_widget_label, multiselect_checkboxes, fetch_all_model_info, fetch_known_locations, show_model_info, add_new_location, radio_buttons_with_captions, select_model_widget
from backend.analyse_utils import browse_directory_widget

# load language settings
txts = load_txts()
vars = load_vars()
lang = vars.get("lang", "en")
mode = vars.get("mode", 1)
prev_selected_folder_str = vars.get("selected_folder_str", None)
prev_selected_cls_model = vars.get("selected_cls_model", "EUR-DF-v1.3")
prev_selected_det_model = vars.get("selected_det_model", "MD5A")
prev_selected_model_type = vars.get("selected_model_type", 'IDENTIFY')

st.markdown("*This is where the AI detection happens. Peter will figure this out as this is mainly a task of rearrangin the previous code.*")

# header
st.header(":material/rocket_launch: AI detect", divider="grey")
st.write(
    "You can analyze one deployment at a time using AI models. "
    "A deployment refers to all the images and videos stored on a single SD card retrieved from the field. "
    "This typically corresponds to one physical camera at one location during a specific period. "
    "The analysis results are saved to a recognition file, which can then be used by other tools in the platform."
)

# model section
st.write("")
st.subheader(":material/smart_toy: Model settings", divider="gray")

# select folder
st.write("")
with st.container(border=True):
    print_widget_label("Which deployment do you want to analyse?",
                       help_text="Select the folder where your deployment is located.")
    selected_folder_str = browse_directory_widget(prev_selected_folder_str)
st.write("")

# if a folder is selected, show the model selection and save the folder
if selected_folder_str:
    save_global_vars({"selected_folder_str": selected_folder_str})

    # widget to select the model type
    with st.container(border=True):
        print_widget_label("Would you like to only locate where the animals are or also identify their species?",
                           help_text="Choose whether you want the model to simply show where animals are, or also classify them by species.")
        selected_model_type = radio_buttons_with_captions(
            option_caption_dict={
                "IDENTIFY": {"option": "Identify",
                             "caption": "Detect animals and automatically identify their species."},
                "LOCATE": {"option": "Locate",
                           "caption": "Just detect animals and show where they are — I’ll identify them myself."}
            },
            key="model_type",
            scrollable=False,
            default_option=prev_selected_model_type)
    st.write("")

    # if the user just want to locate animals, then only select the detection model
    if selected_model_type == 'LOCATE':

        # select detection model
        with st.container(border=True):
            print_widget_label("Which model do you want to use to locate the animals?",
                               help_text="Here you can select the model of your choosing.")
            selected_det_model = select_model_widget(
                "det", prev_selected_det_model)
        st.write("")

    # if the user want to indentify the species, then select both detection and classification models
    elif selected_model_type == 'IDENTIFY':

        # select detection model
        with st.container(border=True):
            print_widget_label("Which model do you want to use to locate the animals?",
                               help_text="Here you can select the model of your choosing.")
            selected_det_model = select_model_widget(
                "det", prev_selected_det_model)
        st.write("")

        # select classification model
        if selected_det_model:
            with st.container(border=True):
                print_widget_label("Which model do you want to use to locate the animals?",
                                   help_text="Here you can select the model of your choosing.")
                selected_cls_model = select_model_widget(
                    "cls", prev_selected_cls_model)
            st.write("")

            # select species presence
            if selected_cls_model:

                # fetch info about the selected model
                slected_model_info = fetch_all_model_info(
                    "cls")[selected_cls_model]

                # select classes
                with st.container(border=True):
                    print_widget_label("Which species are present in your project area?",
                                       help_text="Select the species that are present in your project area.")
                    selected_classes = multiselect_checkboxes(slected_model_info['all_classes'],
                                                              slected_model_info['selected_classes'])
                st.write("")

# model section
st.write("")
st.subheader(":material/sd_card: Deployment metadata", divider="gray")
st.write("This is where you fill in the metadata of your deployment. This information will be saved in the recognition file "
         "and can be used to filter the results later on. It is optional, but required to make maps and other visualizations."
         " You can always change it later on.")

# location metadata
print_widget_label(
    "What was the location of this deployment?", help_text="help text")
selected_locationID = location_selector_widget()
st.write(f"Selected location: {selected_locationID}")


st.write("")
st.write("")
st.write("")
if st.button(":material/rocket_launch: Let's go!", key="save_vars_button", use_container_width=True, type="primary"):

    if selected_model_type == 'IDENTIFY':
        save_global_vars({"selected_folder_str": selected_folder_str,
                          "selected_model_type": selected_model_type,
                          "selected_det_model": selected_det_model,
                          "selected_cls_model": selected_cls_model})

        st.write("DEBUG: selected_cls_model", selected_cls_model)

        # TODO: this is saving it to the wrong place, it should be saved to models/cls/<model_name>/variables.json
        save_cls_classes(selected_cls_model, selected_classes)
    elif selected_model_type == 'LOCATE':
        save_global_vars({"selected_folder_str": selected_folder_str,
                          "selected_model_type": selected_model_type,
                          "selected_det_model": selected_det_model})

    st.success("Variables saved successfully!")
