import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from backend.utils import *


# sys.path.insert(0, "/Users/peter/Desktop/streamlit_tree_selector")


# # import streamlit_tree_select
# # print(streamlit_tree_select.__file__)



# from streamlit_tree_select import tree_select

# nodes = [
#     {"value": "mars", "label": "Mars"},
#     {"value": "venus", "label": "Venus"},
#     {"value": "jupiter", "label": "Jupiter"},
# ]

# checked = []
# expanded = []

# result = tree_select(nodes, checked=checked, expanded=expanded, single_select=False)

# st.write("Selected nodes:", result)










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


# from st_ant_tree import st_ant_tree

# # Example tree structure
# tree_data = [
#     {
#         "title": "Fruits",
#         "value": "fruits",
#         "key": "fruits",
#         "children": [
#             {"title": "Apple", "value": "apple", "key": "apple"},
#             {"title": "Banana", "value": "banana", "key": "banana"},
#         ],
#     },
#     {
#         "title": "Vegetables",
#         "value": "vegetables",
#         "key": "vegetables",
#         "children": [
#             {"title": "Carrot", "value": "carrot", "key": "carrot"},
#             {"title": "Broccoli", "value": "broccoli", "key": "broccoli"},
#         ],
#     },
# ]

# st_ant_tree(treeData=tree_data, multiple=False, key = "y")

# # Show the component with arrows and checkboxes
# selected = st_ant_tree(
#     treeData=tree_data,
#     placeholder="Select your favorite foods",
#     multiple=True,
#     treeCheckable=True,
#     treeDefaultExpandAll=True,
#     showArrow=True,
#     treeLine=True,
#     bordered=True,
#     allowClear=True,
#     maxTagCount=0
# )

# st.write("You selected:", selected)


from streamlit_tree_select import tree_select

nodes = [
    {
        "label": "Class Mammalia", "value": "class_mammalia", "children": [
            {
                "label": "Order Rodentia", "value": "order_rodentia", "children": [
                    {
                        "label": "Family Sciuridae (squirrels)", "value": "family_sciuridae", "children": [
                            {
                                "label": "Genus Tamias (chipmunks)", "value": "genus_tamias", "children": [
                                    {"label": "Tamias striatus (eastern chipmunk)", "value": "tamias_striatus"},
                                    {"label": "Tamiasciurus hudsonicus (red squirrel)", "value": "tamiasciurus_hudsonicus"}  # This one is a species in genus Tamiasciurus, might move later
                                ]
                            },
                            {
                                "label": "Genus Sciurus (tree squirrels)", "value": "genus_sciurus", "children": [
                                    {"label": "Sciurus niger (eastern fox squirrel)", "value": "sciurus_niger"},
                                    {"label": "Sciurus carolinensis (eastern gray squirrel)", "value": "sciurus_carolinensis"},
                                ]
                            },
                            {
                                "label": "Genus Marmota (marmots)", "value": "genus_marmota", "children": [
                                    {"label": "Marmota monax (groundhog)", "value": "marmota_monax"},
                                    {"label": "Marmota flaviventris (yellow-bellied marmot)", "value": "marmota_flaviventris"},
                                ]
                            },
                            {"label": "Otospermophilus beecheyi (california ground squirrel)", "value": "otospermophilus_beecheyi"}
                        ]
                    },
                    {
                        "label": "Family Muridae (gerbils and relatives)", "value": "family_muridae", "children": [
                            # add species/genus here if any
                        ]
                    },
                    {
                        "label": "Family Geomyidae (pocket gophers)", "value": "family_geomyidae", "children": [
                            # add species/genus here if any
                        ]
                    },
                    {
                        "label": "Family Erethizontidae (new world porcupines)", "value": "family_erethizontidae", "children": [
                            {"label": "Erethizon dorsatus (north american porcupine)", "value": "erethizon_dorsatus"}
                        ]
                    }
                ]
            }
        ]
    },
    {
        "label": "Class Squamata", "value": "class_squamata", "children": [
            {
                "label": "Order Squamata (squamates)", "value": "order_squamata"
                # could add families/genera/species here if you have them
            }
        ]
    }
]



# Initialize state
if "selected_nodes" not in st.session_state:
    st.session_state.selected_nodes = []
if "expanded_nodes" not in st.session_state:
    st.session_state.expanded_nodes = []
if "last_selected" not in st.session_state:
    st.session_state.last_selected = {}

# UI
with st.popover("Select from tree", use_container_width=True):
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

# If the selection is new, update and rerun
if selected is not None:
    new_checked = selected.get("checked", [])
    new_expanded = selected.get("expanded", [])
    last_checked = st.session_state.last_selected.get("checked", [])
    last_expanded = st.session_state.last_selected.get("expanded", [])

    if new_checked != last_checked or new_expanded != last_expanded:
        st.session_state.selected_nodes = new_checked
        st.session_state.expanded_nodes = new_expanded
        st.session_state.last_selected = selected
        st.rerun()  # 🔁 Force a rerun so the component picks up the change

# Feedback


def count_leaf_nodes(nodes):
    count = 0
    for node in nodes:
        if "children" in node and node["children"]:
            count += count_leaf_nodes(node["children"])
        else:
            count += 1
    return count

# Example usage
leaf_count = count_leaf_nodes(nodes)
# st.write(f"Number of leaf nodes: {leaf_count}")

st.write("You selected:", len(st.session_state.selected_nodes), " of ", leaf_count, "classes")

# selected = st_ant_tree(
#     treeData=...,
#     multiple=True,
#     treeCheckable=True,
#     maxTagCount=1,  # Collapse to "+N more"
# )

# if selected:
#     st.write(f"Selected {len(selected)} species")


# select folder
# st.write("")
with st.container(border=True):
    print_widget_label("Folder",
                       help_text="Select the folder where your deployment is located.")
    selected_folder = browse_directory_widget(settings["selected_folder"])
# st.write("")

# exit()

# prev_selected_folder = project_vars.get("selected_folder", None)



# if a folder is selected, show the model selection
if selected_folder:
    
    
    st.write(check_folder_metadata())
    
    
    
    
    
    
    
    
    
    
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
