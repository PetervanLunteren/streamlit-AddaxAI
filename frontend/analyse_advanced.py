import streamlit as st
import pandas as pd
from backend.utils import *

# load language settings
txts = load_txts()
vars = load_vars()
lang = vars.get("lang", "en")
mode = vars.get("mode", 1)
selected_folder_str = vars.get("selected_folder_str", None)
selected_cls_model_key = vars.get("selected_cls_model_key", "NAM-ADS-v1")
selected_det_model_idx = vars.get("selected_det_model_idx", 0)
selected_model_type_idx = vars.get("selected_model_type_idx", 0)


# header
st.header(":material/rocket_launch: AI detect", divider="grey")
st.write("Here you can deploy AI models to analyze your data. It saves the results to a recognition file which will enable the other tools to use it.")

# st.info(extract_gps_from_image("/Users/peter/Downloads/namib/05.JPG"))



# # Create base map
# m = fl.Map(location=[52.37, 4.89], zoom_start=12)

# m.add_child(
#     fl.ClickForMarker("<b>Lat:</b> ${lat}<br /><b>Lon:</b> ${lng}")
# )

# st.write(m._children)
# st.write(list(m._children.values()))
# st.write(m._children.keys())

# # Show map in Streamlit
# st_folium(m, width=700, height=500)






# import folium as fl
# import streamlit as st

# # Create base map
# m = fl.Map(location=[52.37, 4.89], zoom_start=12)

# # Initialize the variable to store the last marker
# previous_marker = None

# # Function to add a marker and remove the previous one
# def on_click(event):
#     global previous_marker
    
#     # Get latitude and longitude of the click
#     lat, lon = event.latlng
    
#     # Remove the previous marker if it exists
#     if previous_marker:
#         m.remove_child(previous_marker)
    
#     # Create the new marker and add it to the map
#     previous_marker = fl.Marker([lat, lon])
#     previous_marker.add_to(m)
#     previous_marker.bindPopup(f"<b>Lat:</b> {lat}<br /><b>Lon:</b> {lon}")

# # Use a custom LatLngPopup to get coordinates on click
# m.add_child(fl.LatLngPopup())

# # Show map in Streamlit
# st_folium(m, width=700, height=500)


# import folium

# # does not work
# st.info("Map 1")
# m = folium.Map(location=[45.3311, -121.7113], zoom_start=13)
# folium.Marker([45.3311, -121.7113]).add_to(m)
# st_folium(m, width=700, height=500)

# # Works
# st.info("Map 2")
# m = folium.Map(location=[45.3311, -121.7113], zoom_start=13)
# st_folium(m, width=700, height=500)

# # does not work 
# st.info("Map 3")
# m = folium.Map([45.35, -121.6972], zoom_start=12)
# folium.Marker(
#     location=[45.3288, -121.6625]
# ).add_to(m)


# model section
st.write("")
st.subheader(":material/smart_toy: Model settings", divider="gray")

# select folder
# st.divider()
st.write("")
print_widget_label("Which deployment do you want to analyse?", help_text = "Select the folder where your deployment is located.")
col1, col2 = st.columns([1, 3], vertical_alignment="center")
with col1:
    if st.button(":material/folder: Browse", key="folder_select_button", use_container_width=True):
        selected_folder_str = select_folder()
if not selected_folder_str:
    with col2:
        st.write('<span style="color: grey;">None selected...</span>', unsafe_allow_html=True)
else:
    with col2:
        selected_folder_str_short = "..." + selected_folder_str[-45:] if len(selected_folder_str) > 45 else selected_folder_str
        st.markdown(f'Selected folder <code style="color:#086164; font-family:monospace;">{selected_folder_str_short}</code>', unsafe_allow_html=True)

    save_global_vars({"selected_folder_str": selected_folder_str})

    # chose detector or classifier
    st.divider()
    # st.write("")
    print_widget_label("Do you just want to locate animals or also identify them?", help_text= "Help text")
    model_type_options = [
        "Yes, I'd like a model to help identify the animals.",
        "No, just show me where the animals are—I'll handle the identification myself."
    ]
    
    selected_model_type = st.radio(
        "Select model type",
        model_type_options,
        label_visibility="collapsed",
        index = selected_model_type_idx,
    )
    selected_model_type_idx = model_type_options.index(selected_model_type)
    
    # if the user just want to run MegaDetector
    if selected_model_type_idx == 1:

        # select detection model
        st.divider()
        # st.write("")
        print_widget_label("Which model do you want to use to locate the animals?", help_text = "Here you can select the model of your choosing.")

        # prepare radio button options
        det_model_info = fetch_all_model_info("det")
        keys = list(det_model_info.keys())
        friendly_names = [info["friendly_name"] for info in det_model_info.values()]
        descriptions = [info["short_description"] for info in det_model_info.values()]
        releases = [info["release"] for info in det_model_info.values()]
        developers = [info["developer"] for info in det_model_info.values()]

        # Streamlit radio button selection
        with st.container(border = True, height = 275):
            selected_friendly_name = st.radio(
                "Detection Model Selection",
                label_visibility="collapsed",
                options=friendly_names,
                index = selected_det_model_idx,
                captions= [f":material/calendar_today: Released {release} &nbsp;|&nbsp; :material/code_blocks: Developed by {developer} &nbsp;|&nbsp; :material/description: {desc}" for desc, release, developer in zip(descriptions, releases, developers)],
            )

        # Find the key of the selected model
        selected_model_key = next(
            key for key, info in det_model_info.items() if info["friendly_name"] == selected_friendly_name
        )

        # more info button
        st.button(f":material/info: More info about :grey-background[{selected_friendly_name}]", key="model_info_button")
        # if st.button(f":material/info: More info about :grey-background[{selected_friendly_name}]", key="model_info_button"):
        #     show_model_info(slected_model_info)

    # if the user want to indentify the species
    else:

        # select model
        st.divider()
        # st.write("")
        print_widget_label("Which species identification model do you want to use?", help_text= "Here you can select the model of your choosing.")
        
        
        # Fetch model data
        cls_model_info = fetch_all_model_info("cls")
        keys = list(cls_model_info.keys())
        friendly_names = [info["friendly_name"] for info in cls_model_info.values()]
        friendly_names_incl_flag = [f'{info["flag"]} {info["friendly_name"]}' for info in cls_model_info.values()]
        descriptions = [info["short_description"] for info in cls_model_info.values()]
        releases = [info["release"] for info in cls_model_info.values()]
        developers = [info["developer"] for info in cls_model_info.values()]

        # Determine index of last selection, default to 0 if not found
        selected_cls_model_idx = keys.index(selected_cls_model_key) if selected_cls_model_key in keys else 0

        # Streamlit radio button selection
        with st.container(border=True, height=275):
            selected_friendly_name_incl_flag = st.radio(
                "Classification Model Selection",
                label_visibility="collapsed",
                options=friendly_names_incl_flag,
                index=selected_cls_model_idx,
                captions=[f":material/calendar_today: Released {release} &nbsp;|&nbsp; "
                        f":material/code_blocks: Developed by {developer} &nbsp;|&nbsp; "
                        f":material/description: {desc}" 
                        for desc, release, developer in zip(descriptions, releases, developers)]
            )
        
        # Find the key of the selected model
        selected_friendly_name = selected_friendly_name_incl_flag.split(" ", 1)[1]  # Remove the flag from the friendly name
        
        selected_cls_model_key = keys[friendly_names_incl_flag.index(selected_friendly_name_incl_flag)]

        
        
        if selected_cls_model_key:
            
            slected_model_info = cls_model_info[selected_cls_model_key]

            # add button to show model info
            if st.button(f":material/info: More info about :grey-background[{selected_friendly_name}]", key="model_info_button"):
                show_model_info(slected_model_info)

            # select classes
            st.divider()
            print_widget_label("Which species are present in your project area?", "pets", "Select the species that are present in your project area.")
            selected_classes = st.multiselect(label = 'Select classes',
                                            options = slected_model_info['all_classes'],
                                            default = slected_model_info['selected_classes'],
                                            label_visibility="collapsed")

# model section
st.write("")
st.write("")
st.subheader(":material/photo_camera: Deployment metadata", divider="gray")
st.write("This is where you fill in the metadata of your deployment. This information will be saved in the recognition file and can be used to filter the results later on. It is optional, but required to make maps and other visualizations. You can always change it later on.")

# location metadata
print_widget_label("What was the location of this deployment?", help_text = "help text")
locations, selected_index = fetch_known_locations()
# location_ids = ['Location A', 'Location B', 'Location C']

# st.write("DEBUG: locations", locations)

if locations == []:
    if st.button(":material/add_circle: Add location", key="add_new_location_button", use_container_width=False):
        # add_new_location(selected_folder_str)
        add_new_location()
        # locations, selected_index = fetch_known_locations()
        # selected_index = 0
else:

    location_ids = [location["id"] for location in locations]
    
    selected_location = st.selectbox(
        "Choose a location ID",
        options=location_ids + ["+ Add new"],
        index=selected_index,
    )
    # If "Add a new location" is selected, show a dialog with a map and text input
    if selected_location.startswith("+ "):
        add_new_location()
        # add_new_location(selected_folder_str)


st.write("")
st.write("")
st.write("")
if st.button(":material/rocket_launch: Let's go!", key="save_vars_button", use_container_width=True, type="primary"):
    save_global_vars({"selected_folder_str": selected_folder_str,
                        "selected_model_type_idx": selected_model_type_idx,
                        "selected_cls_model_key": selected_cls_model_key})
    
    save_cls_classes(selected_cls_model_key, selected_classes)
    
    
    st.success("Variables saved successfully!")