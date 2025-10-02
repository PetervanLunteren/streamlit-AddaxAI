import streamlit as st
import pandas as pd
from PIL import Image
import os
import numpy as np
from streamlit_image_zoom import image_zoom


# load data
@st.cache_data
def load():
    detections = pd.read_csv('./assets/test_images/results_detections.csv')
    files = pd.read_csv('./assets/test_images/results_files.csv')
    return detections, files

detections, files = load()
label_list = detections["label"].unique().tolist()


# labels & confidence range to filter data 
with st.container():
    selected_labels = st.multiselect(
        "Select labels", label_list, key="filter_labels"
    )
    conf_range = st.slider(
        "Confidence range", 0.0, 1.0, (0.4, 0.8), key="filter_conf"
    )


# filter data
filtered_detections = detections[
    detections["label"].isin(selected_labels) &
    detections["confidence"].between(conf_range[0], conf_range[1])
]


base_dir = "assets/test_images"


# filtered images gallery loop
for idx, row in filtered_detections.iterrows():
    img_path = os.path.join(base_dir, row['relative_path'])
    original_label = row['label']

    with st.container():
        col_img, col_panel = st.columns([3, 1])

        with col_img:
            st.header(original_label)
            #st.write(original_label)
            image = Image.open(img_path)
            image_zoom(image)

        with col_panel:
            default_idx = label_list.index(original_label) if original_label in label_list else 0
            new_label = st.selectbox(
                "Change label",
                options=label_list,
                index=default_idx,
                key=f"label_{idx}"
            )
            
        st.divider()



## This would be where the users can verify their predictions. It bascially gives the user to select certain labels and confidence ranges to double check the predicitons and adjust if needed

# Example, user is specifically intersted in wolf detections for his study. He want to verify all wolf predictions with a confidence between 0.4 and 0.8. 

# tools that might help you get started: 
# - Streamlit Image Labelling (https://github.com/lit26/streamlit-img-label)
# - Streamlit Image Annotation (https://github.com/hirune924/Streamlit-Image-Annotation)

# exmaple images and detections can be found here: assets/test_images
# Detections-level results: assets/test_images/results_detections.csv
# file-level results: assets/test_images/results_files.csv


# It would be great if we could have these features:
# - select labels to filter by
# - select confidence range to filter by
# - next image button
# - previous image button
# - show image with bounding boxes and labels
# - allow user to change label or delete detection
# - save the changes (can be a dummy button for now)
# - show progress (e.g., "You have verified 10 out of 50 images to review")
# - perhaps you can check out how they do it here: https://github.com/microsoft/aerial_wildlife_detection?tab=readme-ov-file#demo


# if that works, we can perhjaps move to more advanced features like:
# - zoom in/out
# - keyboard shortcuts for common actions (e.g., 'e' edit, 'd' delete, 'spacebar' for verify next image, arrows for navigation, etc, see https://pypi.org/project/streamlit-shortcuts/)
# - burst mode (see all images in a sequence and verify them together)
# - video frames verification
