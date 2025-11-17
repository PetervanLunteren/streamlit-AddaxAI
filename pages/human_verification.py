import streamlit as st
import pandas as pd
from PIL import Image
import os
import numpy as np




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



# load data
detections = pd.read_csv('./assets/test_images/results_detections.csv')

# this is the test data
st.markdown("## Example detections")
st.write(detections)

# example UI
st.markdown("## Example UI")
st.image('./assets/images/mockup.png')