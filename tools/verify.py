# from utils import init_paths

import streamlit as st
import os
from utils.common import print_widget_label
import pandas as pd

st.markdown("*This would be where the users can verify their predictions. Somthing where you first select which label and conf range you are interested in, and then verify that selection*")

# # debug this needs to be real data somehow
# csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "namibia-example-images", "results_detections.csv")
# df = pd.read_csv(csv_path)


dummy_df = {
    "location": ["Location1", "Location1", "Location2", "Location2", "Location3"],
    "deployment": ["Deployment1", "Deployment2", "Deployment1", "Deployment2", "Deployment1"],
    "label": ["Lion", "Elephant", "Lion", "Elephant", "Lion"],
    "confidence": [0.95, 0.85, 0.90, 0.80, 0.75],
    "human_verified": [True, False, True, False, True],
    "absolute_path": ["/images/"] * 5,
    "relative_path": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]
}
df = pd.DataFrame(dummy_df)

with st.form(key="verify_form", clear_on_submit=False):

    st.subheader(":material/crop_free: Select predictions", divider="gray")
    n_total = df.shape[0]
    n_hv = df["human_verified"].sum()
    percentage_hv = n_hv / n_total * 100 if n_total > 0 else 0
    st.write(f"Here you can select which predictions you want to verify. You currently have a total of {df.shape[0]} observations, of which {df['human_verified'].sum()} ({percentage_hv:.0f}%) are human verified.")

    # Label multiselect
    st.divider()
    print_widget_label("Which label do you want to verify?", help_text = "help text")
    labels = df["label"].dropna().unique().tolist()
    selected_label = st.selectbox("Label", options=labels, label_visibility="collapsed")

    # Confidence range slider
    st.divider()
    print_widget_label("In what confidence range?", help_text = "help text")
    min_conf, max_conf = 0.01, 1.0
    selected_conf = st.slider("Confidence", min_value=0.01, max_value=1.00, value=(0.4, 1.0), step=0.01, format="%.2f", label_visibility="collapsed")



    st.divider()
    submitted = st.form_submit_button(":material/calculate: Count number of images to verify")

if submitted:

    # Apply filters
    filtered_df = df[
        (df["label"] == (selected_label)) &
        (df["confidence"] >= selected_conf[0]) &
        (df["confidence"] <= selected_conf[1])
    ]


    filtered_df['full_image_path'] = filtered_df['absolute_path'] + filtered_df['relative_path']
    
    # Save filtered dataframe to session state so it persists across reruns
    st.session_state['filtered_df'] = filtered_df
    st.session_state['selected_label'] = selected_label    
    
if 'filtered_df' in st.session_state:
    filtered_df = st.session_state['filtered_df']
    selected_label = st.session_state['selected_label']
    st.write(f"There are {filtered_df.shape[0]} {selected_label} observations within the set confidence range, which are on {filtered_df['full_image_path'].nunique()} images.")
    if st.button(":material/check: Verify selection"):
    
        st.markdown("*Here comes the actual verification of the images... I though to make use of these tools and see if we can make it work for our cause*")
        st.write("Streamlit Image Labelling (https://github.com/lit26/streamlit-img-label)")
        st.write("Streamlit Image Annotation (https://github.com/hirune924/Streamlit-Image-Annotation)")
        st.write("perhaps you can check out how they do it here: https://github.com/microsoft/aerial_wildlife_detection?tab=readme-ov-file#demo")
    
    
    
    