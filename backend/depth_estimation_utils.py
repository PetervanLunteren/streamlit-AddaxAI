import streamlit as st
import os
import pandas as pd

def deployment_selector():
    # Load data
    csv_path = os.path.join( # DEBUG
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "namibia-example-images",
        "results_detections.csv"
    )
    df = pd.read_csv(csv_path)

    # Make sure 'depth_estimated' column exists and is bool
    if "depth_estimated" not in df.columns:
        df["depth_estimated"] = False
    df["depth_estimated"] = df["depth_estimated"].astype(bool)

    grouped = df.groupby(["location", "deployment"])["depth_estimated"].any().reset_index()

    selected_deployment = None

    with st.container(border=True, height=300):
        # For each location add an expander
        for location, group_df in grouped.groupby("location"):
            with st.expander(f":material/pin_drop: {location}", expanded=False):
                for _, row in group_df.iterrows():
                    col1, col2 = st.columns([1, 1], vertical_alignment="center")
                    with col1:
                        if st.button(f":material/sd_card: {row.deployment}", key=f"{location}_{row.deployment}", use_container_width=True):
                            selected_deployment = row.deployment
                    with col2:
                        st.markdown(f"{'depth estimated already' if row.depth_estimated else 'depth not yet estimated'}")

    return selected_deployment