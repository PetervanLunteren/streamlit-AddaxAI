import streamlit as st
from . import camera_management, depth_estimation

def main():
    st.title("Settings")

    # Add a sidebar for tool selection
    st.sidebar.title("Tools")
    tool_selection = st.sidebar.radio("Go to", ["Camera Management", "Depth Estimation"])

    if tool_selection == "Camera Management":
        camera_management.main()
    elif tool_selection == "Depth Estimation":
        depth_estimation.main()