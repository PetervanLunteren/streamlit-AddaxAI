
import streamlit as st
import os
import sys
import subprocess
from backend.utils import AddaxAI_files

def class_selector(classes, preselected):
        
    # select all or none buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button(":material/check_box: Select all", use_container_width=True):
            for species in classes:
                st.session_state[f"species_{species}"] = True
    with col2:
        if st.button(":material/check_box_outline_blank: Select none", use_container_width=True):
            for species in classes:
                st.session_state[f"species_{species}"] = False

    # checkboxes in a scrollable container
    selected_species = []
    with st.container(border=True, height=300):
        for species in classes:
            key = f"species_{species}"
            checked = st.session_state.get(key, True if species in preselected else False)
            if st.checkbox(species, value=checked, key=key):
                selected_species.append(species)

    # log selected species
    st.write(f"You selected the presence of {len(selected_species)} classes.") 

    # return list
    return selected_species

# run a parallel tkinter script to select a folder
def select_folder():
    result = subprocess.run([sys.executable, os.path.join(AddaxAI_files, "AddaxAI", "streamlit-AddaxAI", "frontend", "folder_selector.py")], capture_output=True, text=True)
    folder_path = result.stdout.strip()
    if folder_path != "" and result.returncode == 0:
        return folder_path
    else:
        return None

def browse_directory_widget(selected_folder_str):
    col1, col2 = st.columns([1, 3], vertical_alignment="center")
    with col1:
        if st.button(":material/folder: Browse", key="folder_select_button", use_container_width=True):
            selected_folder_str = select_folder()
    if not selected_folder_str:
        with col2:
            st.write('<span style="color: grey;"> None selected...</span>', unsafe_allow_html=True)
    else:
        with col2:
            selected_folder_str_short = "..." + selected_folder_str[-45:] if len(selected_folder_str) > 45 else selected_folder_str
            st.markdown(f'Selected folder <code style="color:#086164; font-family:monospace;">{selected_folder_str_short}</code>', unsafe_allow_html=True)
    
    return selected_folder_str

