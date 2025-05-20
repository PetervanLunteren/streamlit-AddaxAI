
import streamlit as st
import os
import sys
import subprocess
from backend.utils import AddaxAI_files




# run a parallel tkinter script to select a folder


def select_folder():
    result = subprocess.run([sys.executable, os.path.join(
        AddaxAI_files, "AddaxAI", "streamlit-AddaxAI", "frontend", "folder_selector.py")], capture_output=True, text=True)
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
            st.write('<span style="color: grey;"> None selected...</span>',
                     unsafe_allow_html=True)
    else:
        with col2:
            selected_folder_str_short = "..." + \
                selected_folder_str[-45:] if len(
                    selected_folder_str) > 45 else selected_folder_str
            st.markdown(
                f'Selected folder <code style="color:#086164; font-family:monospace;">{selected_folder_str_short}</code>', unsafe_allow_html=True)

    return selected_folder_str
