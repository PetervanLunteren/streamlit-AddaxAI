# this file is used to define global variables that are used across the project

import os
import platform

def get_os_name():
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Linux":
        return "linux"
    elif system == "Darwin":
        return "macos"

os_name = get_os_name()


ADDAXAI_FILES = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
ADDAXAI_FILES_ST = os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI") # this is only temporary, will be removed later
MICROMAMBA = os.path.join(ADDAXAI_FILES_ST, "bin", os_name, "micromamba")
VIDEO_EXTENSIONS = ('.mp4','.avi','.mpeg','.mpg','.mov','.mkv')
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.gif', '.png', '.tif', '.tiff', '.bmp')
