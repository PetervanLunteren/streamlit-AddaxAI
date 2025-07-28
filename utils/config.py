# this file is used to define global variables that are used across the project

import os
import platform
from appdirs import user_cache_dir, user_config_dir

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
TEMP_DIR = os.path.join(user_cache_dir("AddaxAI"), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


CLS_DIR = os.path.join(ADDAXAI_FILES, "models", "cls")
DET_DIR = os.path.join(ADDAXAI_FILES, "models", "det")

# load camera IDs
CONFIG_DIR = user_config_dir("AddaxAI")
os.makedirs(CONFIG_DIR, exist_ok=True)

def log(msg):
    with open(os.path.join(ADDAXAI_FILES_ST, 'assets', 'logs', 'log.txt'), 'a') as f:
        f.write(f"{msg}\n")
    print(msg)
        
    