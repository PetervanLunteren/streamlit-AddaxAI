

# WAARWAS IK? ik was bezig om de main.py beter te maken, zodat niet altijd alles weer opnieuw hoeft te gaan bij iedere rerun. Er zitten vast nog wel bugs in. Dus beter even testen.
# als dat allemaal werkt, dan ga ik dit script opschonen, dan gewoon van het begin af naar het eind werken van de AI pipeline en alles klaar maken, dus taxon-mappings, env-requiremetns, etc.
# Goed bijhouden de TODO's en de issues, zodat ik niet vergeet wat ik nog moet doen.




# for later
# TODO: https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/classification_postprocessing.py
# TODO: https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
# TODO: RDE
# todo: put all golabl variables in st.sessionstate, like model_meta, txts, vars, map, etc.
# todo: make function that checks the model_meta online and adds the variables.json file to the model folder if it does not exist
# TODO: this must be offline, but I can only do that at the end when I know which icons I'm using.


# CLI to run it
# cd /Applications/AddaxAI_files/AddaxAI/streamlit-AddaxAI && conda activate env-streamlit-addaxai && streamlit run main.py >> assets/logs/log.txt 2>&1 &


# packages

import streamlit as st
import sys
import os
import time
import json
import platform
from appdirs import user_config_dir, user_cache_dir
import folium




# Force all output to be unbuffered and go to stdout
sys.stdout.reconfigure(line_buffering=True)
sys.stderr = sys.stdout  # Redirect stderr to stdout too


# ADDAXAI_FILES = st.session_state["shared"].get("ADDAXAI_FILES", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

if "shared" in st.session_state and "ADDAXAI_FILES" in st.session_state["shared"]:
    # If the shared state already has ADDAXAI_FILES, use it
    ADDAXAI_FILES = st.session_state["shared"]["ADDAXAI_FILES"]
else:
    # If not, set it to the default value
    ADDAXAI_FILES = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# if "ADDAXAI_FILES" not in st.session_state["shared"]:

# ADDAXAI_FILES = st.session_state.get("shared").get("ADDAXAI_FILES", ADDAXAI_FILES)

# make sure the config file is set
os.environ["STREAMLIT_CONFIG"] = os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI", ".streamlit", "config.toml") 

st.set_page_config(initial_sidebar_state="auto", page_icon=os.path.join(
    ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI", "assets", "images", "logo_square.png"), page_title="AddaxAI")


# from utils.config import ADDAXAI_FILES






st.write(st.session_state)


# init shared session state
# Initialize shared session state for cross-tool temporary variables
STARTUP_APP = False
if "shared" not in st.session_state:
    
    STARTUP_APP = True    
    
    
    CONFIG_DIR = user_config_dir("AddaxAI")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    TEMP_DIR = os.path.join(user_cache_dir("AddaxAI"), "temp")
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Initialize shared session state for cross-tool temporary variables
    if "shared" not in st.session_state:
        st.session_state["shared"] = {}
    
    def get_os_name():
        system = platform.system()
        if system == "Windows":
            return "windows"
        elif system == "Linux":
            return "linux"
        elif system == "Darwin":
            return "macos"

    OS_NAME = get_os_name()
    
    # ADDAXAI_FILES = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    ADDAXAI_FILES_ST = os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI") # this is only temporary, will be removed later
    MICROMAMBA = os.path.join(ADDAXAI_FILES_ST, "bin", OS_NAME, "micromamba")
    VIDEO_EXTENSIONS = ('.mp4','.avi','.mpeg','.mpg','.mov','.mkv')
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.gif', '.png', '.tif', '.tiff', '.bmp')
    MAP_FILE_PATH = os.path.join(CONFIG_DIR, "map.json")
    
    
    def log_function(msg):
        with open(os.path.join(ADDAXAI_FILES_ST, 'assets', 'logs', 'log.txt'), 'a') as f:
            f.write(f"{msg}\n")
        print(msg)
    
    
    st.session_state["shared"] = {
        "CONFIG_DIR": CONFIG_DIR,
        "TEMP_DIR": TEMP_DIR,
        "OS_NAME": OS_NAME,
        "ADDAXAI_FILES": ADDAXAI_FILES,
        "ADDAXAI_FILES_ST": ADDAXAI_FILES_ST,
        "MICROMAMBA": MICROMAMBA,
        "VIDEO_EXTENSIONS": VIDEO_EXTENSIONS,
        "IMG_EXTENSIONS": IMG_EXTENSIONS,
        "log": log_function,
        "MAP_FILE_PATH": MAP_FILE_PATH
    }

else:
    STARTUP_APP = False
    CONFIG_DIR = st.session_state["shared"]["CONFIG_DIR"]
    TEMP_DIR = st.session_state["shared"]["TEMP_DIR"]
    OS_NAME = st.session_state["shared"]["OS_NAME"]
    ADDAXAI_FILES = st.session_state["shared"]["ADDAXAI_FILES"]
    ADDAXAI_FILES_ST = st.session_state["shared"]["ADDAXAI_FILES_ST"]
    MICROMAMBA = st.session_state["shared"]["MICROMAMBA"]
    VIDEO_EXTENSIONS = st.session_state["shared"]["VIDEO_EXTENSIONS"]
    IMG_EXTENSIONS = st.session_state["shared"]["IMG_EXTENSIONS"]
    MAP_FILE_PATH = st.session_state["shared"]["MAP_FILE_PATH"]







from utils.common import load_lang_txts, load_vars, update_vars
from utils.analyse_advanced import load_known_projects, load_model_metadata
from utils.common import print_widget_label

general_settings_vars = load_vars(section = "general_settings")
lang = general_settings_vars["lang"]
mode = general_settings_vars["mode"]

# only do this when the app starts up, not on every rerun
if STARTUP_APP:
    
    
    # import platform
    # from appdirs import user_config_dir, user_cache_dir
    
    status_placeholder = st.empty()
    status_placeholder.status("Loading AddaxAI Streamlit app...")
    time.sleep(2)
    
    # remove the status placeholder
    status_placeholder.empty()
    







    # CONFIG_DIR = user_config_dir("AddaxAI")
    # os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # TEMP_DIR = os.path.join(user_cache_dir("AddaxAI"), "temp")
    # os.makedirs(TEMP_DIR, exist_ok=True)

    # # Initialize shared session state for cross-tool temporary variables
    # if "shared" not in st.session_state:
    #     st.session_state["shared"] = {}
    
    # def get_os_name():
    #     system = platform.system()
    #     if system == "Windows":
    #         return "windows"
    #     elif system == "Linux":
    #         return "linux"
    #     elif system == "Darwin":
    #         return "macos"

    # OS_NAME = get_os_name()
    
    # ADDAXAI_FILES = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    # ADDAXAI_FILES_ST = os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI") # this is only temporary, will be removed later
    # MICROMAMBA = os.path.join(ADDAXAI_FILES_ST, "bin", OS_NAME, "micromamba")
    # VIDEO_EXTENSIONS = ('.mp4','.avi','.mpeg','.mpg','.mov','.mkv')
    # IMG_EXTENSIONS = ('.jpg', '.jpeg', '.gif', '.png', '.tif', '.tiff', '.bmp')
    # MAP_FILE_PATH = os.path.join(CONFIG_DIR, "map.json")
    
    
    # def log_function(msg):
    #     with open(os.path.join(ADDAXAI_FILES_ST, 'assets', 'logs', 'log.txt'), 'a') as f:
    #         f.write(f"{msg}\n")
    #     print(msg)
    
    
    # # populate the shared session state with some default values
    # st.session_state["shared"]["CONFIG_DIR"] = CONFIG_DIR
    # st.session_state["shared"]["TEMP_DIR"] = TEMP_DIR
    # st.session_state["shared"]["OS_NAME"] = OS_NAME
    # st.session_state["shared"]["ADDAXAI_FILES"] = ADDAXAI_FILES
    # st.session_state["shared"]["ADDAXAI_FILES_ST"] = ADDAXAI_FILES_ST
    # st.session_state["shared"]["MICROMAMBA"] = MICROMAMBA
    # st.session_state["shared"]["VIDEO_EXTENSIONS"] = VIDEO_EXTENSIONS
    # st.session_state["shared"]["IMG_EXTENSIONS"] = IMG_EXTENSIONS
    # st.session_state["shared"]["log"] = log_function
    # st.session_state["shared"]["MAP_FILE_PATH"] = MAP_FILE_PATH
    
    
    
    
    # this is where the overall settings are stored
    # these will remain constant over different versions of the app
    # so think of language settings, mode settings, etc.
    # locations per camera, paths to deployments, etc.
    map_file = os.path.join(CONFIG_DIR, "map.json") 
    if not os.path.exists(map_file):

        # start with a clean slate
        map = {
            "projects": {}
        }

        with open(map_file, "w") as f:
            json.dump(map, f, indent=2)

    general_settings_file = os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI", "vars", f"general_settings.json")
    if not os.path.exists(general_settings_file):
        
        # make directory if it does not exist
        os.makedirs(os.path.dirname(general_settings_file), exist_ok=True)
        
        # create a default settings file
        general_settings = {
            "lang": "en",
            "mode": 1,  # 0: simple mode, 1: advanced mode
            "selected_projectID": None
        }
        with open(general_settings_file, "w") as f:
            json.dump(general_settings, f, indent=2)





    # Load language texts into session state on startup only
    if not st.session_state.get("txts"):
        full_txts = load_lang_txts()
        # Store only the current language's texts in flattened structure
        st.session_state["txts"] = {key: value[lang] for key, value in full_txts.items()}

    # Load model metadata into session state on startup only
    if not st.session_state.get("model_meta"):
        st.session_state["model_meta"] = load_model_metadata()



    # render a dummy map with a marker so that the rest of the markers this session will be rendered
    m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
    folium.Marker([39.949610, -75.150282]).add_to(m)
    
    
# Load custom CSS from external file
with open(os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI", "assets", "css", "styles.css"), "r") as f:
    css_content = f.read()
st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# Inject Material Icons for the header stepper bar
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
""", unsafe_allow_html=True)

# else:
#     # if the session state is not empty, we assume that the app is already running
#     # and we can use the existing session state
#     CONFIG_DIR = st.session_state["shared"]["CONFIG_DIR"]
    

# only import the modules after startup
# from utils.config import ADDAXAI_FILES




# save only the last 1000 lines of the log file
# log_fpath = "/Users/peter/Desktop/streamlit_app/frontend/streamlit_log.txt"
log_fpath = os.path.join(ADDAXAI_FILES, "AddaxAI",
                         "streamlit-AddaxAI", "assets", "logs", "log.txt")
if not os.path.exists(log_fpath):
    # create an empty log file
    with open(log_fpath, "w", encoding="utf-8") as file:
        file.write("AddaxAI Streamlit App Log\n")
        file.write("========================================\n")
        file.write("This is the log file for the AddaxAI Streamlit app.\n")
        file.write("It will contain the last 1000 lines of the log.\n")
        file.write("========================================\n\n")  
with open(log_fpath, "r", encoding="utf-8") as file:
    log = file.readlines()
    if len(log) > 1000:
        log = log[-1000:]
with open(log_fpath, "w", encoding="utf-8") as file:
    file.writelines(log)
try:  # DEBUG there seems to be a problem with the permissions here
    with open(log_fpath, "r", encoding="utf-8") as file:
        log = file.readlines()
    if len(log) > 1000:
        log = log[-1000:]
    with open(log_fpath, "w", encoding="utf-8") as file:
        file.writelines(log)
except PermissionError:
    print(
        f"Permission denied when accessing {log_fpath}. Could not trim log file.")
except FileNotFoundError:
    print(f"Log file {log_fpath} not found.")
except Exception as e:
    print(f"Unexpected error: {e}")




# Use session state texts and model metadata
txts = st.session_state["txts"]
model_meta = st.session_state["model_meta"]

# page navigation
# st.logo("/Users/peter/Desktop/streamlit_app/frontend/logo.png", size = "large")
st.logo(os.path.join(ADDAXAI_FILES, "AddaxAI",
        "streamlit-AddaxAI", "assets", "images", "logo.png"), size="large")
if mode == 0:  # simple mode
    analyse_sim_page = st.Page(
        os.path.join("tools", "analyse_simple.py"), title=txts["analyse_txt"], icon=":material/rocket_launch:")
    pg = st.navigation([analyse_sim_page])
elif mode == 1:  # advanced mode
    analyse_adv_page = st.Page(
        os.path.join("tools", "analyse_advanced.py"), title=txts["analyse_txt"], icon=":material/add:")
    repeat_detection_elimination_page = st.Page(
        os.path.join("tools", "repeat_detection_elimination.py"), title="Repeat detection elimination", icon=":material/reset_image:")
    verify_page = st.Page(
        os.path.join("tools", "verify.py"), title="Human verification", icon=":material/checklist:")
    depth_estimation_page = st.Page(
        os.path.join("tools", "depth_estimation.py"), title="Depth estimation", icon=":material/square_foot:")
    explore_page = st.Page(
        os.path.join("tools", "explore.py"), title="Explore results", icon=":material/bar_chart_4_bars:")
    postprocess_page = st.Page(
        os.path.join("tools", "postprocess.py"), title=txts["postprocess_txt"], icon=":material/stylus_note:")
    camera_management_page = st.Page(
        os.path.join("tools", "camera_management.py"), title="Metadata management", icon=":material/photo_camera:")
    settings_page = st.Page(
        os.path.join("tools", "settings.py"), title=txts["settings_txt"], icon=":material/settings:")
    pg = st.navigation([analyse_adv_page, repeat_detection_elimination_page, verify_page,
                       depth_estimation_page, explore_page, postprocess_page, camera_management_page, settings_page])
pg.run()

# mode settings
mode_options = {
    0: "Simple",
    1: "Advanced",
}


def on_mode_change():
    # Only update persistent storage, no session state needed
    if "mode_selection" in st.session_state:
        mode_selection = st.session_state["mode_selection"]
        update_vars("general_settings", {
            "mode": mode_selection
        })


print_widget_label("Mode", help_text="help text", sidebar=True)
# Use persistent value directly
mode_selected = st.sidebar.segmented_control(
    "Mode",
    options=mode_options.keys(),
    format_func=mode_options.get,
    selection_mode="single",
    label_visibility="collapsed",
    help=txts["mode_explanation_txt"],
    key="mode_selection",
    on_change=on_mode_change,
    default=mode)

# No session state cleanup needed - only persistent storage used


def on_project_change():
    # Only update persistent storage, no session state needed
    if "project_selection_sidebar" in st.session_state:
        project_selection = st.session_state["project_selection_sidebar"]
        update_vars("general_settings", {
            "selected_projectID": project_selection
        })

if mode == 1:  # advanced mode

    # check what is already known and selected
    projects, selected_projectID = load_known_projects()

    # if first project, show only button and no dropdown
    if not projects == {}:
        
        options = list(projects.keys())
        # Use persistent value directly
        selected_index = options.index(selected_projectID) if selected_projectID in options else 0

        # overwrite selected_projectID if user has selected a different project
        print_widget_label("Project", help_text="The project selected here is the one that all tools will work with. If you have a new project, you can add it at + add deployment when you want to process the first batch data.", sidebar=True)
        selected_projectID = st.sidebar.selectbox(
            "Project",
            options=options,
            index=selected_index,
            label_visibility="collapsed",
            key="project_selection_sidebar",
            on_change=on_project_change
        )
        
        # No session state cleanup needed - only persistent storage used