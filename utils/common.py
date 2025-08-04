
import os
import json
import streamlit as st
from appdirs import user_config_dir
import string
import re
import random
from datetime import datetime
import requests


from utils.config import *

# MAP_FILE_PATH will be set from session state when available
# This prevents errors when importing outside of Streamlit context
try:
    MAP_FILE_PATH = st.session_state["shared"]["MAP_FILE_PATH"]
except (KeyError, AttributeError):
    MAP_FILE_PATH = None

# set versions
with open(os.path.join(ADDAXAI_FILES_ST, 'assets', 'version.txt'), 'r') as file:
    
    current_AA_version = file.read().strip()

def unique_animal_string():
    
    # 5-letter intensifying adverbs
    adverbs = [
        "quite", "ultra", "super", "fully", "truly", "awful", "sheer"]
    
    # 5-letter adjectives  
    adjectives = [
        "quick", "brave", "smart", "swift", "alert", "sharp", "quiet",
        "fresh", "happy", "light", "clean", "clear", "close", "sappy", 
        "dense", "fancy", "giant", "green", "heavy", "tough", "vivid", 
        "young", "sound", "sleek", "proud", "noble", "loyal", "fleet", 
        "agile", "shiny", "furry", "scaly", "rough", "daffy", "dicey",
        "solid", "fluid", "misty", "foggy", "sunny", "windy", "proof", 
        "sandy", "rocky", "muddy", "snowy", "leafy", "woody", "sarky",
        "mossy", "sweet", "spicy", "salty", "tangy", "crisp", "poppy",
        "alive", "lucky", "cheap", "false", "rapid", "fifth", "awful",
        "plain", "still", "round", "silly", "above", "blind", "dirty",
        "sixth", "loose", "level", "outer", "gross", "acute", "valid",
        "tight", "exact", "sheer", "naked", "spare", "nasty", "crazy",
        "magic", "upset", "super", "steep", "harsh", "naval", "vague",
        "faint", "rigid", "stiff", "eager", "fatal", "rival", "blank",
        "cruel", "crude", "solar", "toxic", "awake", "tenth", "novel",
        "weird", "civic", "noisy", "ninth", "alien", "handy", "tense",
        "grave", "bleak", "comic", "ample", "prone", "petty", "naive",
        "vocal", "dusty", "penal", "polar", "risky", "weary", "olive",
        "stark", "audio", "overt", "frank", "sober", "frail", "basal",
        "moist", "split", "adult", "focal", "merry", "privy", "undue",
        "fatty", "renal", "brisk", "utter", "motor", "fiery", "plump",
        "jolly", "shaky", "blunt", "tidal", "bland", "cubic", "pagan",
        "witty", "micro", "stern", "curly", "blond", "hairy", "stale",
        "viral", "dizzy", "hardy", "token", "tasty", "unfit", "stout",
        "pious", "fetal", "messy", "rusty", "heady", "hefty", "bogus",
        "bulky", "stray", "hasty", "stony", "roast", "dodgy", "regal",
        "molar", "eerie", "inert", "tacit", "queer", "milky", "macho",
        "timid", "alike", "scant", "lofty", "shady", "nasal", "amber",
        "lunar", "flash", "erect", "needy", "adept", "silky", "rainy",
        "baggy", "murky", "lowly", "cross", "aloof", "lousy", "manic",
        "slack", "scary", "tawny", "tonal", "moody", "humid", "lucid",
        "beige", "smoky", "slick", "ileal", "husky", "gaunt", "fussy",
        "juicy", "sonic", "ionic", "fuzzy", "modal", "mauve", "ducal",
        "inept", "lumpy", "lurid", "spiky", "awash", "burly", "soggy",
        "lyric", "freak", "ruddy", "bumpy", "gaudy", "banal", "obese",
        "grimy", "slimy", "khaki", "brash", "squat", "nutty", "plush",
        "axial", "dingy", "hilly", "canny", "bushy", "macro", "soapy",
        "batty", "after", "funky", "musty", "jumbo", "seedy", "taboo",
        "gruff", "manly", "jaded", "livid", "tacky", "bonny", "hazel",
        "sooty", "colic", "cocky", "lithe", "acrid", "bossy", "fishy",
        "comfy", "feral", "lilac", "rowdy", "wacky", "crass", "fizzy",
        "weedy", "giddy", "loath", "tatty", "aural", "unlit", "godly",
        "algal", "roomy", "peaty", "kinky", "saucy", "cream", "puffy",
        "dusky", "surly", "pushy", "brawn", "barmy", "jerky", "flush",
        "sulky", "wispy", "runny", "trite", "afoot", "nosey", "mucky",
        "spiny", "loony", "horny", "tepid", "meaty", "dotty", "lanky",
        "leaky", "gutsy", "itchy", "ashen", "rabid", "elite", "balmy",
        "bawdy", "randy", "suave", "waste", "boggy", "dowdy", "pithy",
        "dumpy", "mushy", "tipsy", "lusty", "jumpy", "showy", "askew",
        "nervy", "inane", "epoxy", "snide", "leggy", "zonal", "nifty",
        "areal", "perky", "beefy", "tinny", "prize", "butch", "corny",
        "cagey", "natal", "droll", "beady", "skint", "hunky", "downy",
        "crack", "soppy", "buxom", "ochre", "folic", "hoary", "gusty",
        "oleic", "potty", "avian", "retro", "fetid", "flaky", "nippy",
        "wonky", "teeny", "yummy", "swell", "ritzy", "tubby", "jazzy",
        "gooey", "brill", "glial", "jokey", "boozy", "reedy", "stoic",
        "aglow", "musky", "nodal", "wordy", "natty", "velar", "hyper",
        "silty", "podgy", "tardy", "ziggy", "mousy", "chewy", "elven",
        "ratty", "loopy", "pygmy", "wimpy", "model", "chill", "peaky",
        "goofy", "warty", "bendy", "filmy", "jammy", "bandy", "waxen",
        "lytic", "dicky", "tarty", "dishy", "bitty", "venal", "dippy",
        "pudgy", "muggy", "cushy", "corky", "yucky", "pasty", "humic",
        "footy", "muzzy", "ducky", "gawky", "mangy", "elect", "weeny",
        "quasi", "picky", "ropey", "boney", "sassy", "plumb", "weepy",
        "mealy", "vagal", "sable", "dopey", "rangy", "bluey", "class",
        "seely", "porky", "vapid", "chary", "curvy", "horsy", "tubal",
        "punky", "ovoid", "tonic", "blase", "nitty", "zippy", "gummy",
        "afire", "beaky", "beery", "busty", "holey", "gauze", "huggy",
        "gassy", "faddy", "phony", "pupal", "kooky", "ludic", "caped",
        "unwed", "leery", "wormy", "ferny", "minty", "girly", "yogic",
        "pukka", "foamy", "fusty", "rogue", "seamy", "huffy", "passe"
    ]
    
    # 5-letter animals
    animals = [
        "addax", "aguti", "ammon", "ariel", "bison", "bitch", "bobac", 
        "bobak", "bongo", "brock", "bruin", "burro", "camel", "canis", 
        "chimp", "chiru", "civet", "coati", "coney", "coypu", "crone", 
        "cuddy", "daman", "dhole", "dingo", "dogie", "drill", "eland", 
        "equus", "felis", "filly", "fitch", "fossa", "gayal", "genet", 
        "goral", "grice", "gryce", "hinny", "hippo", "horse", "hutia", 
        "hyena", "hyrax", "indri", "izard", "jocko", "kaama", "kiang", 
        "koala", "kulan", "kyloe", "lemur", "liger", "llama", "loris", 
        "magot", "manis", "manul", "mhorr", "moose", "morse", "mouse", 
        "nagor", "nyala", "okapi", "orang", "oribi", "otary", "otter", 
        "ounce", "panda", "pekan", "phoca", "pongo", "potto", "puppy", 
        "ratel", "rhino", "royal", "sable", "saiga", "sajou", "sasin", 
        "serow", "sheep", "shoat", "shote", "shrew", "skunk", "sloth", 
        "sorel", "spade", "spado", "steer", "stirk", "stoat", "swine", 
        "tabby", "takin", "tapir", "tatou", "tiger", "tigon", "urial", 
        "urson", "vison", "vixen", "whale", "whelp", "yapok", "zebra"
    ]
    
    # Randomly select one from each list
    adverb = random.choice(adverbs)
    animal = random.choice(animals)
    adjective = random.choice(adjectives)
    
    # random suffix of letters and numbers in captial letters
    suffix = ''.join([random.choice(string.ascii_uppercase + string.digits) for _ in range(5)])
    return f"{adverb}-{adjective}-{animal}-{suffix}"



def default_converter(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(
        f"Object of type {obj.__class__.__name__} is not JSON serializable")


def clear_vars(section):
    """
    Clear only temporary variables in session state for a specific section.
    This preserves persistent variables like process_queue.
    """
    if section in st.session_state:
        st.session_state[section] = {}


def init_session_state(section):
    """
    Initialize session state for a specific section if it doesn't exist.
    """
    if section not in st.session_state:
        st.session_state[section] = {}


def get_session_var(section, var_name, default=None):
    """
    Get a variable from session state for a specific section.
    """
    init_session_state(section)
    return st.session_state[section].get(var_name, default)


def set_session_var(section, var_name, value):
    """
    Set a variable in session state for a specific section.
    """
    init_session_state(section)
    st.session_state[section][var_name] = value


def update_session_vars(section, updates):
    """
    Update multiple variables in session state for a specific section.
    """
    init_session_state(section)
    st.session_state[section].update(updates)


def replace_vars(section, new_vars):
    vars_file = os.path.join(ADDAXAI_FILES, "AddaxAI",
                             "streamlit-AddaxAI", "config", f"{section}.json")

    # Overwrite with only the new updates
    with open(vars_file, "w", encoding="utf-8") as file:
        json.dump(new_vars, file, indent=2, default=default_converter)


def update_vars(section, updates):
    vars_file = os.path.join(ADDAXAI_FILES, "AddaxAI",
                             "streamlit-AddaxAI", "config", f"{section}.json")
    if not os.path.exists(vars_file):
        with open(vars_file, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

    # read section vars
    with open(vars_file, "r", encoding="utf-8") as f:
        section_vars = json.load(f)

    # update
    section_vars.update(updates)

    # Use `default=default_converter` to catch any lingering datetime objects
    with open(vars_file, "w") as file:
        json.dump(section_vars, file, indent=2, default=default_converter)


def load_map():
    """Reads the data from the JSON file and returns it as a dictionary."""
    global MAP_FILE_PATH
    
    # Get MAP_FILE_PATH from session state if not already set
    if MAP_FILE_PATH is None:
        try:
            MAP_FILE_PATH = st.session_state["shared"]["MAP_FILE_PATH"]
        except (KeyError, AttributeError):
            # Fallback path if session state not available
            from appdirs import user_config_dir
            MAP_FILE_PATH = os.path.join(user_config_dir("AddaxAI"), "map.json")

    # Load full settings or initialize
    try:
        if os.path.exists(MAP_FILE_PATH):
            with open(MAP_FILE_PATH, "r", encoding="utf-8") as f:
                settings = json.load(f)
        else:
            settings = {}
    except (json.JSONDecodeError, IOError):
        settings = {}

    return settings, MAP_FILE_PATH


def load_vars(section):
    # if not exist, create empty vars file
    vars_file = os.path.join(ADDAXAI_FILES, "AddaxAI",
                             "streamlit-AddaxAI", "config", f"{section}.json")
    if not os.path.exists(vars_file):
        with open(vars_file, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

    # read section vars
    with open(vars_file, "r", encoding="utf-8") as f:
        section_vars = json.load(f)

    return section_vars
    # return {var: section_vars.get(var, None) for var in requested_vars}.values()


def load_lang_txts():
    txts_fpath = os.path.join(ADDAXAI_FILES, "AddaxAI",
                              "streamlit-AddaxAI", "assets", "language", "lang.json")
    with open(txts_fpath, "r", encoding="utf-8") as file:
        txts = json.load(file)
    return txts

#########################
### GENERAL UTILITIES ###
#########################

# UNUSED FUNCTION - Vulture detected unused function
# def multiselect_checkboxes(classes, preselected):
# 
#     # select all or none buttons
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button(":material/check_box: Select all", use_container_width=True):
#             for species in classes:
#                 st.session_state[f"species_{species}"] = True
#     with col2:
#         if st.button(":material/check_box_outline_blank: Select none", use_container_width=True):
#             for species in classes:
#                 st.session_state[f"species_{species}"] = False
# 
#     # checkboxes in a scrollable container
#     selected_species = []
#     with st.container(border=True, height=300):
#         for species in classes:
#             key = f"species_{species}"
#             checked = st.session_state.get(
#                 key, True if species in preselected else False)
#             if st.checkbox(species, value=checked, key=key):
#                 selected_species.append(species)
# 
#     # log selected species
#     st.markdown(
#         f'&nbsp; You selected the presence of <code style="color:#086164; font-family:monospace;">{len(selected_species)}</code> classes', unsafe_allow_html=True)
# 
#     # return list
#     return selected_species






# check if the user needs an update
# UNUSED FUNCTION - Vulture detected unused function
# def requires_addaxai_update(required_version):
#     current_parts = list(map(int, current_AA_version.split('.')))
#     required_parts = list(map(int, required_version.split('.')))
# 
#     # Pad the shorter version with zeros
#     while len(current_parts) < len(required_parts):
#         current_parts.append(0)
#     while len(required_parts) < len(current_parts):
#         required_parts.append(0)
# 
#     # Compare each part of the version
#     for current, required in zip(current_parts, required_parts):
#         if current < required:
#             return True  # current_version is lower than required_version
#         elif current > required:
#             return False  # current_version is higher than required_version
# 
#     # All parts are equal, consider versions equal
#     return False



# import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL METADATA MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_latest_model_info():
    """
    Downloads the latest model metadata from GitHub and creates folder structure
    for new models. Shows notifications for any new models found.
    
    This function:
    1. Downloads model_meta.json from the GitHub repository
    2. Compares models in metadata vs existing model directories
    3. Creates folders and variables.json for any new models
    4. Shows info_box notifications with model info popover for new models
    
    Called during app startup to keep model metadata current.
    """
    import streamlit as st
    
    from utils.config import log
    log(f"EXECUTED: fetch_latest_model_info()")
    
    # Define paths using the global ADDAXAI_FILES_ST from config
    model_meta_url = "https://raw.githubusercontent.com/PetervanLunteren/streamlit-AddaxAI/refs/heads/main/assets/model_meta/model_meta.json"
    model_meta_local = os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI", "assets", "model_meta", "model_meta.json")
    models_dir = os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI", "models")
    
    try:
        # Download latest model metadata with reasonable timeout
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Accept-Encoding": "*",
            "Connection": "keep-alive"
        }
        
        response = requests.get(model_meta_url, timeout=10, headers=headers)
        
        if response.status_code == 200:
            # Save updated model metadata
            os.makedirs(os.path.dirname(model_meta_local), exist_ok=True)
            with open(model_meta_local, 'wb') as file:
                file.write(response.content)
            
            log("Updated model_meta.json successfully.")
            
            # Load the downloaded metadata
            with open(model_meta_local, 'r') as f:
                model_meta = json.load(f)
            
            # Check if this is first startup (no models exist at all)
            is_first_startup = True
            for model_type in ["det", "cls"]:
                type_dir = os.path.join(models_dir, model_type)
                if os.path.exists(type_dir):
                    existing_models = [d for d in os.listdir(type_dir) 
                                     if os.path.isdir(os.path.join(type_dir, d))]
                    if existing_models:
                        is_first_startup = False
                        break
            
            if is_first_startup:
                log("First startup detected - creating all model directories without notifications")
            
            # Process both detection and classification models
            for model_type in ["det", "cls"]:
                if model_type in model_meta:
                    model_dicts = model_meta[model_type]
                    all_models = list(model_dicts.keys())
                    
                    # Get existing model directories
                    type_dir = os.path.join(models_dir, model_type)
                    os.makedirs(type_dir, exist_ok=True)
                    
                    existing_models = []
                    if os.path.exists(type_dir):
                        existing_models = [d for d in os.listdir(type_dir) 
                                         if os.path.isdir(os.path.join(type_dir, d))]
                    
                    # Find new models (in metadata but not in filesystem)
                    new_models = [model for model in all_models if model not in existing_models]
                    
                    # Create directories for new models
                    for model_id in new_models:
                        model_info = model_dicts[model_id]
                        
                        # Create model directory
                        model_dir = os.path.join(type_dir, model_id)
                        os.makedirs(model_dir, exist_ok=True)
                        
                        # Create variables.json file with model metadata
                        variables_file = os.path.join(model_dir, "variables.json")
                        with open(variables_file, 'w') as f:
                            json.dump(model_info, f, indent=4)
                        
                        log(f"Created directory and variables.json for new {model_type} model: {model_id}")
                        
                        # Show notification for new model (only if not first startup)
                        if not is_first_startup:
                            friendly_name = model_info.get('friendly_name', model_id)
                            model_type_name = "species identification model" if model_type == "cls" else "detection"
                            
                            # Create info box
                            info_box(
                                msg = f"New {model_type_name.lower()} model available: {friendly_name}",
                                title="New model added",
                                icon=":material/new_releases:"
                            )
                                

            
        else:
            log(f"Failed to download model metadata. Status code: {response.status_code}")
            
    except requests.exceptions.Timeout:
        log("Request timed out. Model metadata update stopped.")
    except Exception as e:
        log(f"Could not update model metadata: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK ERROR LOGGING WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def logged_callback(func):
    """
    Decorator to wrap Streamlit callbacks with error logging.
    
    This ensures that any exceptions in callbacks are logged to the file
    before Streamlit catches and displays them in the UI.
    
    Usage:
        @logged_callback
        def on_button_click():
            # Your callback code here
            pass
    """
    import functools
    import traceback
    from utils.config import log
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log(f"ERROR in callback {func.__name__}: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            # Re-raise so Streamlit still shows the error in UI
            raise
    
    return wrapper
