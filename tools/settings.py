
import streamlit as st
import sys
import platform

from utils.common import *



# load language settings
txts = load_lang_txts()
# settings, _ = load_map()

general_settings_vars = load_vars(section = "general_settings")
lang = general_settings_vars["lang"]
mode = general_settings_vars["mode"]

lang_options = txts["languages"]
lang_idx = list(lang_options.keys()).index(lang)
st.subheader(":material/language: Language", divider="grey")
lang_selected = st.selectbox(
    "Language",
    options=lang_options.keys(),
    format_func=lambda option: lang_options[option],
    index=lang_idx,
    label_visibility="collapsed"
)


# start from scratch debug button
if st.button("Start from scratch", use_container_width=True, type="primary"):
    
    # remove all vars files in the vars directory
    vars_dir = os.path.join(ADDAXAI_FILES, "AddaxAI", "streamlit-AddaxAI", "vars")
    for filename in os.listdir(vars_dir):
        file_path = os.path.join(vars_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        
    # remove the map file
    map, MAP_FILE_PATH = load_map()
    if os.path.exists(MAP_FILE_PATH):
        os.remove(MAP_FILE_PATH)
    
    # reset session state
    st.session_state.clear()
    
    st.rerun()



# import settings file
map, MAP_FILE_PATH = load_map()

st.write("map_file:", MAP_FILE_PATH)
if st.button("Open map.json"):
    if platform.system() == "Darwin":  # macOS
        os.system(f'open "{MAP_FILE_PATH}"')
    else:
        st.warning("This action is only supported on macOS.")
with st.expander("Current settings", expanded=False):
    st.write("map:", map)
with st.expander("st.session_state", expanded=False):
    st.write(st.session_state)

save_btn = st.button(":material/save: Save settings", use_container_width=True,
                        type = "primary", key = "language_btn")
if save_btn:
    # save_global_vars({"lang": lang_selected})
    set_session_var("shared", "lang", lang_selected)
    update_vars("general_settings", {
        "lang": lang_selected
    })
    st.rerun()
    
    
# # logs
# st.subheader(":material/description: Logs", divider="grey")
# st.write("Here, you can export the application logs to review errors, warnings, and other messages, which can be helpful when sharing details with the development team.")
# log_fpath = "/Users/peter/Desktop/streamlit_app/frontend/streamlit_log.txt"
# import time
# print(f"\n\nExported Log file '{log_fpath}' on {time.strftime('%Y-%m-%d %H:%M:%S')}")
# sys.stdout.flush()
# with open(log_fpath, "r") as file:
#     lines = file.readlines()[-1000:]
# file_data = "".join(lines)
# st.download_button(
#     label=":material/save: Export logs",
#     data=file_data,
#     file_name="streamlit_log.txt",
#     mime="text/plain"
# )
# st.text("")



# # Save settings
# save_btn = st.button(":material/save: Save settings", use_container_width=True,
#                         type = "primary")
# if save_btn:
#     save_vars({"mode": mode_selected, "lang": lang_selected})
#     st.rerun()