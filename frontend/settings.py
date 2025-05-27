
import streamlit as st
import sys

from backend.utils import *

# load language settings
txts = load_txts()
settings, _ = load_settings()
lang = settings["lang"]
mode = settings["mode"]


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


# import settings file
settings_, settings_file = load_settings()

st.write("settings_file:", settings_file)
st.write("settings:", settings_)


# st.write("settings_file: /Users/peter/Library/Application Support/AddaxAI/settings.json")



save_btn = st.button(":material/save: Save settings", use_container_width=True,
                        type = "primary", key = "language_btn")
if save_btn:
    save_global_vars({"lang": lang_selected})
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