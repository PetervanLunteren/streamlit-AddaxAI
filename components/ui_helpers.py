"""
UI helper functions for AddaxAI Streamlit application.
"""

import streamlit as st
from st_flexible_callout_elements import flexible_callout


def print_widget_label(label_text, icon=None, help_text=None, sidebar=False):
    """
    Print a formatted widget label with optional icon and help text.
    
    Args:
        label_text: The text to display
        icon: Optional material icon name (without 'material/' prefix)
        help_text: Optional help text tooltip
        sidebar: If True, displays in sidebar with smaller text
    """
    if icon:
        line = f":material/{icon}: &nbsp; "
    else:
        line = ""
        
    if sidebar:
        st.sidebar.markdown(f"<small>{line}<b>{label_text}</b></small>", unsafe_allow_html=True, help=help_text)
    else:
        st.markdown(f"{line}**{label_text}**", help=help_text)


def info_box(msg, title=None, icon=":material/info:"):
    """
    Display an informational callout box.
    
    Args:
        msg: The message to display
        title: Optional title (will be bold)
        icon: Icon to display (default: info icon)
    """
    if title:
        msg = f'<span style="font-weight: bold;">{title}</span><br>{msg}'
    
    flexible_callout(msg,
                     icon=icon,
                     background_color="#d9e3e7af",
                     font_color="#086164",
                     icon_size=23)


def warning_box(msg, title=None, icon=":material/warning:"):
    """
    Display a warning callout box.
    
    Args:
        msg: The message to display
        title: Optional title (will be bold)
        icon: Icon to display (default: warning icon)
    """
    if title:
        msg = f'<span style="font-weight: bold;">{title}</span><br>{msg}'
    
    flexible_callout(msg,
                     icon=icon,
                     background_color="#fffbeb",
                     font_color="#936b0c",
                     icon_size=23)


def radio_buttons_with_captions(option_caption_dict, key, scrollable, default_option):
    """
    Create radio buttons with captions from a dictionary structure.
    
    Args:
        option_caption_dict: Dict with format {key: {"option": "label", "caption": "description"}}
        key: Streamlit component key
        scrollable: If True, puts radio buttons in scrollable container
        default_option: The default key to select
        
    Returns:
        The selected key from the option_caption_dict
    """
    # Extract option labels and captions from the dictionary
    options = [v["option"] for v in option_caption_dict.values()]
    captions = [v["caption"] for v in option_caption_dict.values()]
    key_map = {v["option"]: k for k, v in option_caption_dict.items()}

    # Get default index based on default_key
    default_index = list(option_caption_dict.keys()).index(default_option)

    # Create a radio button selection with captions
    with st.container(border=True, height=275 if scrollable else None):
        selected_option = st.radio(
            label=key,
            options=options,
            index=default_index,
            label_visibility="collapsed",
            captions=captions
        )

    # Return the corresponding key
    return key_map[selected_option]