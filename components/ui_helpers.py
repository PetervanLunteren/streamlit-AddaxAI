"""
UI helper functions for AddaxAI Streamlit application.
"""

import streamlit as st
from st_flexible_callout_elements import flexible_callout
from streamlit_lottie import st_lottie


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

def success_box(msg, title=None, icon=":material/check_circle:"):
    """
    Display a success callout box.
    
    Args:
        msg: The message to display
        title: Optional title (will be bold)
        icon: Icon to display (default: check_circle icon)
    """
    if title:
        msg = f'<span style="font-weight: bold;">{title}</span><br>{msg}'
    
    flexible_callout(msg,
                     icon=icon,
                     background_color="#d9e3e7af",
                     font_color="#086164",
                     icon_size=23)


def code_span(text):
    """
    Wrap text in a styled code span with monospace font and specific color.
    
    Args:
        text: The text to wrap in code styling
        
    Returns:
        str: HTML formatted code span
    """
    return f"<code style='color:#086164; font-family:monospace;'>{text}</code>"

class BlockingLoader:
    """
    A simple blocking loader with just CSS overlay and text.
    Blocks all user interaction while active.
    """
    
    def __init__(self, overlay_color="rgba(255, 255, 255, 0.95)"):
        self.overlay_color = overlay_color
        self.current_text = "Loading..."

    def open(self, initial_text="Loading..."):
        """Open the blocking loader with overlay and text."""
        self.current_text = initial_text
        overlay_html = f"""
        <style>
        .blocking-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: {self.overlay_color};
            z-index: 9999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: "Source Sans Pro", sans-serif;
            font-size: 24px;
        }}
        .loader-content {{
            text-align: center;
        }}
        .loader-emoji {{
            font-size: 48px;
            margin-bottom: 20px;
        }}
        .loader-text {{
            color: #333;
            margin-bottom: 10px;
        }}
        </style>
        <div class="blocking-overlay">
            <div class="loader-content">
                <div class="loader-emoji">🐿️</div>
                <div class="loader-text" id="loader-status">{initial_text}</div>
            </div>
        </div>
        """
        st.markdown(overlay_html, unsafe_allow_html=True)

    def update_text(self, new_text):
        """Update the status text by re-rendering the overlay."""
        self.current_text = new_text
        # Re-render the entire overlay with new text
        overlay_html = f"""
        <style>
        .blocking-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: {self.overlay_color};
            z-index: 9999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: "Source Sans Pro", sans-serif;
            font-size: 24px;
        }}
        .loader-content {{
            text-align: center;
        }}
        .loader-emoji {{
            font-size: 48px;
            margin-bottom: 20px;
        }}
        .loader-text {{
            color: #333;
            margin-bottom: 10px;
        }}
        </style>
        <div class="blocking-overlay">
            <div class="loader-content">
                <div class="loader-emoji">🐿️</div>
                <div class="loader-text">{new_text}</div>
            </div>
        </div>
        """
        st.markdown(overlay_html, unsafe_allow_html=True)

    def close(self):
        """Close the loader and remove it from the page."""
        st.markdown(
            "<style>.blocking-overlay{display:none !important;}</style>", 
            unsafe_allow_html=True
        )
