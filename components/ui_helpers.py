"""
UI helper functions for AddaxAI Streamlit application.
"""

import streamlit as st
from st_flexible_callout_elements import flexible_callout
from streamlit_lottie import st_lottie
import random


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
        st.sidebar.markdown(
            f"<small>{line}<b>{label_text}</b></small>", unsafe_allow_html=True, help=help_text)
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
    Full-page blocking loader for Streamlit.
    Prevents any user interaction while active.
    Transparent background, solid fixed-size loader box.
    Shows a random animal emoji on each update.
    """

    def __init__(self, overlay_color="rgba(255, 255, 255, 0.6)"):
        self.overlay_color = overlay_color
        self.current_text = "Loading..."
        self.current_emoji = "🐿️"
        self.animals = [
            "🦌", "🐗", "🦊", "🐺", "🐻", "🐼", "🐨", "🦘",
            "🦡", "🦦", "🦥", "🐘", "🦏", "🦬", "🐒", "🦍",
            "🦧", "🐆", "🐅", "🐈‍⬛", "🐈", "🦅", "🦉", "🦆",
            "🦢", "🦜", "🐦", "🐍", "🐊", "🐢", "🦎", "🐸",
        ]

    def _render_overlay(self, text, emoji):
        overlay_html = f"""
        <style>
        body {{
            pointer-events: none !important; /* Block all interaction */
        }}
        .blocking-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: {self.overlay_color};
            z-index: 999999 !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .loader-box {{
            width: 500px;
            height: 350px;
            background: #ffffff;
            border-radius: 18px;
            box-shadow: 0px 6px 24px rgba(0,0,0,0.25);
            text-align: center;
            pointer-events: all !important;
            font-family: "Source Sans Pro", sans-serif; /* Streamlit default */
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        .loader-emoji {{
            font-size: 64px;
            margin-bottom: 20px;
        }}
        .loader-text {{
            font-size: 24px;
            color: #31333F; /* Streamlit default text color */
        }}
        </style>
        <div class="blocking-overlay">
            <div class="loader-box">
                <div class="loader-emoji">{emoji}</div>
                <div class="loader-text">{text}</div>
            </div>
        </div>
        """
        st.markdown(overlay_html, unsafe_allow_html=True)

    def open(self, initial_text="Loading..."):
        self.current_text = initial_text
        self.current_emoji = random.choice(self.animals)
        self._render_overlay(self.current_text, self.current_emoji)

    def update_text(self, new_text):
        self.current_text = new_text
        self.current_emoji = random.choice(self.animals)
        self._render_overlay(self.current_text, self.current_emoji)

    def close(self):
        st.markdown(
            """
            <style>
            body { pointer-events: auto !important; }
            .blocking-overlay { display: none !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def header_large(text):
    """
    Display a large custom header.

    Args:
        text: The header text to display
    """
    st.markdown(f"<h1 style='font-size: 2.5rem; font-weight: 600; margin-bottom: 0.15rem;'>{text}</h1><hr style='margin-top: 0; margin-bottom: 1rem; border: none; border-top: 2px solid #0f6064;'>", unsafe_allow_html=True)


def header_medium(text):
    """
    Display a medium custom header.

    Args:
        text: The header text to display
    """
    st.markdown(f"<h2 style='font-size: 1.75rem; font-weight: 600; margin-bottom: 0.05rem; margin-top: 0;'>{text}</h2><hr style='margin-top: 0; margin-bottom: 0.75rem; border: none; border-top: 2px solid #0f6064;'>", unsafe_allow_html=True)


def header_small(text):
    """
    Display a small custom header.

    Args:
        text: The header text to display
    """
    st.markdown(f"<h3 style='font-size: 1.25rem; font-weight: 600; margin-bottom: 0.05rem;'>{text}</h3><hr style='margin-top: 0; margin-bottom: 0.5rem; border: none; border-top: 2px solid #0f6064;'>", unsafe_allow_html=True)
