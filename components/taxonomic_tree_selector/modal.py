"""
Taxonomic Tree Selector Modal

Modal component for selecting species using a hierarchical tree view.
"""

import streamlit as st
from st_checkbox_tree import checkbox_tree
from utils.common import get_session_var, set_session_var, update_session_vars
from .tree_builder import build_tree_from_species, get_all_species_from_tree


def tree_selector_modal(available, selected, key="tree_selector"):
    """
    Display a modal with hierarchical tree selector for species.

    Args:
        available: List of all species to show in tree (model_class values)
                  e.g. ["lion", "cat", "bird", "wolf", ...]
        selected: List of pre-selected species (model_class values)
                 e.g. ["lion", "bird"]
        key: Unique key for this modal instance

    Returns:
        list: Selected species if Apply clicked, None if Cancel or modal not shown

    Usage:
        # Open modal
        set_session_var("my_tool", "show_tree_modal", True)

        # Render modal
        if get_session_var("my_tool", "show_tree_modal"):
            result = tree_selector_modal(
                available=all_species,
                selected=current_selection,
                key="my_tool_tree"
            )

            if result is not None:  # Apply was clicked
                save_selection(result)
                set_session_var("my_tool", "show_tree_modal", False)
                st.rerun()
    """

    # Session state keys for this modal instance
    session_key_selected = f"{key}_selected"
    session_key_expanded = f"{key}_expanded"
    session_key_last = f"{key}_last"
    session_key_dismissed = f"{key}_dismissed"

    # Initialize session state for this modal
    if session_key_selected not in st.session_state:
        st.session_state[session_key_selected] = selected or []
    if session_key_expanded not in st.session_state:
        st.session_state[session_key_expanded] = []
    if session_key_last not in st.session_state:
        st.session_state[session_key_last] = {}

    # Build tree from available species
    tree = build_tree_from_species(available)

    if not tree:
        st.warning("No taxonomy data available for the selected species.")
        if st.button("Close", key=f"{key}_close_no_data"):
            return None
        return None

    # Get all species from tree (for select all/none)
    all_species_in_tree = get_all_species_from_tree(tree)

    # Get current selection
    current_selected = st.session_state[session_key_selected]
    current_expanded = st.session_state[session_key_expanded]

    # Calculate selection statistics
    selected_count = len(current_selected)
    total_count = len(all_species_in_tree)

    from components.ui_helpers import code_span
    st.markdown(
        f"<div style='text-align:center;'>Selected {code_span(str(selected_count))} of {code_span(str(total_count))}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Select All / Select None buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Select All", key=f"{key}_select_all", use_container_width=True):
            st.session_state[session_key_selected] = all_species_in_tree
            st.rerun()

    with col2:
        if st.button("Select None", key=f"{key}_select_none", use_container_width=True):
            st.session_state[session_key_selected] = []
            st.rerun()

    # Tree widget
    with st.container(border=True, height=500):
        tree_result = checkbox_tree(
            tree,
            check_model="leaf",
            checked=current_selected,
            expanded=current_expanded,
            show_expand_all=True,
            half_check_color="#086164",
            check_color="#086164",
            key=f"{key}_tree",
            show_tree_lines=True,
            tree_line_color="#e9e9eb"
        )

    # Handle tree selection changes
    if tree_result:
        new_checked = tree_result.get("checked", current_selected)
        new_expanded = tree_result.get("expanded", current_expanded)
        last_state = st.session_state[session_key_last]
        last_checked = last_state.get("checked", [])
        last_expanded = last_state.get("expanded", [])

        # Update session state if selection changed
        if new_checked != last_checked or new_expanded != last_expanded:
            st.session_state[session_key_selected] = new_checked
            st.session_state[session_key_expanded] = new_expanded
            st.session_state[session_key_last] = tree_result
            st.rerun()

    # Selection summary
    # Apply / Cancel buttons
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Cancel", key=f"{key}_cancel", use_container_width=True):
            # Clear modal state
            _clear_modal_state(key)
            st.session_state[session_key_dismissed] = "cancel"
            return None

    with col2:
        if st.button("Apply", key=f"{key}_apply", use_container_width=True, type="primary"):
            # Get final selection
            final_selection = st.session_state[session_key_selected]

            # Clear modal state
            _clear_modal_state(key)
            st.session_state[session_key_dismissed] = "apply"

            # Return selection
            return final_selection

    # Modal is still open, return None (no action yet)
    return None


def _clear_modal_state(key):
    """
    Clear session state for a modal instance.

    Args:
        key: Modal key
    """
    session_keys = [
        f"{key}_selected",
        f"{key}_expanded",
        f"{key}_last",
        f"{key}_dismissed"
    ]

    for session_key in session_keys:
        if session_key in st.session_state:
            del st.session_state[session_key]
