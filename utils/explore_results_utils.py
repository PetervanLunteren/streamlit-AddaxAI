"""
AddaxAI Explore Results Utilities

Simplified utility functions for the explore results page.
Since we're using st.data_editor with built-in ImageColumn,
we no longer need complex thumbnail generation or AgGrid configuration.
"""

import json
from utils.config import ADDAXAI_ROOT, log
from utils.common import load_vars, update_vars


def load_filter_state():
    """
    Load saved filter and display state from persistent storage.
    
    Returns:
        dict: Filter state configuration
    """
    try:
        vars_data = load_vars("explore_results")
        return vars_data.get("filter_state", {})
    except Exception as e:
        log(f"Error loading filter state: {str(e)}")
        return {}


def save_filter_state(filter_state):
    """
    Save filter and display state to persistent storage.
    
    Args:
        filter_state (dict): Filter state configuration to save
    """
    try:
        update_vars("explore_results", {"filter_state": filter_state})
        log("Filter state saved successfully")
    except Exception as e:
        log(f"Error saving filter state: {str(e)}")