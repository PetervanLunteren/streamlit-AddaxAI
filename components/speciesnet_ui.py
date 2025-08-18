"""
SpeciesNet-specific UI components and logic

This module contains all SpeciesNet-specific user interface components,
validation logic, and settings management to keep the main analysis flow clean.
"""

import streamlit as st
import json
from utils.analysis_utils import (
    country_selector_widget, 
    state_selector_widget,
    set_session_var, 
    get_cached_vars, 
    get_cached_map, 
    invalidate_map_cache
)
from components.ui_helpers import print_widget_label, warning_box


def is_speciesnet_model(model_id: str) -> bool:
    """Check if a model ID is a SpeciesNet model."""
    if not model_id:
        return False
    return model_id.upper().startswith("SPECIESNET")


def render_speciesnet_species_selector():
    """
    Render the SpeciesNet-specific species selection UI.
    
    Returns:
        tuple: (selected_species, selected_country, selected_state)
               selected_species will always be None for SpeciesNet
    """
    # SpeciesNet doesn't use species selection - it uses country/state geofencing
    selected_species = None
    
    # Country selection container
    with st.container(border=True):
        print_widget_label("Country selection",
                          help_text="Select a country to determine species presence for SPECIESNET models.")
        selected_country = country_selector_widget()
    
    # State selection (USA only)
    if selected_country == "USA":
        with st.container(border=True):
            print_widget_label("State selection", 
                              help_text="Select a US state for more specific species presence data.")
            selected_state = state_selector_widget()
    else:
        selected_state = None
        _clear_state_selections()
    
    return selected_species, selected_country, selected_state


def _clear_state_selections():
    """Clear state selections when country is not USA."""
    # Clear session state
    set_session_var("analyse_advanced", "selected_state", None)
    set_session_var("analyse_advanced", "selected_state_display", None)
    
    # Clear persistent storage if project is selected
    general_settings_vars = get_cached_vars(section="general_settings")
    selected_projectID = general_settings_vars.get("selected_projectID")
    
    if selected_projectID:
        map_data, map_file_path = get_cached_map()
        if (selected_projectID in map_data["projects"] and 
            "speciesnet_settings" in map_data["projects"][selected_projectID]):
            
            speciesnet_settings = map_data["projects"][selected_projectID]["speciesnet_settings"]
            if speciesnet_settings.get("state") is not None:
                speciesnet_settings["state"] = None
                
                # Save updated map
                with open(map_file_path, 'w') as f:
                    json.dump(map_data, f, indent=2)
                # Invalidate cache so next read gets fresh data
                invalidate_map_cache()


def validate_speciesnet_requirements(selected_country: str) -> bool:
    """
    Validate that SpeciesNet requirements are met.
    
    Args:
        selected_country: The selected country code
        
    Returns:
        bool: True if validation passes, False otherwise
        
    Side effects:
        Shows warning box if validation fails
    """
    if not selected_country:
        warning_box(
            msg="You need to select a country to determine species presence for SPECIESNET models.",
            title="Country selection required"
        )
        return False
    
    return True


def get_speciesnet_model_requirements():
    """
    Get the requirements description for SpeciesNet models.
    
    Returns:
        dict: Requirements information for UI display
    """
    return {
        "requires_species_selection": False,
        "requires_country_selection": True,
        "requires_state_selection": False,  # Optional, only for USA
        "description": "SpeciesNet uses country-based geofencing instead of species selection"
    }