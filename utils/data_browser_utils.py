"""
AddaxAI Data Browser Utilities

Utility functions for the data browser page.
Provides image modal viewer and filter state management for AgGrid data browsing.
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


def image_viewer_modal():
    """
    Display full-resolution image in modal with metadata and navigation.
    Shows original image with optional bounding box overlay.
    """
    import streamlit as st
    from PIL import Image, ImageDraw
    import os
    import pandas as pd
    from utils.common import get_session_var, set_session_var
    
    # Get current modal state
    current_index = get_session_var("explore_results", "modal_current_image_index", 0)
    show_bbox = get_session_var("explore_results", "modal_show_bbox", False)
    
    # Get the filtered dataframe from session state
    df_filtered = st.session_state.get('results_detections_filtered', pd.DataFrame())
    
    if df_filtered.empty or current_index >= len(df_filtered):
        st.error("No image data available")
        if st.button("Close"):
            set_session_var("explore_results", "show_modal_image_viewer", False)
            st.rerun()
        return
    
    # Get current row data
    current_row = df_filtered.iloc[current_index]
    image_path = current_row.get('absolute_path', '')
    
    # Main layout: image on left, metadata on right
    img_col, meta_col = st.columns([3, 1])
    
    with img_col:
        # Segmented control for bbox overlay
        bbox_options = ["Original", "With Detection Box"]
        selected_option = st.segmented_control(
            "Display mode",
            options=bbox_options,
            default=bbox_options[1] if show_bbox else bbox_options[0],
            key="bbox_toggle"
        )
        
        # Update bbox state if changed
        new_show_bbox = (selected_option == "With Detection Box")
        if new_show_bbox != show_bbox:
            set_session_var("explore_results", "modal_show_bbox", new_show_bbox)
            show_bbox = new_show_bbox
            st.rerun()
        
        # Load and display image
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path)
                
                # Add bounding box if requested
                if show_bbox:
                    bbox_data = {
                        'x': current_row.get('bbox_x'),
                        'y': current_row.get('bbox_y'), 
                        'width': current_row.get('bbox_width'),
                        'height': current_row.get('bbox_height')
                    }
                    
                    if all(pd.notna([bbox_data['x'], bbox_data['y'], 
                                   bbox_data['width'], bbox_data['height']])):
                        img_with_bbox = img.copy()
                        draw = ImageDraw.Draw(img_with_bbox)
                        
                        # Convert normalized coordinates to pixel coordinates
                        img_width, img_height = img.size
                        x = int(bbox_data['x'] * img_width)
                        y = int(bbox_data['y'] * img_height)
                        w = int(bbox_data['width'] * img_width)
                        h = int(bbox_data['height'] * img_height)
                        
                        # Draw red bounding box
                        draw.rectangle([x, y, x + w, y + h], outline='red', width=3)
                        img = img_with_bbox
                
                # Display image
                st.image(img, width="stretch")
            else:
                st.error(f"Image file not found: {image_path}")
                
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    
    with meta_col:
        st.subheader("Image Details")
        
        # Detection information
        st.markdown("**Detection**")
        det_label = current_row.get('detection_label', 'N/A')
        det_conf = current_row.get('detection_confidence', 0)
        if pd.notna(det_conf):
            st.write(f"• {det_label} ({det_conf:.2f})")
        else:
            st.write(f"• {det_label}")
        
        # Classification information
        st.markdown("**Classification**")
        cls_label = current_row.get('classification_label', 'N/A')
        cls_conf = current_row.get('classification_confidence', 0)
        if pd.notna(cls_conf):
            st.write(f"• {cls_label} ({cls_conf:.2f})")
        else:
            st.write(f"• {cls_label}")
        
        # Timestamp
        st.markdown("**Timestamp**")
        timestamp = current_row.get('timestamp', 'N/A')
        st.write(f"• {timestamp}")
        
        # Location information
        st.markdown("**Location**")
        location_id = current_row.get('location_id', 'N/A')
        deployment_id = current_row.get('deployment_id', 'N/A')
        st.write(f"• Location: {location_id}")
        st.write(f"• Deployment: {deployment_id}")
        
        # File paths
        st.markdown("**File Path**")
        rel_path = current_row.get('relative_path', 'N/A')
        st.write(f"• Relative: {rel_path}")
        st.code(image_path, language=None)
    
    # Navigation and control buttons
    st.divider()
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("← Previous", disabled=(current_index <= 0)):
            set_session_var("explore_results", "modal_current_image_index", current_index - 1)
            st.rerun()
    
    with col2:
        if st.button("Next →", disabled=(current_index >= len(df_filtered) - 1)):
            set_session_var("explore_results", "modal_current_image_index", current_index + 1)
            st.rerun()
    
    with col3:
        st.write(f"Image {current_index + 1} of {len(df_filtered)}")
    
    with col4:
        if st.button("Close", type="primary"):
            set_session_var("explore_results", "show_modal_image_viewer", False)
            st.rerun()