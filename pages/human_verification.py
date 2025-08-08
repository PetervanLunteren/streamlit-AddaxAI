"""
Human Verification Page - Bounding Box Annotation and Verification

This page allows users to verify and annotate bounding boxes from AI detection results.
Supports event-based batch processing and individual image annotation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from PIL import Image
import random
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager

# Import AddaxAI components and utilities
from components import print_widget_label, StepperBar, info_box, warning_box
from utils.common import (
    init_session_state, get_session_var, set_session_var, update_session_vars, clear_vars
)
from utils.config import ADDAXAI_FILES_ST

# Initialize session state for this tool
init_session_state("human_verification")

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def create_sample_detection_data():
    """Create fictional detection DataFrame with realistic wildlife data"""
    
    # Get list of available test images
    test_images_dir = os.path.join(ADDAXAI_FILES_ST, "data", "test-images")
    available_images = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]
    
    # Species categories for realistic data
    species_list = [
        "Lion", "Elephant", "Zebra", "Giraffe", "Buffalo", "Leopard", 
        "Cheetah", "Rhino", "Hippo", "Warthog", "Impala", "Kudu"
    ]
    
    # Generate sample detection data
    np.random.seed(42)  # For reproducible data
    data = []
    
    event_id = 1
    for i, img_file in enumerate(available_images[:30]):  # Use first 30 images
        # Create 1-3 detections per image
        num_detections = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        
        # Every 3-5 images belong to the same event (video sequence)
        if i % np.random.choice([3, 4, 5]) == 0:
            event_id += 1
            
        for detection_idx in range(num_detections):
            # Random bounding box coordinates (normalized 0-1)
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.1, 0.3)
            
            # Ensure box stays within image bounds
            x_min = max(0, x_center - width/2)
            y_min = max(0, y_center - height/2)
            x_max = min(1, x_center + width/2)
            y_max = min(1, y_center + height/2)
            
            bbox_coords = [x_min, y_min, x_max, y_max]
            
            # Random species and confidences
            species = np.random.choice(species_list)
            det_confidence = np.random.uniform(0.3, 0.98)
            cls_confidence = np.random.uniform(0.4, 0.95) if species != "Unknown" else 0.0
            
            data.append({
                'image_path': os.path.join(test_images_dir, img_file),
                'bbox_coords': bbox_coords,
                'detection_confidence': det_confidence,
                'species_label': species,
                'classification_confidence': cls_confidence,
                'eventID': f"event_{event_id:03d}",
                'projectID': "sample_project",
                'verified_status': np.random.choice(["unverified", "verified", "rejected"], p=[0.7, 0.2, 0.1]),
                'detection_id': f"det_{i}_{detection_idx}"
            })
    
    return pd.DataFrame(data)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_verification_stats(df):
    """Calculate verification statistics"""
    total = len(df)
    verified = len(df[df['verified_status'] == 'verified'])
    rejected = len(df[df['verified_status'] == 'rejected'])
    unverified = len(df[df['verified_status'] == 'unverified'])
    
    return {
        'total': total,
        'verified': verified,
        'rejected': rejected,
        'unverified': unverified,
        'percent_complete': ((verified + rejected) / total * 100) if total > 0 else 0
    }

def convert_bbox_to_pixels(bbox_coords, img_width, img_height):
    """Convert normalized bbox coordinates to pixel coordinates"""
    x_min, y_min, x_max, y_max = bbox_coords
    return [
        int(x_min * img_width),
        int(y_min * img_height), 
        int(x_max * img_width),
        int(y_max * img_height)
    ]

def convert_bbox_to_normalized(bbox_coords, img_width, img_height):
    """Convert pixel bbox coordinates to normalized coordinates"""
    x_min, y_min, x_max, y_max = bbox_coords
    return [
        x_min / img_width,
        y_min / img_height,
        x_max / img_width,
        y_max / img_height
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Load or create sample data
if not get_session_var("human_verification", "sample_data_loaded", False):
    sample_df = create_sample_detection_data()
    set_session_var("human_verification", "detection_data", sample_df)
    set_session_var("human_verification", "sample_data_loaded", True)
else:
    sample_df = get_session_var("human_verification", "detection_data")

# Get current step for stepper
step = get_session_var("human_verification", "step", 0)

# Stepper for workflow navigation
stepper_steps = ["Filter Data", "Event Overview", "Annotate Images"]

stepper = StepperBar(
    steps=stepper_steps,
    orientation="horizontal",
    active_color="#086164",
    completed_color="#0861647D",
    inactive_color="#dadfeb"
)
stepper.set_step(step)

# Display the stepper
st.markdown(stepper.display(), unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# STEP 0: DATA FILTERING
# ───────────────────────────────────────────────────────────────────────────────

if step == 0:
    st.header(":material/filter_alt: Filter Detection Data")
    
    # Display overall statistics
    stats = get_verification_stats(sample_df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Detections", stats['total'])
    with col2:
        st.metric("Verified", stats['verified'])
    with col3:
        st.metric("Rejected", stats['rejected']) 
    with col4:
        st.metric("Progress", f"{stats['percent_complete']:.1f}%")
    
    st.divider()
    
    # Filtering controls
    col1, col2 = st.columns(2)
    
    with col1:
        print_widget_label("Species Filter", help_text="Select species to verify")
        available_species = sorted(sample_df['species_label'].unique())
        selected_species = st.multiselect(
            "Species",
            options=available_species,
            default=available_species[:3],  # Select first 3 by default
            label_visibility="collapsed"
        )
    
    with col2:
        print_widget_label("Verification Status", help_text="Filter by current verification status")
        selected_status = st.multiselect(
            "Status",
            options=["unverified", "verified", "rejected"],
            default=["unverified"],
            label_visibility="collapsed"
        )
    
    # Confidence filters
    col1, col2 = st.columns(2)
    
    with col1:
        print_widget_label("Detection Confidence", help_text="Filter by AI detection confidence")
        det_conf_range = st.slider(
            "Detection Confidence",
            min_value=0.0, max_value=1.0,
            value=(0.3, 1.0),
            step=0.05,
            format="%.2f",
            label_visibility="collapsed"
        )
    
    with col2:
        print_widget_label("Classification Confidence", help_text="Filter by species classification confidence")
        cls_conf_range = st.slider(
            "Classification Confidence", 
            min_value=0.0, max_value=1.0,
            value=(0.0, 1.0),
            step=0.05,
            format="%.2f",
            label_visibility="collapsed"
        )
    
    # Apply filters
    filtered_df = sample_df[
        (sample_df['species_label'].isin(selected_species)) &
        (sample_df['verified_status'].isin(selected_status)) &
        (sample_df['detection_confidence'] >= det_conf_range[0]) &
        (sample_df['detection_confidence'] <= det_conf_range[1]) &
        (sample_df['classification_confidence'] >= cls_conf_range[0]) &
        (sample_df['classification_confidence'] <= cls_conf_range[1])
    ]
    
    # Show filtered results
    st.divider()
    filtered_stats = get_verification_stats(filtered_df)
    st.write(f"**Filtered Results:** {filtered_stats['total']} detections in {filtered_df['eventID'].nunique()} events")
    
    if len(filtered_df) > 0:
        if st.button(":material/arrow_forward: Continue to Event Overview", type="primary"):
            # Save filtered data and advance to next step
            set_session_var("human_verification", "filtered_data", filtered_df)
            set_session_var("human_verification", "step", 1)
            st.rerun()
    else:
        warning_box("No detections match the current filters. Please adjust your criteria.")

# ───────────────────────────────────────────────────────────────────────────────
# STEP 1: EVENT OVERVIEW
# ───────────────────────────────────────────────────────────────────────────────

elif step == 1:
    st.header(":material/view_module: Event Overview")
    
    filtered_df = get_session_var("human_verification", "filtered_data")
    
    if filtered_df is None or len(filtered_df) == 0:
        warning_box("No filtered data found. Please go back to step 1 to filter data.")
        if st.button(":material/arrow_back: Back to Filtering"):
            set_session_var("human_verification", "step", 0)
            st.rerun()
    else:
        # Group by events
        events_df = filtered_df.groupby('eventID').agg({
            'image_path': 'nunique',
            'detection_id': 'count',
            'species_label': lambda x: ', '.join(x.unique()),
            'verified_status': lambda x: (x == 'verified').sum(),
            'detection_confidence': 'mean'
        }).reset_index()
        
        events_df.columns = ['Event ID', 'Images', 'Detections', 'Species', 'Verified', 'Avg Confidence']
        
        st.write(f"**{len(events_df)} events** with detection data:")
        
        # Event selection and batch operations
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Event cards display
            for idx, event_row in events_df.iterrows():
                event_id = event_row['Event ID']
                with st.container(border=True):
                    # Event info and actions in a single row without nested columns
                    st.write(f"**{event_id}** | 📸 {event_row['Images']} images | 🎯 {event_row['Detections']} detections | ✅ {event_row['Verified']} verified | 🔍 {event_row['Avg Confidence']:.2f} avg conf")
                    
                    # Show thumbnail strip of event images using HTML layout
                    event_images = filtered_df[filtered_df['eventID'] == event_id]['image_path'].unique()[:5]
                    
                    # Create thumbnail display using st.columns but not nested
                    if len(event_images) > 0:
                        st.write("**Images:**")
                        # Use HTML for thumbnail display to avoid nested columns
                        thumb_html = "<div style='display: flex; gap: 5px; margin: 10px 0;'>"
                        for img_path in event_images:
                            try:
                                # For now, just show placeholders - we'll improve this later
                                thumb_html += f"<div style='width: 60px; height: 40px; background-color: #f0f0f0; border: 1px solid #ccc; display: flex; align-items: center; justify-content: center; font-size: 12px;'>IMG</div>"
                            except:
                                thumb_html += "<div style='width: 60px; height: 40px; background-color: #ffcccc; border: 1px solid #ccc; display: flex; align-items: center; justify-content: center; font-size: 12px;'>❌</div>"
                        thumb_html += "</div>"
                        st.markdown(thumb_html, unsafe_allow_html=True)
                    
                    # Action button
                    if st.button(f"Annotate {event_id}", key=f"annotate_{event_id}", type="primary"):
                        set_session_var("human_verification", "selected_event", event_id)
                        set_session_var("human_verification", "step", 2)
                        st.rerun()
        
        with col2:
            st.subheader("Batch Operations")
            
            # Batch species assignment
            st.write("**Batch Assign Species:**")
            batch_species = st.selectbox("Select Species", options=sorted(sample_df['species_label'].unique()))
            
            if st.button("Apply to All Events"):
                # This would update all detections in visible events
                st.success(f"Would assign '{batch_species}' to all detections")
            
            st.divider()
            
            # Batch verification
            if st.button("Mark All as Verified", type="secondary"):
                # Update verification status
                filtered_df.loc[:, 'verified_status'] = 'verified'
                set_session_var("human_verification", "filtered_data", filtered_df)
                st.success("All detections marked as verified!")
                st.rerun()
        
        # Navigation
        st.divider()
        if st.button(":material/arrow_back: Back to Filtering"):
            set_session_var("human_verification", "step", 0)
            st.rerun()

# ───────────────────────────────────────────────────────────────────────────────
# STEP 2: IMAGE ANNOTATION
# ───────────────────────────────────────────────────────────────────────────────

elif step == 2:
    st.header(":material/edit_location_alt: Image Annotation")
    
    filtered_df = get_session_var("human_verification", "filtered_data")
    selected_event = get_session_var("human_verification", "selected_event")
    
    if filtered_df is None or selected_event is None:
        warning_box("No event selected. Please go back to event overview.")
        if st.button(":material/arrow_back: Back to Events"):
            set_session_var("human_verification", "step", 1)
            st.rerun()
    else:
        # Get images for selected event
        event_data = filtered_df[filtered_df['eventID'] == selected_event].copy()
        unique_images = event_data['image_path'].unique()
        
        # Image navigation
        current_img_idx = get_session_var("human_verification", "current_image_idx", 0)
        
        # Navigation controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button(":material/arrow_back_ios: Previous") and current_img_idx > 0:
                set_session_var("human_verification", "current_image_idx", current_img_idx - 1)
                st.rerun()
        
        with col2:
            st.write(f"**{selected_event}** - Image {current_img_idx + 1} of {len(unique_images)}")
        
        with col3:
            if st.button(":material/arrow_forward_ios: Next") and current_img_idx < len(unique_images) - 1:
                set_session_var("human_verification", "current_image_idx", current_img_idx + 1)
                st.rerun()
        
        if current_img_idx < len(unique_images):
            current_image_path = unique_images[current_img_idx]
            
            # Get detections for current image
            current_detections = event_data[event_data['image_path'] == current_image_path].copy()
            
            # Image annotation interface
            col_img, col_controls = st.columns([2, 1])
            
            with col_img:
                try:
                    img = Image.open(current_image_path)
                    img_width, img_height = img.size
                    
                    # Convert bounding boxes to the format expected by st_img_label
                    rects = []
                    for _, detection in current_detections.iterrows():
                        bbox_pixels = convert_bbox_to_pixels(detection['bbox_coords'], img_width, img_height)
                        # st_img_label expects: {'left': x, 'top': y, 'width': w, 'height': h}
                        rects.append({
                            'left': bbox_pixels[0],
                            'top': bbox_pixels[1], 
                            'width': bbox_pixels[2] - bbox_pixels[0],
                            'height': bbox_pixels[3] - bbox_pixels[1]
                        })
                    
                    # Image annotation component
                    result = st_img_label(
                        resized_img=img,
                        box_color="red",
                        rects=rects,
                        key=f"annotation_{selected_event}_{current_img_idx}"
                    )
                    
                    # Process annotation results
                    if result is not None:
                        st.write("**Updated Annotations:**")
                        for i, rect in enumerate(result):
                            st.write(f"Box {i+1}: x={rect['left']}, y={rect['top']}, w={rect['width']}, h={rect['height']}")
                    
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
            
            with col_controls:
                st.subheader("Detection Controls")
                
                # Show current detections
                st.write("**Current Detections:**")
                for det_idx, detection in current_detections.iterrows():
                    with st.container(border=True):
                        st.write(f"**Detection {det_idx + 1}**")
                        
                        # Species selection dropdown
                        current_species = detection['species_label']
                        available_species = sorted(sample_df['species_label'].unique())
                        
                        new_species = st.selectbox(
                            "Species:",
                            options=available_species,
                            index=available_species.index(current_species) if current_species in available_species else 0,
                            key=f"species_{detection['detection_id']}"
                        )
                        
                        # Update species if changed
                        if new_species != current_species:
                            mask = filtered_df['detection_id'] == detection['detection_id']
                            filtered_df.loc[mask, 'species_label'] = new_species
                            set_session_var("human_verification", "filtered_data", filtered_df)
                        
                        st.write(f"Det Conf: {detection['detection_confidence']:.2f}")
                        st.write(f"Cls Conf: {detection['classification_confidence']:.2f}")
                        st.write(f"Status: {detection['verified_status']}")
                        
                        # Quick actions
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("✅ Verify", key=f"verify_{detection['detection_id']}"):
                                # Update detection status
                                mask = filtered_df['detection_id'] == detection['detection_id']
                                filtered_df.loc[mask, 'verified_status'] = 'verified'
                                set_session_var("human_verification", "filtered_data", filtered_df)
                                st.rerun()
                        
                        with col_b:
                            if st.button("❌ Reject", key=f"reject_{detection['detection_id']}"):
                                # Update detection status
                                mask = filtered_df['detection_id'] == detection['detection_id']
                                filtered_df.loc[mask, 'verified_status'] = 'rejected'
                                set_session_var("human_verification", "filtered_data", filtered_df)
                                st.rerun()
                
                st.divider()
                
                # Add new detection option
                if st.button("➕ Add New Detection", type="secondary"):
                    st.info("Draw a new bounding box on the image above, then refresh to add it to the detections.")
                
                st.divider()
                st.subheader("Keyboard Shortcuts")
                st.write("• **Space**: Next image")
                st.write("• **Enter**: Verify all")
                st.write("• **Backspace**: Previous")
                st.write("• **R**: Reject all")
        
        # Navigation back to events
        st.divider()
        if st.button(":material/arrow_back: Back to Event Overview"):
            set_session_var("human_verification", "step", 1)
            set_session_var("human_verification", "current_image_idx", 0)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.subheader("Verification Progress")
    
    # Get current data for stats
    current_data = get_session_var("human_verification", "filtered_data")
    if current_data is not None:
        stats = get_verification_stats(current_data)
        
        # Progress bar
        progress = stats['percent_complete'] / 100
        st.progress(progress)
        st.write(f"{stats['percent_complete']:.1f}% Complete")
        
        # Detailed stats
        st.metric("Total Detections", stats['total'])
        st.metric("Verified", stats['verified'])
        st.metric("Rejected", stats['rejected'])
        st.metric("Remaining", stats['unverified'])
    
    st.divider()
    
    # Reset option
    if st.button("🔄 Reset Session", type="secondary"):
        clear_vars("human_verification")
        st.rerun()