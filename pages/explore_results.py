"""
AddaxAI Results Explorer

Interactive data table for exploring camera trap detection results with:
- AgGrid table with advanced filtering and sorting
- Image thumbnails with click-to-view modal
- Persistent filter state and pagination
- Performance optimized for large datasets
"""

import streamlit as st
import os
from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode, GridOptionsBuilder
from st_modal import Modal

from utils.config import log
from utils.explore_results_utils import (
    load_filter_state, save_filter_state,
    add_thumbnails_to_dataframe,
    format_dataframe_for_display
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE INITIALIZATION AND DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(layout="wide")
st.title("🔍 Explore Detection Results")

# Check if detection results are available
if "results_detections" not in st.session_state:
    st.error("No detection results found. Please ensure the application has been started properly and detection data is loaded.")
    st.info("Detection results are loaded during application startup. Try refreshing the page or restarting the application.")
    st.stop()

# Load detection results dataframe
results_df = st.session_state["results_detections"]

if results_df.empty:
    st.warning("No detection results found in the database.")
    st.info("This could mean:")
    st.write("- No deployments have been processed yet")
    st.write("- All deployments failed to process")
    st.write("- Detection result files are missing or corrupted")
    st.stop()

# Load saved filter state
filter_state = load_filter_state()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.header("Display Options")

# Thumbnail size control
thumbnail_size = st.sidebar.slider(
    "Thumbnail Size (px)",
    min_value=50,
    max_value=200,
    value=filter_state.get("thumbnail_size", 100),
    step=10,
    help="Size of image thumbnails in the table"
)

# Page size control
page_size = st.sidebar.selectbox(
    "Rows per Page",
    options=[25, 50, 100, 200],
    index=[25, 50, 100, 200].index(filter_state.get("page_size", 50)),
    help="Number of rows to display per page"
)

# Show/hide columns toggle
show_all_columns = st.sidebar.checkbox(
    "Show All Columns",
    value=filter_state.get("columns_visible", True),
    help="Toggle visibility of technical columns (paths, IDs)"
)

# Update filter state if changed
current_filter_state = {
    "page_size": page_size,
    "thumbnail_size": thumbnail_size,
    "columns_visible": show_all_columns,
    "grid_filters": filter_state.get("grid_filters", {}),
    "sort_model": filter_state.get("sort_model", [])
}

# Save state if settings changed
if current_filter_state != filter_state:
    save_filter_state(current_filter_state)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA TABLE PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

# Format dataframe for display
display_df = format_dataframe_for_display(results_df)

# Store original for thumbnails but don't add thumbnail column to AgGrid yet
display_df_with_thumbnails = add_thumbnails_to_dataframe(display_df.copy(), thumbnail_size)

# Configure AgGrid with GridOptionsBuilder for better image support
gb = GridOptionsBuilder.from_dataframe(display_df)

# Configure general grid options
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=page_size)
gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False)
gb.configure_selection(selection_mode='single', use_checkbox=False)

# Configure specific columns
for col in display_df.columns:
    if "confidence" in col:
        gb.configure_column(col, 
                          type=["numericColumn"],
                          precision=3,
                          width=100)
    elif col in ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]:
        gb.configure_column(col,
                          type=["numericColumn"], 
                          precision=4,
                          width=90)
    elif col == "absolute_path":
        gb.configure_column(col, hide=True)

grid_options = gb.build()

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE THUMBNAILS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

# Show thumbnails in a separate section above the table
st.subheader("Image Thumbnails")
if not display_df.empty:
    # Calculate how many images to show per page
    current_page = st.selectbox("Page", range(1, (len(display_df) // page_size) + 2), index=0, key="thumbnail_page")
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, len(display_df))
    
    # Display thumbnails for current page
    page_data = display_df.iloc[start_idx:end_idx]
    
    # Create columns for thumbnails (max 10 per row)
    cols_per_row = min(10, len(page_data))
    if cols_per_row > 0:
        rows_needed = (len(page_data) + cols_per_row - 1) // cols_per_row
        
        for row in range(rows_needed):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                data_idx = row * cols_per_row + col_idx
                if data_idx < len(page_data):
                    row_data = page_data.iloc[data_idx]
                    image_path = row_data['absolute_path']
                    
                    with cols[col_idx]:
                        # Create thumbnail using streamlit image
                        if image_path and os.path.exists(image_path):
                            if st.button(f"📷", key=f"thumb_{start_idx + data_idx}", help=f"{row_data['relative_path']}"):
                                # Store selected image in session state for modal
                                st.session_state['selected_image_data'] = row_data.to_dict()
                                st.rerun()
                            
                            # Show small image
                            try:
                                st.image(image_path, width=thumbnail_size, caption=f"{row_data['relative_path']}")
                            except:
                                st.write("❌ No image")
                        else:
                            st.write("❌ Missing")

# ═══════════════════════════════════════════════════════════════════════════════
# AGGRID TABLE DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("Data Table")

# Create AgGrid
try:
    grid_response = AgGrid(
        display_df,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=False,
        height=600,
        width='100%',
        reload_data=False
    )
    
    # Get selected rows
    selected_rows = grid_response['selected_rows']
    
except Exception as e:
    st.error(f"Error displaying data table: {str(e)}")
    log(f"AgGrid error: {str(e)}")
    st.info("Falling back to standard dataframe display...")
    
    # Fallback to standard dataframe if AgGrid fails
    st.dataframe(
        display_df.drop('thumbnail', axis=1, errors='ignore'),
        use_container_width=True,
        height=600
    )
    selected_rows = []

# ═══════════════════════════════════════════════════════════════════════════════
# ROW SELECTION HANDLING AND IMAGE MODAL
# ═══════════════════════════════════════════════════════════════════════════════

# Handle image selection - either from table row or thumbnail click
selected_row = None

# Check for thumbnail selection first
if 'selected_image_data' in st.session_state:
    selected_row = st.session_state['selected_image_data']
    # Clear the selection after use
    del st.session_state['selected_image_data']
# Then check for table row selection
elif selected_rows is not None and len(selected_rows) > 0:
    selected_row = selected_rows[0]  # Get first selected row

if selected_row:
    
    # Extract image information
    image_path = selected_row.get('absolute_path', '')
    image_name = selected_row.get('relative_path', 'Unknown')
    project_id = selected_row.get('project_id', 'Unknown')
    deployment_id = selected_row.get('deployment_id', 'Unknown')
    detection_conf = selected_row.get('detection_confidence', 0)
    classification_label = selected_row.get('classification_label', 'None')
    classification_conf = selected_row.get('classification_confidence', 0)
    
    # Create and show modal
    modal = Modal(
        title=f"📷 {image_name}",
        key="image_modal",
        max_width=800
    )
    
    with modal.container():
        # Image information
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display full-size image
            if image_path and image_path.strip():
                try:
                    st.image(image_path, caption=image_name, use_column_width=True)
                except Exception as e:
                    st.error(f"Could not load image: {str(e)}")
                    st.write(f"Path: {image_path}")
            else:
                st.warning("No image path available")
        
        with col2:
            # Image metadata
            st.write("**Image Details:**")
            st.write(f"📁 Project: {project_id}")
            st.write(f"📍 Deployment: {deployment_id}")
            st.write(f"🎯 Detection: {detection_conf:.3f}")
            
            if classification_label and classification_label != 'None':
                st.write(f"🏷️ Species: {classification_label}")
                st.write(f"📊 Confidence: {classification_conf:.3f}")
            else:
                st.write("🏷️ No classification")
            
            # Technical details
            if st.expander("Technical Details"):
                bbox_x = selected_row.get('bbox_x', 0)
                bbox_y = selected_row.get('bbox_y', 0)
                bbox_w = selected_row.get('bbox_width', 0)
                bbox_h = selected_row.get('bbox_height', 0)
                
                st.write(f"**Bounding Box:**")
                st.write(f"X: {bbox_x:.4f}, Y: {bbox_y:.4f}")
                st.write(f"W: {bbox_w:.4f}, H: {bbox_h:.4f}")
                
                img_w = selected_row.get('image_width', 'Unknown')
                img_h = selected_row.get('image_height', 'Unknown')
                st.write(f"**Image Size:** {img_w} × {img_h}")
                
                timestamp = selected_row.get('timestamp', 'Unknown')
                st.write(f"**Timestamp:** {timestamp}")

# Performance information
filtered_count = len(grid_response['data']) if 'grid_response' in locals() and grid_response else len(display_df)
st.caption(f"Showing {filtered_count:,} of {len(results_df):,} total detections")