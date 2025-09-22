"""
AddaxAI AggGrid Viewer - Simplified Version

Simple AgGrid viewer showing cropped images with detection labels.
"""

import warnings
import sys

# Force warnings module into sys.modules if it's missing
if 'warnings' not in sys.modules:
    sys.modules['warnings'] = warnings

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode
import base64
import os
from PIL import Image, ImageDraw
import io
import pandas as pd

# Page config
st.set_page_config(layout="wide")
st.title("AggGrid Viewer - Simple")

# Check if detection results are available
if "results_detections" not in st.session_state:
    st.error("No detection results found. Please ensure detection data is loaded.")
    st.stop()

# Load data
df = st.session_state["results_detections"]

if df.empty:
    st.warning("No detection results found.")
    st.stop()

# Constants
THUMBNAIL_SIZE = 100
IMAGE_BACKGROUND_COLOR = (220, 227, 232)
IMAGE_PADDING_PERCENT = 0.01
IMAGE_PADDING_MIN = 10

# Pagination
DEFAULT_PAGE_SIZE = 20
PAGE_SIZE_OPTIONS = [20, 50, 100]

# Initialize pagination state
if 'aggrid_page_size' not in st.session_state:
    st.session_state.aggrid_page_size = DEFAULT_PAGE_SIZE
if 'aggrid_current_page' not in st.session_state:
    st.session_state.aggrid_current_page = 1

# Calculate total pages
total_rows = len(df)
total_pages = max(1, (total_rows + st.session_state.aggrid_page_size - 1) // st.session_state.aggrid_page_size)

# Ensure current page is valid
if st.session_state.aggrid_current_page > total_pages:
    st.session_state.aggrid_current_page = total_pages

# Calculate row indices for current page (needed for data slicing)
start_idx = (st.session_state.aggrid_current_page - 1) * st.session_state.aggrid_page_size
end_idx = min(start_idx + st.session_state.aggrid_page_size, total_rows)

def image_to_base64_url(image_path, bbox_data, max_size=(100, 100)):
    """Convert image to base64 with cropping and red border."""
    try:
        if not image_path or not os.path.exists(image_path):
            # Return a placeholder image
            return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='
        
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Crop to bounding box if available
            if bbox_data and all(pd.notna([bbox_data['x'], bbox_data['y'], 
                                           bbox_data['width'], bbox_data['height']])):
                img_width, img_height = img.size
                x = int(bbox_data['x'] * img_width)
                y = int(bbox_data['y'] * img_height)
                w = int(bbox_data['width'] * img_width)
                h = int(bbox_data['height'] * img_height)
                
                # Calculate padding
                padding_x = max(IMAGE_PADDING_MIN, int(w * IMAGE_PADDING_PERCENT))
                padding_y = max(IMAGE_PADDING_MIN, int(h * IMAGE_PADDING_PERCENT))
                
                # Apply padding
                x1_padded = x - padding_x
                y1_padded = y - padding_y
                x2_padded = x + w + padding_x
                y2_padded = y + h + padding_y
                
                # Make square
                padded_width = x2_padded - x1_padded
                padded_height = y2_padded - y1_padded
                square_size = max(padded_width, padded_height)
                
                center_x = x1_padded + padded_width // 2
                center_y = y1_padded + padded_height // 2
                
                x1_square = center_x - square_size // 2
                y1_square = center_y - square_size // 2
                x2_square = x1_square + square_size
                y2_square = y1_square + square_size
                
                # Crop within bounds
                x1_square = max(0, x1_square)
                y1_square = max(0, y1_square)
                x2_square = min(img_width, x2_square)
                y2_square = min(img_height, y2_square)
                
                if x2_square > x1_square and y2_square > y1_square:
                    img = img.crop((x1_square, y1_square, x2_square, y2_square))
                    
                    # Store info for red border
                    bbox_x_in_crop = x - x1_square
                    bbox_y_in_crop = y - y1_square
                    crop_info = {
                        'bbox_x_in_crop': bbox_x_in_crop,
                        'bbox_y_in_crop': bbox_y_in_crop,
                        'bbox_w': w,
                        'bbox_h': h,
                        'crop_size': square_size
                    }
                else:
                    crop_info = None
            else:
                crop_info = None
            
            # Create thumbnail
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            thumbnail_size = img.size
            
            # Draw red border if we have crop info
            if crop_info:
                scale_x = thumbnail_size[0] / crop_info['crop_size']
                scale_y = thumbnail_size[1] / crop_info['crop_size']
                
                bbox_x_thumb = crop_info['bbox_x_in_crop'] * scale_x
                bbox_y_thumb = crop_info['bbox_y_in_crop'] * scale_y
                bbox_w_thumb = crop_info['bbox_w'] * scale_x
                bbox_h_thumb = crop_info['bbox_h'] * scale_y
                
                draw = ImageDraw.Draw(img)
                x1 = int(max(0, bbox_x_thumb))
                y1 = int(max(0, bbox_y_thumb))
                x2 = int(min(thumbnail_size[0]-1, bbox_x_thumb + bbox_w_thumb))
                y2 = int(min(thumbnail_size[1]-1, bbox_y_thumb + bbox_h_thumb))
                
                draw.rectangle([x1, y1, x2, y2], outline='red', width=1)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_data}"
    except:
        return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='

# Initialize thumbnail cache for current page only
cache_key = f"page_{st.session_state.aggrid_current_page}_size_{st.session_state.aggrid_page_size}"

# Always clear cache and keep only current page
if 'aggrid_thumbnail_cache' not in st.session_state:
    st.session_state.aggrid_thumbnail_cache = {}

# If we're on a different page or page size, clear the entire cache
if 'aggrid_last_cache_key' not in st.session_state or st.session_state.aggrid_last_cache_key != cache_key:
    st.session_state.aggrid_thumbnail_cache = {}  # Clear all cached pages
    st.session_state.aggrid_last_cache_key = cache_key

# Get data for current page only
df_page = df.iloc[start_idx:end_idx].copy()

# Process images for current page only
with st.spinner("Processing images for current page..."):
    # Create display dataframe with all columns
    display_data = []
    image_urls = []
    
    # Check if thumbnails for this page are cached
    if cache_key in st.session_state.aggrid_thumbnail_cache:
        # Use cached thumbnails
        image_urls = st.session_state.aggrid_thumbnail_cache[cache_key]
    else:
        # Generate thumbnails for this page
        for idx, row in df_page.iterrows():
            bbox_data = {
                'x': row.get('bbox_x'),
                'y': row.get('bbox_y'),
                'width': row.get('bbox_width'),
                'height': row.get('bbox_height')
            }
            image_url = image_to_base64_url(row.get('absolute_path'), bbox_data)
            image_urls.append(image_url)
        
        # Cache the thumbnails for this page
        st.session_state.aggrid_thumbnail_cache[cache_key] = image_urls
    
    # Build display data with cached or generated images
    for (idx, row), image_url in zip(df_page.iterrows(), image_urls):
        # Add all columns in the same order as explore_results.py
        # Use None for missing numeric values (AgGrid will display as empty)
        display_data.append({
            'image': image_url,  # Image column (was image_url in explore_results)
            'relative_path': row.get('relative_path') if pd.notna(row.get('relative_path')) else '',
            'detection_label': row.get('detection_label') if pd.notna(row.get('detection_label')) else '',
            'detection_confidence': round(row.get('detection_confidence'), 2) if pd.notna(row.get('detection_confidence')) else None,
            'classification_label': row.get('classification_label') if pd.notna(row.get('classification_label')) else '',
            'classification_confidence': round(row.get('classification_confidence'), 2) if pd.notna(row.get('classification_confidence')) else None,
            'timestamp': row.get('timestamp') if pd.notna(row.get('timestamp')) else '',
            'project_id': row.get('project_id') if pd.notna(row.get('project_id')) else '',
            'location_id': row.get('location_id') if pd.notna(row.get('location_id')) else '',
            'deployment_id': row.get('deployment_id') if pd.notna(row.get('deployment_id')) else '',
            'bbox_x': round(row.get('bbox_x'), 4) if pd.notna(row.get('bbox_x')) else None,
            'bbox_y': round(row.get('bbox_y'), 4) if pd.notna(row.get('bbox_y')) else None,
            'bbox_width': round(row.get('bbox_width'), 4) if pd.notna(row.get('bbox_width')) else None,
            'bbox_height': round(row.get('bbox_height'), 4) if pd.notna(row.get('bbox_height')) else None,
            'image_width': row.get('image_width') if pd.notna(row.get('image_width')) else None,
            'image_height': row.get('image_height') if pd.notna(row.get('image_height')) else None,
            'latitude': round(row.get('latitude'), 6) if pd.notna(row.get('latitude')) else None,
            'longitude': round(row.get('longitude'), 6) if pd.notna(row.get('longitude')) else None,
            'detection_model_id': row.get('detection_model_id') if pd.notna(row.get('detection_model_id')) else '',
            'classification_model_id': row.get('classification_model_id') if pd.notna(row.get('classification_model_id')) else ''
        })
    
    display_df = pd.DataFrame(display_data)

# Configure AgGrid
gb = GridOptionsBuilder.from_dataframe(display_df)

# Configure image column - use DOM element creation with click handling
image_jscode = JsCode("""
    class ImageRenderer {
        init(params) {
            const img = document.createElement('img');
            if (params.value) {
                img.src = params.value;
                img.style.width = '500px';
                img.style.height = '500px';
                img.style.objectFit = 'contain';
                img.style.border = '1px solid #ddd';
                img.style.cursor = 'pointer';
                
                // Add click handler to the image
                img.addEventListener('click', (e) => {
                    e.stopPropagation(); // Prevent row selection
                    // Select the row when image is clicked
                    params.node.setSelected(true);
                    // Store that this was an image click
                    window.lastImageClick = true;
                });
            }
            this.eGui = document.createElement('div');
            this.eGui.appendChild(img);
        }
        
        getGui() {
            return this.eGui;
        }
    }
""")

gb.configure_column(
    "image",
    headerName="Image",
    cellRenderer=image_jscode,
    width=520,
    autoHeight=True
)

# Configure all columns with proper names and widths
gb.configure_column("relative_path", headerName="File", width=150)
gb.configure_column("detection_label", headerName="Detection", width=100)
gb.configure_column("detection_confidence", headerName="Det. Conf", width=100)
gb.configure_column("classification_label", headerName="Species", width=100)
gb.configure_column("classification_confidence", headerName="Class. Conf", width=100)
gb.configure_column("timestamp", headerName="Timestamp", width=150)
gb.configure_column("project_id", headerName="Project", width=100, hide=True)  # Hidden like in explore_results
gb.configure_column("location_id", headerName="Location", width=100)
gb.configure_column("deployment_id", headerName="Deployment", width=100)
gb.configure_column("bbox_x", headerName="BBox X", width=80)
gb.configure_column("bbox_y", headerName="BBox Y", width=80)
gb.configure_column("bbox_width", headerName="BBox W", width=80)
gb.configure_column("bbox_height", headerName="BBox H", width=80)
gb.configure_column("image_width", headerName="Img W", width=80)
gb.configure_column("image_height", headerName="Img H", width=80)
gb.configure_column("latitude", headerName="Lat", width=100)
gb.configure_column("longitude", headerName="Lon", width=100)
gb.configure_column("detection_model_id", headerName="Det Model", width=120)
gb.configure_column("classification_model_id", headerName="Class Model", width=120)

# Set column order to match explore_results.py
column_order = [
    'image',
    'relative_path',
    'detection_label',
    'detection_confidence',
    'classification_label', 
    'classification_confidence',
    'timestamp',
    'project_id',  # Hidden
    'location_id',
    'deployment_id',
    'bbox_x',
    'bbox_y',
    'bbox_width',
    'bbox_height',
    'image_width',
    'image_height',
    'latitude',
    'longitude',
    'detection_model_id',
    'classification_model_id'
]

# Don't use AgGrid pagination since we're doing manual pagination
gb.configure_default_column(resizable=True, sortable=True, filter=True)

# Configure row selection
gb.configure_selection(selection_mode='single', use_checkbox=True)

# Build grid options
grid_options = gb.build()

# Reorder columns after building
if 'columnDefs' in grid_options:
    ordered_cols = []
    for col_name in column_order:
        for col in grid_options['columnDefs']:
            if col.get('field') == col_name:
                ordered_cols.append(col)
                break
    grid_options['columnDefs'] = ordered_cols

# ═══════════════════════════════════════════════════════════════════════════════
# ROW SELECTION DISPLAY (ABOVE TABLE)
# ═══════════════════════════════════════════════════════════════════════════════

# Create placeholder for selected row info that will be updated after grid response
selected_row_placeholder = st.empty()

# Calculate grid height based on actual rows on this page
actual_rows = len(display_df)
row_height = 520  # Height per row for images
header_height = 40  # Header height
grid_height = min(800, (actual_rows * row_height) + header_height)  # Max 800px

# Display grid without built-in pagination
grid_response = AgGrid(
    display_df,
    gridOptions=grid_options,
    height=grid_height,
    allow_unsafe_jscode=True,
    theme='streamlit',
    fit_columns_on_grid_load=False,  # Don't auto-fit columns
    reload_data=False,  # Don't reload data on every interaction
    update_mode='SELECTION_CHANGED'  # Only update on selection changes
)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW SELECTION HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

# Handle row selection and update the placeholder above the table
# Show details for any row selection (clicking image is preferred but any column works)
if (grid_response['selected_rows'] is not None and len(grid_response['selected_rows']) > 0):
    # grid_response['selected_rows'] is a DataFrame, get the first row as a Series
    selected_row = grid_response['selected_rows'].iloc[0]
    
    with selected_row_placeholder.container():
        image_name = selected_row.get('relative_path', 'N/A')
        det_label = selected_row.get('detection_label', 'N/A')
        det_conf = selected_row.get('detection_confidence', 0)
        
        if det_conf and det_conf != 0:
            st.info(f"Selected image {image_name} with detection {det_label} at conf {det_conf:.2f}")
        else:
            st.info(f"Selected image {image_name} with detection {det_label} at conf N/A")
else:
    # Clear the placeholder when no row is selected
    selected_row_placeholder.empty()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGINATION CONTROLS (BOTTOM) - Same as explore_results.py
# ═══════════════════════════════════════════════════════════════════════════════

# Pagination controls at the bottom
bottom_menu = st.columns((4, 1, 1))

with bottom_menu[0]:
    st.markdown(f"Page **{st.session_state.aggrid_current_page}** of **{total_pages}** "
                f"(Showing rows {start_idx + 1}-{end_idx} of {total_rows:,})")

with bottom_menu[1]:
    new_page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        step=1,
        value=st.session_state.aggrid_current_page,
        key="aggrid_page_input"
    )
    if new_page != st.session_state.aggrid_current_page:
        st.session_state.aggrid_current_page = new_page
        st.rerun()

with bottom_menu[2]:
    new_page_size = st.selectbox(
        "Page Size",
        options=PAGE_SIZE_OPTIONS,
        index=PAGE_SIZE_OPTIONS.index(st.session_state.aggrid_page_size),
        key="aggrid_page_size_input"
    )
    if new_page_size != st.session_state.aggrid_page_size:
        st.session_state.aggrid_page_size = new_page_size
        st.session_state.aggrid_current_page = 1  # Reset to first page
        st.rerun()