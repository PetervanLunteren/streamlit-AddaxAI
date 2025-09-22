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
from PIL import Image, ImageDraw, ImageFilter
import io
import pandas as pd
from datetime import datetime, timedelta
from utils.common import load_vars, update_vars, get_session_var, set_session_var
from utils.explore_results_utils import image_viewer_modal
from st_modal import Modal

def parse_timestamps(timestamp_series):
    """Parse timestamps in EXIF format: 2013:01:17 13:05:40"""
    return pd.to_datetime(timestamp_series, format='%Y:%m:%d %H:%M:%S')

# Constants
# Row heights and image sizes
ROW_HEIGHT_OPTIONS = {
    "small": 30,   # Small thumbnail
    "medium": 100,  # Medium size
    "large": 250   # Large size
}

# Image size ratio (width = height * ratio)
IMAGE_SIZE_RATIO = 1.5

# Image column widths (calculated from row height * ratio)
IMAGE_COLUMN_WIDTHS = {
    size: int(height * IMAGE_SIZE_RATIO) 
    for size, height in ROW_HEIGHT_OPTIONS.items()
}

# Thumbnail generation settings
IMAGE_BACKGROUND_COLOR = (220, 227, 232)
IMAGE_PADDING_PIXELS = 100  # Padding around bbox in pixels (new padding system)
IMAGE_BLUR_RADIUS = 15  # Blur radius for background extension
IMAGE_CORNER_RADIUS = 20  # Corner radius for rounded thumbnails (pixels)
IMAGE_BBOX_COLOR = 'red'  # Color for bounding box outline
IMAGE_QUALITY = 85
DEFAULT_SIZE_OPTION = "medium"  # Default size selection

# Legacy padding settings (no longer used)
IMAGE_PADDING_PERCENT = 0.01
IMAGE_PADDING_MIN = 10

# Page config
st.set_page_config(layout="wide")
st.title("AggGrid Viewer - Simple")

# Load filter settings from config
filter_config = load_vars("explore_results")
saved_settings = filter_config.get("aggrid_settings", {})

# Check if detection results are available
if "results_detections" not in st.session_state:
    st.error("No detection results found. Please ensure detection data is loaded.")
    st.stop()

# Load data
df = st.session_state["results_detections"]

if df.empty:
    st.warning("No detection results found.")
    st.stop()

# Image size controls both row height and image column width
image_size_options = {
    size: {"height": height, "width": IMAGE_COLUMN_WIDTHS[size]}
    for size, height in ROW_HEIGHT_OPTIONS.items()
}

# Controls row
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)

with col8:
    # Simple date filter popover
    with st.popover(":material/filter_alt:", help="Date Filter", use_container_width=True):
        with st.form("date_filter_form"):
            
            # Get min and max dates from the full data
            if 'timestamp' in df.columns and df['timestamp'].notna().any():
                df_timestamps = parse_timestamps(df['timestamp'])
                min_date = df_timestamps.min()
                max_date = df_timestamps.max()
                
                # If we can't parse dates, use defaults
                if pd.isna(min_date) or pd.isna(max_date):
                    min_date = datetime.now() - timedelta(days=365)
                    max_date = datetime.now()
            else:
                # Default to last year if no timestamp data
                min_date = datetime.now() - timedelta(days=365)
                max_date = datetime.now()
            
            # Get saved date range or use full range as default
            saved_start = saved_settings.get('date_start')
            saved_end = saved_settings.get('date_end')
            
            # Handle None values by using full range
            if saved_start is None:
                saved_start = min_date.date()
            elif isinstance(saved_start, str):
                saved_start = pd.to_datetime(saved_start).date()
                
            if saved_end is None:
                saved_end = max_date.date()
            elif isinstance(saved_end, str):
                saved_end = pd.to_datetime(saved_end).date()
            
            # Date range slider
            date_range = st.slider(
                "Date range",
                min_value=min_date.date(),
                max_value=max_date.date(),
                value=(saved_start, saved_end),
                format="YYYY-MM-DD"
            )
            
            # Detection confidence range
            saved_det_conf_min = saved_settings.get('det_conf_min', 0.0)
            saved_det_conf_max = saved_settings.get('det_conf_max', 1.0)
            det_conf_range = st.slider(
                "Detection confidence range",
                min_value=0.0,
                max_value=1.0,
                value=(saved_det_conf_min, saved_det_conf_max),
                step=0.01,
                format="%.2f"
            )
            
            # Classification confidence range  
            saved_cls_conf_min = saved_settings.get('cls_conf_min', 0.0)
            saved_cls_conf_max = saved_settings.get('cls_conf_max', 1.0)
            cls_conf_range = st.slider(
                "Classification confidence range",
                min_value=0.0,
                max_value=1.0,
                value=(saved_cls_conf_min, saved_cls_conf_max),
                step=0.01,
                format="%.2f"
            )
            
            # Buttons
            col1, col2 = st.columns([1, 1])
            with col1:
                apply_filter = st.form_submit_button("Apply", use_container_width=True, type="primary")
            with col2:
                clear_all = st.form_submit_button("Clear All", use_container_width=True)
            
            if apply_filter:
                # Save filter settings to config
                filter_settings = {
                    "aggrid_settings": {
                        "date_start": date_range[0].isoformat(),
                        "date_end": date_range[1].isoformat(),
                        "det_conf_min": det_conf_range[0],
                        "det_conf_max": det_conf_range[1],
                        "cls_conf_min": cls_conf_range[0],
                        "cls_conf_max": cls_conf_range[1],
                        "image_size": saved_settings.get('image_size', 'medium')
                    }
                }
                update_vars("explore_results", filter_settings)
                
                # Clear caches
                if 'aggrid_thumbnail_cache' in st.session_state:
                    del st.session_state['aggrid_thumbnail_cache']
                if 'aggrid_last_cache_key' in st.session_state:
                    del st.session_state['aggrid_last_cache_key']
                if 'results_detections_filtered' in st.session_state:
                    del st.session_state['results_detections_filtered']
                
                # Reset to first page
                st.session_state.aggrid_current_page = 1
                
                st.rerun()
            
            if clear_all:
                # Clear all filters to default ranges
                filter_settings = {
                    "aggrid_settings": {
                        "date_start": min_date.date().isoformat(),
                        "date_end": max_date.date().isoformat(), 
                        "det_conf_min": 0.0,
                        "det_conf_max": 1.0,
                        "cls_conf_min": 0.0,
                        "cls_conf_max": 1.0,
                        "image_size": saved_settings.get('image_size', 'medium')
                    }
                }
                update_vars("explore_results", filter_settings)
                
                # Clear caches
                if 'aggrid_thumbnail_cache' in st.session_state:
                    del st.session_state['aggrid_thumbnail_cache']
                if 'aggrid_last_cache_key' in st.session_state:
                    del st.session_state['aggrid_last_cache_key']
                if 'results_detections_filtered' in st.session_state:
                    del st.session_state['results_detections_filtered']
                
                # Reset to first page
                st.session_state.aggrid_current_page = 1
                
                st.rerun()

with col9:
    # Download popover with material icon
    with st.popover(":material/download:", help="Download Data", use_container_width=True):
        # Get the filtered dataframe from session state
        df_to_download = st.session_state.get('results_detections_filtered', df)
        
        # Generate timestamp for filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Download formats label and container
        st.markdown('<p style="font-size: 14px; margin-bottom: 4px;">Download formats</p>', unsafe_allow_html=True)
        with st.container(border=True):
            # Four download buttons in one row
            download_col1, download_col2, download_col3, download_col4 = st.columns(4)
        
        with download_col1:
            # XLSX Download Button
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_to_download.to_excel(writer, index=False, sheet_name='Filtered_Data')
            xlsx_data = output.getvalue()
            
            st.download_button(
                label="XLSX",
                data=xlsx_data,
                file_name=f"addaxai-export-{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch"
            )
        
        with download_col2:
            # CSV Download Button
            csv_data = df_to_download.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="CSV",
                data=csv_data,
                file_name=f"addaxai-export-{timestamp}.csv",
                mime="text/csv",
                width="stretch"
            )
        
        with download_col3:
            # TSV Download Button
            tsv_data = df_to_download.to_csv(index=False, sep='\t').encode('utf-8')
            
            st.download_button(
                label="TSV",
                data=tsv_data,
                file_name=f"addaxai-export-{timestamp}.tsv",
                mime="text/tab-separated-values",
                width="stretch"
            )
        
        with download_col4:
            # JSON Download Button
            json_data = df_to_download.to_json(orient='records', indent=2).encode('utf-8')
            
            st.download_button(
                label="JSON",
                data=json_data,
                file_name=f"addaxai-export-{timestamp}.json",
                mime="application/json",
                width="stretch"
            )

with col10:
    # Settings popover with material icon
    with st.popover(":material/settings:", help="Image Size Settings", use_container_width=True):        
        # Get the saved or default value for the segmented control
        default_image_size = saved_settings.get('image_size', DEFAULT_SIZE_OPTION)
        
        selected_size = st.segmented_control(
            "Image Size",
            options=list(image_size_options.keys()),
            default=default_image_size,
            key="aggrid_image_size_control",
            width="stretch"
        )
        
        # Save the setting when changed
        if selected_size != default_image_size:
            filter_settings = {
                "aggrid_settings": {
                    "date_start": saved_settings.get('date_start', ''),
                    "date_end": saved_settings.get('date_end', ''),
                    "image_size": selected_size
                }
            }
            update_vars("explore_results", filter_settings)
            
            # Clear the thumbnail cache when size changes
            if 'aggrid_thumbnail_cache' in st.session_state:
                del st.session_state['aggrid_thumbnail_cache']
            if 'aggrid_last_cache_key' in st.session_state:
                del st.session_state['aggrid_last_cache_key']
            st.rerun()

# Get current size configuration
selected_size = st.session_state.get("aggrid_image_size_control", DEFAULT_SIZE_OPTION)
size_config = image_size_options.get(selected_size, image_size_options[DEFAULT_SIZE_OPTION])
current_row_height = size_config["height"]
current_image_width = size_config["width"]

# Create or get filtered dataframe
if 'results_detections_filtered' not in st.session_state:
    # Start with full dataframe
    filtered_df = df.copy()
    
    # Apply date filter if set
    if ('date_start' in saved_settings and 'date_end' in saved_settings and 
        saved_settings['date_start'] is not None and saved_settings['date_end'] is not None):
        if 'timestamp' in filtered_df.columns:
            df_timestamps = parse_timestamps(filtered_df['timestamp'])
            date_start = pd.to_datetime(saved_settings['date_start'])
            date_end = pd.to_datetime(saved_settings['date_end'])
            filtered_df = filtered_df[
                (df_timestamps >= date_start) & 
                (df_timestamps <= date_end)
            ].copy()
    
    # Apply detection confidence filter if set
    if ('det_conf_min' in saved_settings and 'det_conf_max' in saved_settings):
        if 'detection_confidence' in filtered_df.columns:
            det_conf_min = saved_settings['det_conf_min']
            det_conf_max = saved_settings['det_conf_max']
            filtered_df = filtered_df[
                (filtered_df['detection_confidence'] >= det_conf_min) & 
                (filtered_df['detection_confidence'] <= det_conf_max)
            ].copy()
    
    # Apply classification confidence filter if set
    if ('cls_conf_min' in saved_settings and 'cls_conf_max' in saved_settings):
        if 'classification_confidence' in filtered_df.columns:
            cls_conf_min = saved_settings['cls_conf_min']
            cls_conf_max = saved_settings['cls_conf_max']
            filtered_df = filtered_df[
                (filtered_df['classification_confidence'] >= cls_conf_min) & 
                (filtered_df['classification_confidence'] <= cls_conf_max)
            ].copy()
    
    # Store filtered dataframe in session state
    st.session_state['results_detections_filtered'] = filtered_df

# Use filtered dataframe for all operations
df_filtered = st.session_state['results_detections_filtered']

if df_filtered.empty:
    st.warning("No results match the current filters.")
    st.stop()

# Pagination
DEFAULT_PAGE_SIZE = 20
PAGE_SIZE_OPTIONS = [20, 50, 100]

# Initialize pagination state
if 'aggrid_page_size' not in st.session_state:
    st.session_state.aggrid_page_size = DEFAULT_PAGE_SIZE
if 'aggrid_current_page' not in st.session_state:
    st.session_state.aggrid_current_page = 1

# Calculate total pages using filtered dataframe
total_rows = len(df_filtered)
total_pages = max(1, (total_rows + st.session_state.aggrid_page_size - 1) // st.session_state.aggrid_page_size)

# Ensure current page is valid
if st.session_state.aggrid_current_page > total_pages:
    st.session_state.aggrid_current_page = total_pages

# Calculate row indices for current page (needed for data slicing)
start_idx = (st.session_state.aggrid_current_page - 1) * st.session_state.aggrid_page_size
end_idx = min(start_idx + st.session_state.aggrid_page_size, total_rows)

def image_to_base64_url(image_path, bbox_data, max_size):
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
                
                # New padding procedure: max(bbox_width, bbox_height) + padding pixels
                target_size = max(w, h) + (IMAGE_PADDING_PIXELS * 2)
                
                # Calculate bbox center
                bbox_center_x = x + w // 2
                bbox_center_y = y + h // 2
                
                # Try to center the square crop around bbox
                x1_target = bbox_center_x - target_size // 2
                y1_target = bbox_center_y - target_size // 2
                x2_target = x1_target + target_size
                y2_target = y1_target + target_size
                
                # Edge handling: shift padding if hitting boundaries
                if x1_target < 0:
                    # Shift right
                    shift_right = -x1_target
                    x1_target = 0
                    x2_target = min(img_width, target_size)
                elif x2_target > img_width:
                    # Shift left
                    shift_left = x2_target - img_width
                    x2_target = img_width
                    x1_target = max(0, img_width - target_size)
                
                if y1_target < 0:
                    # Shift down
                    shift_down = -y1_target
                    y1_target = 0
                    y2_target = min(img_height, target_size)
                elif y2_target > img_height:
                    # Shift up
                    shift_up = y2_target - img_height
                    y2_target = img_height
                    y1_target = max(0, img_height - target_size)
                
                # Get the actual crop size we can achieve
                actual_crop_width = x2_target - x1_target
                actual_crop_height = y2_target - y1_target
                
                # Check if we can achieve target square size within image
                if actual_crop_width == target_size and actual_crop_height == target_size:
                    # Perfect fit - crop directly
                    img = img.crop((x1_target, y1_target, x2_target, y2_target))
                    
                    # Store info for red border
                    bbox_x_in_crop = x - x1_target
                    bbox_y_in_crop = y - y1_target
                    crop_info = {
                        'bbox_x_in_crop': bbox_x_in_crop,
                        'bbox_y_in_crop': bbox_y_in_crop,
                        'bbox_w': w,
                        'bbox_h': h,
                        'crop_size': target_size
                    }
                else:
                    # Need background extension - use blurred crop as background
                    cropped_img = img.crop((x1_target, y1_target, x2_target, y2_target))
                    
                    # Create blurred version of the crop for background
                    blurred_crop = cropped_img.copy()
                    blurred_crop = blurred_crop.filter(ImageFilter.GaussianBlur(radius=IMAGE_BLUR_RADIUS))
                    
                    # Resize blurred crop to fill target square (may stretch/distort)
                    blurred_background = blurred_crop.resize((target_size, target_size), Image.Resampling.LANCZOS)
                    
                    # Calculate position to center original crop on blurred background
                    paste_x = (target_size - actual_crop_width) // 2
                    paste_y = (target_size - actual_crop_height) // 2
                    
                    # Paste original sharp crop onto blurred background
                    blurred_background.paste(cropped_img, (paste_x, paste_y))
                    img = blurred_background
                    
                    # Store info for red border (adjusted for blurred background)
                    bbox_x_in_crop = (x - x1_target) + paste_x
                    bbox_y_in_crop = (y - y1_target) + paste_y
                    crop_info = {
                        'bbox_x_in_crop': bbox_x_in_crop,
                        'bbox_y_in_crop': bbox_y_in_crop,
                        'bbox_w': w,
                        'bbox_h': h,
                        'crop_size': target_size
                    }
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
                
                draw.rectangle([x1, y1, x2, y2], outline=IMAGE_BBOX_COLOR, width=1)
            
            # Apply rounded corners
            if IMAGE_CORNER_RADIUS > 0:
                # Create a mask for rounded corners
                mask = Image.new('L', img.size, 0)
                mask_draw = ImageDraw.Draw(mask)
                
                # Scale corner radius to thumbnail size
                scaled_radius = int(IMAGE_CORNER_RADIUS * (thumbnail_size[0] / 250))  # Scale based on size
                scaled_radius = max(5, min(scaled_radius, min(thumbnail_size) // 4))  # Reasonable bounds
                
                # Draw rounded rectangle mask
                mask_draw.rounded_rectangle(
                    [(0, 0), (thumbnail_size[0]-1, thumbnail_size[1]-1)],
                    radius=scaled_radius,
                    fill=255
                )
                
                # Apply mask to create rounded corners
                output = Image.new('RGBA', img.size, (255, 255, 255, 0))
                output.paste(img, (0, 0))
                output.putalpha(mask)
                
                # Convert back to RGB with white background
                final = Image.new('RGB', img.size, 'white')
                final.paste(output, (0, 0), output)
                img = final
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_data}"
    except:
        return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='

# Initialize thumbnail cache for current page only
# Include image size in cache key so cache invalidates when image size changes
cache_key = f"page_{st.session_state.aggrid_current_page}_pagesize_{st.session_state.aggrid_page_size}_imgsize_{selected_size}"

# Always clear cache and keep only current page
if 'aggrid_thumbnail_cache' not in st.session_state:
    st.session_state.aggrid_thumbnail_cache = {}

# If we're on a different page or page size, clear the entire cache
if 'aggrid_last_cache_key' not in st.session_state or st.session_state.aggrid_last_cache_key != cache_key:
    st.session_state.aggrid_thumbnail_cache = {}  # Clear all cached pages
    st.session_state.aggrid_last_cache_key = cache_key

# Get data for current page only from filtered dataframe
df_page = df_filtered.iloc[start_idx:end_idx].copy()

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
            # Use the current selected image size for thumbnail generation
            thumbnail_size = (current_row_height, current_row_height)
            image_url = image_to_base64_url(row.get('absolute_path'), bbox_data, max_size=thumbnail_size)
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
            'detection_confidence': round(float(row.get('detection_confidence')), 2) if pd.notna(row.get('detection_confidence')) else None,
            'classification_label': row.get('classification_label') if pd.notna(row.get('classification_label')) else '',
            'classification_confidence': round(float(row.get('classification_confidence')), 2) if pd.notna(row.get('classification_confidence')) else None,
            'timestamp': parse_timestamps(pd.Series([row.get('timestamp')])).iloc[0].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row.get('timestamp')) else '',
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
image_jscode = JsCode(f"""
    class ImageRenderer {{
        init(params) {{
            const img = document.createElement('img');
            if (params.value) {{
                img.src = params.value;
                img.style.width = '{current_row_height}px';
                img.style.height = '{current_row_height}px';
                img.style.objectFit = 'contain';
                img.style.border = 'none';
                img.style.cursor = 'pointer';
                
                // Add click handler to the image
                img.addEventListener('click', (e) => {{
                    e.stopPropagation(); // Prevent row selection
                    // Select the row when image is clicked
                    params.node.setSelected(true);
                    // Store that this was an image click
                    window.lastImageClick = true;
                }});
            }}
            this.eGui = document.createElement('div');
            this.eGui.appendChild(img);
        }}
        
        getGui() {{
            return this.eGui;
        }}
    }}
""")

gb.configure_column(
    "image",
    headerName="Image",
    cellRenderer=image_jscode,
    width=current_image_width + 20,  # Add padding
    autoHeight=True
)

# Configure all columns with proper names and widths
gb.configure_column("relative_path", headerName="File", width=150, filter=False, sortable=False)
gb.configure_column("detection_label", headerName="Detection", width=100, filter=False, sortable=False)
gb.configure_column("detection_confidence", headerName="Detection confidence", width=120, filter=False, sortable=False, valueFormatter="x.toFixed(2)", type="numericColumn", headerClass="ag-left-aligned-header")
gb.configure_column("classification_label", headerName="Classification", width=100, filter=False, sortable=False)
gb.configure_column("classification_confidence", headerName="Classification confidence", width=140, filter=False, sortable=False, valueFormatter="x.toFixed(2)", type="numericColumn", headerClass="ag-left-aligned-header")
gb.configure_column("timestamp", headerName="Timestamp", width=150, filter=False, sortable=False)
gb.configure_column("project_id", headerName="Project ID", width=100, hide=True, filter=False, sortable=False)  # Hidden like in explore_results
gb.configure_column("location_id", headerName="Location ID", width=100, filter=False, sortable=False)
gb.configure_column("deployment_id", headerName="Deployment ID", width=110, filter=False, sortable=False)
gb.configure_column("bbox_x", headerName="BBox X", width=80, filter=False, sortable=False, type="numericColumn", headerClass="ag-left-aligned-header", hide=True)
gb.configure_column("bbox_y", headerName="BBox Y", width=80, filter=False, sortable=False, type="numericColumn", headerClass="ag-left-aligned-header", hide=True)
gb.configure_column("bbox_width", headerName="BBox W", width=80, filter=False, sortable=False, type="numericColumn", headerClass="ag-left-aligned-header", hide=True)
gb.configure_column("bbox_height", headerName="BBox H", width=80, filter=False, sortable=False, type="numericColumn", headerClass="ag-left-aligned-header", hide=True)
gb.configure_column("image_width", headerName="Img W", width=80, filter=False, sortable=False, type="numericColumn", headerClass="ag-left-aligned-header", hide=True)
gb.configure_column("image_height", headerName="Img H", width=80, filter=False, sortable=False, type="numericColumn", headerClass="ag-left-aligned-header", hide=True)
gb.configure_column("latitude", headerName="Latitude", width=100, filter=False, sortable=False, type="numericColumn", headerClass="ag-left-aligned-header")
gb.configure_column("longitude", headerName="Longitude", width=100, filter=False, sortable=False, type="numericColumn", headerClass="ag-left-aligned-header")
gb.configure_column("detection_model_id", headerName="Detection model", width=120, filter=False, sortable=False)
gb.configure_column("classification_model_id", headerName="Classification model", width=140, filter=False, sortable=False)

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

# Add custom CSS for header alignment
st.markdown("""
<style>
.ag-header-cell-label {
    justify-content: flex-start !important;
    text-align: left !important;
}
.ag-header-cell {
    text-align: left !important;
}
.ag-left-aligned-header .ag-header-cell-label {
    justify-content: flex-start !important;
    text-align: left !important;
}
.ag-numeric-column .ag-header-cell-label {
    justify-content: flex-start !important;
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)

# Don't use AgGrid pagination since we're doing manual pagination
gb.configure_default_column(
    resizable=True, 
    sortable=False, 
    filter=False,
    cellStyle={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-start'},
    headerClass='ag-left-aligned-header'
)

# Configure row selection
gb.configure_selection(selection_mode='single', use_checkbox=True)

# Configure row height to match image height + padding
gb.configure_grid_options(rowHeight=current_row_height + 3)

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
# IMAGE VIEWER MODAL
# ═══════════════════════════════════════════════════════════════════════════════

# Modal for image viewer - only create when needed
if get_session_var("explore_results", "show_modal_image_viewer", False):
    modal_image_viewer = Modal(
        title="#### Image Viewer", 
        key="image_viewer", 
        show_close_button=False
    )
    with modal_image_viewer.container():
        image_viewer_modal()

# ═══════════════════════════════════════════════════════════════════════════════
# ROW SELECTION DISPLAY (ABOVE TABLE)
# ═══════════════════════════════════════════════════════════════════════════════

# Create placeholder for selected row info that will be updated after grid response
selected_row_placeholder = st.empty()

# Calculate grid height based on actual rows on this page
actual_rows = len(display_df)
# AgGrid uses exact image height as row height with small padding
row_height = current_row_height + 3  # Small padding for AgGrid
header_height = 35  # Actual header height
grid_height = (actual_rows * row_height) + header_height  # Exact size based on rows

# Display grid without built-in pagination
grid_response = AgGrid(
    display_df,
    gridOptions=grid_options,
    height=grid_height,
    allow_unsafe_jscode=True,
    theme='streamlit',
    fit_columns_on_grid_load=False,  # Don't auto-fit columns
    update_on=['selectionChanged']  # Only update on selection changes
)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW SELECTION HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

# Handle row selection and trigger modal
# Show modal for any row selection (clicking image is preferred but any column works)
if (grid_response['selected_rows'] is not None and len(grid_response['selected_rows']) > 0):
    # grid_response['selected_rows'] is a DataFrame, get the first row as a Series
    selected_row = grid_response['selected_rows'].iloc[0]
    
    # Find the index of the selected row in the filtered dataframe
    # We need to match by a unique identifier since display_df is just the current page
    selected_path = selected_row.get('relative_path')
    
    # Find the actual index in the full filtered dataframe
    if selected_path and 'results_detections_filtered' in st.session_state:
        df_filtered = st.session_state['results_detections_filtered']
        matching_rows = df_filtered[df_filtered['relative_path'] == selected_path]
        
        if not matching_rows.empty:
            # Get the index of the first match in the filtered dataframe
            selected_index = matching_rows.index[0]
            # Convert to position in the filtered dataframe
            modal_index = df_filtered.index.get_loc(selected_index)
            
            # Set modal state and open
            set_session_var("explore_results", "modal_current_image_index", modal_index)
            set_session_var("explore_results", "show_modal_image_viewer", True)
            st.rerun()

# Show selection hint when no row is selected
if not (grid_response['selected_rows'] is not None and len(grid_response['selected_rows']) > 0):
    with selected_row_placeholder.container():
        st.info("Click on any row to view the full-resolution image")

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