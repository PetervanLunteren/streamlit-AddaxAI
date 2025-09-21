"""
AddaxAI AggGrid Viewer

Simple AgGrid viewer showing full images as thumbnails with detection labels.
"""

import warnings
import sys

# Force warnings module into sys.modules if it's missing
# This fixes the KeyError: 'warnings' issue with pandas
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

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(layout="wide")

st.title("AggGrid Viewer")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

# Check if detection results are available
if "results_detections" not in st.session_state:
    st.error("No detection results found. Please ensure the application has been started properly and detection data is loaded.")
    st.stop()

# Load detection results dataframe
df = st.session_state["results_detections"]

if df.empty:
    st.warning("No detection results found in the database.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Image processing
THUMBNAIL_SIZE = 100  # Thumbnail size in pixels (100x100 for AgGrid)
IMAGE_BACKGROUND_COLOR = (220, 227, 232)  # #dce3e8 in RGB
IMAGE_PADDING_PERCENT = 0.01  # 1% padding
IMAGE_PADDING_MIN = 10  # Minimum padding in pixels
IMAGE_QUALITY = 85  # JPEG compression quality

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def image_to_base64_url(image_path, bbox_data, max_size=(100, 100)):
    """Convert image file to base64 data URL for AgGrid display, cropped to detection bbox with red border."""
    try:
        if not image_path or not os.path.exists(image_path):
            return None
        
        # Open image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Crop to bounding box if we have valid bbox data
            if bbox_data and all(pd.notna([bbox_data['x'], bbox_data['y'], 
                                           bbox_data['width'], bbox_data['height']])):
                # Convert normalized bbox coordinates to pixels
                img_width, img_height = img.size
                x = int(bbox_data['x'] * img_width)
                y = int(bbox_data['y'] * img_height)
                w = int(bbox_data['width'] * img_width)
                h = int(bbox_data['height'] * img_height)
                
                # Calculate padding: 1% of detection size, minimum 10px
                padding_x = max(IMAGE_PADDING_MIN, int(w * IMAGE_PADDING_PERCENT))
                padding_y = max(IMAGE_PADDING_MIN, int(h * IMAGE_PADDING_PERCENT))
                
                # Apply padding
                x1_padded = x - padding_x
                y1_padded = y - padding_y
                x2_padded = x + w + padding_x
                y2_padded = y + h + padding_y
                
                # Calculate dimensions for square crop
                padded_width = x2_padded - x1_padded
                padded_height = y2_padded - y1_padded
                square_size = max(padded_width, padded_height)
                
                # Center the square crop on the padded detection
                center_x = x1_padded + padded_width // 2
                center_y = y1_padded + padded_height // 2
                
                x1_square = center_x - square_size // 2
                y1_square = center_y - square_size // 2
                x2_square = x1_square + square_size
                y2_square = y1_square + square_size
                
                # Check if we need to add bars (if crop exceeds image boundaries)
                needs_bars = (x1_square < 0 or y1_square < 0 or 
                            x2_square > img_width or y2_square > img_height)
                
                if needs_bars:
                    # Create new image with background color
                    bg_color = IMAGE_BACKGROUND_COLOR
                    
                    # Create canvas with background
                    canvas = Image.new('RGB', (square_size, square_size), bg_color)
                    
                    # Calculate where to paste the original image
                    paste_x = max(0, -x1_square)
                    paste_y = max(0, -y1_square)
                    
                    # Crop the original image to valid bounds
                    crop_x1 = max(0, x1_square)
                    crop_y1 = max(0, y1_square)
                    crop_x2 = min(img_width, x2_square)
                    crop_y2 = min(img_height, y2_square)
                    
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                        canvas.paste(cropped, (paste_x, paste_y))
                    
                    img = canvas
                    
                    # Calculate bounding box position within the square canvas
                    bbox_x_in_crop = x - x1_square
                    bbox_y_in_crop = y - y1_square
                    
                    # Store crop info for bbox drawing after thumbnail creation
                    crop_info = {
                        'bbox_x_in_crop': bbox_x_in_crop,
                        'bbox_y_in_crop': bbox_y_in_crop,
                        'bbox_w': w,
                        'bbox_h': h,
                        'crop_size': square_size
                    }
                else:
                    # Regular crop within bounds
                    x1_square = max(0, x1_square)
                    y1_square = max(0, y1_square)
                    x2_square = min(img_width, x2_square)
                    y2_square = min(img_height, y2_square)
                    
                    if x2_square > x1_square and y2_square > y1_square:
                        img = img.crop((x1_square, y1_square, x2_square, y2_square))
                
                    # Calculate bounding box position within the cropped square image
                    # Original bbox coordinates relative to the cropped square
                    bbox_x_in_crop = x - x1_square
                    bbox_y_in_crop = y - y1_square
                    
                    # Store crop info for bbox drawing after thumbnail creation
                    crop_info = {
                        'bbox_x_in_crop': bbox_x_in_crop,
                        'bbox_y_in_crop': bbox_y_in_crop,
                        'bbox_w': w,
                        'bbox_h': h,
                        'crop_size': square_size
                    }
            else:
                crop_info = None
            
            # Create thumbnail to reduce size
            original_size = img.size
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            thumbnail_size = img.size
            
            # Draw red border around exact bounding box if we have crop info
            if crop_info and bbox_data:
                # Calculate scale factor from original crop to thumbnail
                scale_x = thumbnail_size[0] / crop_info['crop_size']
                scale_y = thumbnail_size[1] / crop_info['crop_size']
                
                # Scale bounding box coordinates to thumbnail size
                bbox_x_thumb = crop_info['bbox_x_in_crop'] * scale_x
                bbox_y_thumb = crop_info['bbox_y_in_crop'] * scale_y
                bbox_w_thumb = crop_info['bbox_w'] * scale_x
                bbox_h_thumb = crop_info['bbox_h'] * scale_y
                
                # Draw red border (1px)
                draw = ImageDraw.Draw(img)
                x1 = int(bbox_x_thumb)
                y1 = int(bbox_y_thumb)
                x2 = int(bbox_x_thumb + bbox_w_thumb)
                y2 = int(bbox_y_thumb + bbox_h_thumb)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, thumbnail_size[0] - 1))
                y1 = max(0, min(y1, thumbnail_size[1] - 1))
                x2 = max(0, min(x2, thumbnail_size[0] - 1))
                y2 = max(0, min(y2, thumbnail_size[1] - 1))
                
                # Draw 1px red rectangle outline
                draw.rectangle([x1, y1, x2, y2], outline='red', width=1)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            
            # Return as data URL
            return f"data:image/jpeg;base64,{img_data}"
    except Exception:
        # Return None silently if image cannot be processed
        return None

# Debug: Show available columns
st.write("Available columns in dataframe:", df.columns.tolist())

# Process images - only keep essential columns
with st.spinner("Processing images..."):
    # Create simplified dataframe with just image and detection label
    display_df = pd.DataFrame()
    
    # Add image thumbnails with bbox cropping
    image_urls = []
    for idx, row in df.iterrows():
        # Prepare bbox data for cropping
        bbox_data = {
            'x': row.get('bbox_x', None),
            'y': row.get('bbox_y', None),
            'width': row.get('bbox_width', None),
            'height': row.get('bbox_height', None)
        }
        # Process image with cropping and red border
        image_url = image_to_base64_url(row.get('absolute_path'), bbox_data, max_size=(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        
        # Debug: Check if we're getting valid URLs
        if idx < 3 and image_url:  # Check first 3 images
            st.write(f"Image {idx}: URL starts with: {image_url[:50]}...")
        
        image_urls.append(image_url)
    
    display_df['image'] = image_urls
    display_df['detection_label'] = df['detection_label'].fillna('No detection')
    display_df['filename'] = df['relative_path']
    display_df['confidence'] = df['detection_confidence'].round(2)
    
    # Debug: Check the display dataframe
    st.write("Display dataframe shape:", display_df.shape)
    st.write("First row of display_df:")
    if not display_df.empty:
        first_row = display_df.iloc[0]
        st.write(f"- filename: {first_row['filename']}")
        st.write(f"- detection_label: {first_row['detection_label']}")
        st.write(f"- image URL length: {len(first_row['image']) if first_row['image'] else 'None'}")
        st.write(f"- image URL starts: {first_row['image'][:100] if first_row['image'] else 'None'}...")

# ═══════════════════════════════════════════════════════════════════════════════
# AGGRID CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Configure AgGrid
gb = GridOptionsBuilder.from_dataframe(display_df)

# Configure image column with custom JS renderer (function-based approach that works)
gb.configure_column(
    "image",
    headerName="Image",
    cellRenderer=JsCode("""
        function(params) {
            console.log('Image renderer called with:', params.value ? params.value.substring(0, 50) + '...' : 'NO VALUE');
            if (params.value) {
                return '<img src="' + params.value + '" style="height:100px;width:100px;object-fit:contain;border:1px solid #ddd;" />';
            } else {
                return "<span>No image</span>";
            }
        }
    """),
    width=120,
    autoHeight=True
)

# Configure other columns
gb.configure_column("detection_label", headerName="Detection", width=150)
gb.configure_column("filename", headerName="File", width=200)
gb.configure_column("confidence", headerName="Confidence", width=100)

# Grid options
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
gb.configure_default_column(resizable=True, sortable=True, filter=True)
gb.configure_grid_options(domLayout='normal')

grid_options = gb.build()

# Debug: Check grid options
st.write("Grid options columns:", [col.get('field') for col in grid_options.get('columnDefs', [])])

# Test if JS rendering works at all
st.write("--- TEST GRID ---")
test_df = pd.DataFrame({
    'text': ['Row 1', 'Row 2'],
    'test_image': [
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==',
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
    ]
})

test_gb = GridOptionsBuilder.from_dataframe(test_df)
test_gb.configure_column(
    "test_image",
    headerName="Test Image",
    cellRenderer=JsCode("""
        function(params) {
            return '<img src="' + params.value + '" style="width:50px;height:50px;background:yellow;border:2px solid red;" />';
        }
    """)
)
test_options = test_gb.build()

test_grid = AgGrid(
    test_df,
    gridOptions=test_options,
    height=150,
    allow_unsafe_jscode=True,
    theme='streamlit'
)
st.write("If you see red-bordered squares above, JS is working")
st.write("--- END TEST ---")

# Custom CSS for image display
custom_css = {
    "#gridToolBar": {
        "padding-bottom": "0px !important",
    },
    ".ag-row": {
        "height": "120px !important"
    },
    ".ag-cell": {
        "display": "flex !important",
        "align-items": "center !important",
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY GRID
# ═══════════════════════════════════════════════════════════════════════════════

# Render AgGrid
grid_response = AgGrid(
    display_df,
    gridOptions=grid_options,
    height=800,
    allow_unsafe_jscode=True,
    enable_enterprise_modules=False,
    custom_css=custom_css,
    theme='streamlit',
    update_mode='NO_UPDATE'  # Try to prevent re-rendering issues
)

# Display selected row info
if grid_response['selected_rows'] is not None and len(grid_response['selected_rows']) > 0:
    with st.expander("Selected Row Details", expanded=True):
        selected = grid_response['selected_rows'][0]
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**File:** {selected.get('filename', 'N/A')}")
        with col2:
            st.write(f"**Detection:** {selected.get('detection_label', 'N/A')}")
            st.write(f"**Confidence:** {selected.get('confidence', 0):.2f}")