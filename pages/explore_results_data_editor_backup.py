"""
AddaxAI Results Explorer - DATA EDITOR BACKUP VERSION

This is a backup of the st.data_editor implementation.
The main file has been converted to use st.dataframe for selection support.
"""

import warnings
import sys

# Force warnings module into sys.modules if it's missing
# This fixes the KeyError: 'warnings' issue with pandas
if 'warnings' not in sys.modules:
    sys.modules['warnings'] = warnings

import streamlit as st
import base64
import os
from PIL import Image
import io
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Column widths
COLUMN_WIDTH_STANDARD = 100  # Standard width for most columns

# Row heights and image sizes
ROW_HEIGHT_OPTIONS = {
    "small": 35,   # Default streamlit row height
    "medium": 92,  # Middle size
    "large": 150   # Large size
}

# Image size ratio (width = height * ratio)
IMAGE_SIZE_RATIO = 1.5

# Image column widths (calculated from row height * ratio)
IMAGE_COLUMN_WIDTHS = {
    size: int(height * IMAGE_SIZE_RATIO) 
    for size, height in ROW_HEIGHT_OPTIONS.items()
}

# Image processing
IMAGE_BORDER_WIDTH = 2  # Border thickness in pixels
IMAGE_BORDER_COLOR = (20, 95, 100)  # #145f64 in RGB
IMAGE_BACKGROUND_COLOR = (220, 227, 232)  # #dce3e8 in RGB
IMAGE_PADDING_PERCENT = 0.01  # 1% padding
IMAGE_PADDING_MIN = 10  # Minimum padding in pixels
IMAGE_QUALITY = 85  # JPEG compression quality

# Table display
TABLE_HEIGHT_DEFAULT = 800  # Default table height
DEFAULT_SIZE_OPTION = "medium"  # Default size selection

# Number formatting
CONFIDENCE_DECIMALS = 2  # Decimal places for confidence values
COORDINATE_DECIMALS = 4  # Decimal places for bbox coordinates
LAT_LON_DECIMALS = 6  # Decimal places for lat/lon

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(layout="wide")

# Title with settings popover
col1, col2 = st.columns([10, 1])
with col1:
    st.title("🔍 Explore Detection Results (Data Editor Backup)")
with col2:
    with st.popover("⚙️", help="Image Size Settings"):        
        # Image size controls both row height and image column width
        # Proportion: 1 row height = 1.5 image column width
        image_size_options = {
            size: {"height": height, "width": IMAGE_COLUMN_WIDTHS[size]}
            for size, height in ROW_HEIGHT_OPTIONS.items()
        }
        
        selected_size = st.segmented_control(
            "Image Size",
            options=list(image_size_options.keys()),
            default=DEFAULT_SIZE_OPTION,
            key="image_size_control",
            width="stretch"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════

# Add download popover at the top
col1, col2 = st.columns([10, 1])
with col1:
    pass  # Empty space for alignment
with col2:
    download_popover = st.popover("⬇️ Download", help="Download results without images")

# Get settings values
image_size = st.session_state.get("image_size_setting", 300)

# Get row height and image width from segmented control
image_size_options = {
    size: {"height": height, "width": IMAGE_COLUMN_WIDTHS[size]}
    for size, height in ROW_HEIGHT_OPTIONS.items()
}
selected_size = st.session_state.get("image_size_control", DEFAULT_SIZE_OPTION)
size_config = image_size_options.get(selected_size, image_size_options[DEFAULT_SIZE_OPTION])
row_height_val = size_config["height"]
image_width = size_config["width"]

table_height = st.session_state.get("table_height_setting", TABLE_HEIGHT_DEFAULT)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

# Check if detection results are available
if "results_detections" not in st.session_state:
    st.error("No detection results found. Please ensure the application has been started properly and detection data is loaded.")
    st.info("Detection results are loaded during application startup. Try refreshing the page or restarting the application.")
    st.stop()

# Load detection results dataframe
df = st.session_state["results_detections"]

if df.empty:
    st.warning("No detection results found in the database.")
    st.info("This could mean:")
    st.write("- No deployments have been processed yet")
    st.write("- All deployments failed to process")
    st.write("- Detection result files are missing or corrupted")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

# Function to convert image to base64 data URL with cropping
def image_to_base64_url(image_path, bbox_data, max_size):
    """Convert image file to base64 data URL for ImageColumn display, cropped to detection bbox."""
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
                else:
                    # Regular crop within bounds
                    x1_square = max(0, x1_square)
                    y1_square = max(0, y1_square)
                    x2_square = min(img_width, x2_square)
                    y2_square = min(img_height, y2_square)
                    
                    if x2_square > x1_square and y2_square > y1_square:
                        img = img.crop((x1_square, y1_square, x2_square, y2_square))
            
            # Define border settings
            border_color = IMAGE_BORDER_COLOR
            border_width = IMAGE_BORDER_WIDTH
            
            # Reduce max_size to account for border that will be added
            adjusted_max_size = (max_size[0] - border_width * 2, max_size[1] - border_width * 2)
            
            # Create thumbnail to reduce size (accounting for border)
            img.thumbnail(adjusted_max_size, Image.Resampling.LANCZOS)
            
            # Create new image with border space
            w, h = img.size
            bordered_img = Image.new('RGB', (w + border_width * 2, h + border_width * 2), border_color)
            # Paste the image inside the border
            bordered_img.paste(img, (border_width, border_width))
            img = bordered_img
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            
            # Return as data URL
            return f"data:image/jpeg;base64,{img_data}"
    except Exception:
        # Return None silently if image cannot be processed
        return None

# Convert images to base64 data URLs for ImageColumn
# Process images and create new dataframe with image column
with st.spinner("Processing images..."):
    # Create the image URLs list first
    image_urls = []
    for index, row in df.iterrows():
        bbox_data = {
            'x': row['bbox_x'] if 'bbox_x' in row else None,
            'y': row['bbox_y'] if 'bbox_y' in row else None,
            'width': row['bbox_width'] if 'bbox_width' in row else None,
            'height': row['bbox_height'] if 'bbox_height' in row else None
        }
        image_urls.append(image_to_base64_url(row['absolute_path'], bbox_data, max_size=(image_size, image_size)))
    
    # Create new dataframe with all original columns plus image_url
    display_df = pd.DataFrame(df)
    display_df.insert(0, 'image_url', image_urls)  # Insert at beginning

# Convert timestamp to datetime if it exists and isn't already
if 'timestamp' in display_df.columns:
    # Replace colons with hyphens in the date part (YYYY:MM:DD -> YYYY-MM-DD)
    display_df['timestamp'] = display_df['timestamp'].str.replace(r'(\d{4}):(\d{2}):(\d{2})', r'\1-\2-\3', regex=True)
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])

# Round numeric columns for better display
for col in ['detection_confidence', 'classification_confidence']:
    if col in display_df.columns:
        display_df[col] = display_df[col].round(CONFIDENCE_DECIMALS)

for col in ['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']:
    if col in display_df.columns:
        display_df[col] = display_df[col].round(COORDINATE_DECIMALS)
        
for col in ['latitude', 'longitude']:
    if col in display_df.columns:
        display_df[col] = display_df[col].round(LAT_LON_DECIMALS)

# Define column order - image first, then important columns
column_order = [
    'image_url',  # Will be displayed as image (base64 data URL)
    'relative_path',
    'project_id',
    'location_id', 
    'deployment_id',
    'detection_label',
    'detection_confidence',
    'classification_label',
    'classification_confidence',
    'timestamp',
    'bbox_x',
    'bbox_y', 
    'bbox_width',
    'bbox_height',
    'latitude',
    'longitude',
    'image_width',
    'image_height',
    'detection_model_id',
    'classification_model_id'
]

# Filter column order to only include columns that exist
column_order = [col for col in column_order if col in display_df.columns]

# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD POPOVER CONTENT (now that data is ready)
# ═══════════════════════════════════════════════════════════════════════════════

# Create download dataframe without image column
download_df = display_df.drop(columns=['image_url'])

# Add the download options inside the popover
with download_popover:
    format_option = st.selectbox(
        "Select format:",
        options=["CSV", "XLSX"],
        key="download_format"
    )
    
    if format_option == "CSV":
        csv = download_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="detection_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:  # XLSX
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            download_df.to_excel(writer, index=False, sheet_name='Detection Results')
        excel_data = buffer.getvalue()
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name="detection_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# ═══════════════════════════════════════════════════════════════════════════════
# DATA DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

# Display data editor with image column
edited_df = st.data_editor(
    display_df,
    column_config={
        "image_url": st.column_config.ImageColumn(
            "Image",
            help="Detection image thumbnail",
            width=image_width
        ),
        "absolute_path": None,  # Hide the original absolute path column
        "relative_path": st.column_config.TextColumn(
            "File",
            help="Image filename",
            width=COLUMN_WIDTH_STANDARD
        ),
        "project_id": st.column_config.TextColumn(
            "Project",
            width=COLUMN_WIDTH_STANDARD
        ),
        "location_id": st.column_config.TextColumn(
            "Location",
            width=COLUMN_WIDTH_STANDARD
        ),
        "deployment_id": st.column_config.TextColumn(
            "Deployment",
            width=COLUMN_WIDTH_STANDARD
        ),
        "detection_label": st.column_config.TextColumn(
            "Detection",
            width=COLUMN_WIDTH_STANDARD
        ),
        "detection_confidence": st.column_config.ProgressColumn(
            "Det. Conf",
            min_value=0,
            max_value=1,
            format=f"%.{CONFIDENCE_DECIMALS}f",
            width=COLUMN_WIDTH_STANDARD
        ),
        "classification_label": st.column_config.TextColumn(
            "Species",
            width=COLUMN_WIDTH_STANDARD
        ),
        "classification_confidence": st.column_config.ProgressColumn(
            "Class. Conf",
            min_value=0,
            max_value=1,
            format=f"%.{CONFIDENCE_DECIMALS}f",
            width=COLUMN_WIDTH_STANDARD
        ),
        "timestamp": st.column_config.DatetimeColumn(
            "Timestamp",
            format="YYYY-MM-DD HH:mm:ss",
            width=COLUMN_WIDTH_STANDARD
        ),
        "bbox_x": st.column_config.NumberColumn(
            "BBox X",
            format=f"%.{COORDINATE_DECIMALS}f",
            width=COLUMN_WIDTH_STANDARD
        ),
        "bbox_y": st.column_config.NumberColumn(
            "BBox Y", 
            format=f"%.{COORDINATE_DECIMALS}f",
            width=COLUMN_WIDTH_STANDARD
        ),
        "bbox_width": st.column_config.NumberColumn(
            "BBox W",
            format=f"%.{COORDINATE_DECIMALS}f",
            width=COLUMN_WIDTH_STANDARD
        ),
        "bbox_height": st.column_config.NumberColumn(
            "BBox H",
            format=f"%.{COORDINATE_DECIMALS}f",
            width=COLUMN_WIDTH_STANDARD
        ),
        "latitude": st.column_config.NumberColumn(
            "Lat",
            format=f"%.{LAT_LON_DECIMALS}f",
            width=COLUMN_WIDTH_STANDARD
        ),
        "longitude": st.column_config.NumberColumn(
            "Lon",
            format=f"%.{LAT_LON_DECIMALS}f",
            width=COLUMN_WIDTH_STANDARD
        ),
        "image_width": st.column_config.NumberColumn(
            "Img W",
            width=COLUMN_WIDTH_STANDARD
        ),
        "image_height": st.column_config.NumberColumn(
            "Img H",
            width=COLUMN_WIDTH_STANDARD
        ),
        "detection_model_id": st.column_config.TextColumn(
            "Det Model",
            width=COLUMN_WIDTH_STANDARD
        ),
        "classification_model_id": st.column_config.TextColumn(
            "Class Model",
            width=COLUMN_WIDTH_STANDARD
        ),
    },
    column_order=column_order,
    disabled=True,  # Read-only mode
    width="stretch",  # Full width display
    hide_index=True,
    height=table_height,  # Dynamic height from settings
    num_rows="fixed",
    row_height=row_height_val  # Dynamic row height from settings
)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.caption(f"Showing {len(display_df):,} detection results")