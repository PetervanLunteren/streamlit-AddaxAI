"""
AddaxAI Data Browser Utilities

Utility functions for the data browser page.
Provides image modal viewer, filter state management, and thumbnail generation for AgGrid data browsing.
"""

import json
import os
import base64
import io
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd
from utils.config import ADDAXAI_ROOT, log
from utils.common import load_vars, update_vars

# Thumbnail generation constants
MODAL_IMAGE_SIZE = 1000  # High resolution modal images (pixels) - 1000x1000 for quality
IMAGE_BACKGROUND_COLOR = (220, 227, 232)
IMAGE_PADDING_PIXELS = 100  # Padding around bbox in pixels (new padding system)
IMAGE_BLUR_RADIUS = 15  # Blur radius for background extension
IMAGE_CORNER_RADIUS = 20  # Corner radius for rounded thumbnails (pixels)
IMAGE_BBOX_COLOR = 'red'  # Color for bounding box outline
IMAGE_QUALITY = 85


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


def format_detection_label(det_label, det_conf, cls_label=None, cls_conf=None):
    """
    Format detection and classification labels for thumbnail overlay.
    
    Args:
        det_label (str): Detection label (e.g., "animal")
        det_conf (float): Detection confidence (0-1)
        cls_label (str, optional): Classification label (e.g., "lion") 
        cls_conf (float, optional): Classification confidence (0-1)
    
    Returns:
        str: Formatted label text
    """
    # Start with detection info
    if det_conf is not None and pd.notna(det_conf):
        label_parts = [f"{det_label} ({det_conf:.2f})"]
    else:
        label_parts = [str(det_label)]
    
    # Add classification info if available
    if cls_label and cls_label != 'N/A' and pd.notna(cls_label):
        if cls_conf is not None and pd.notna(cls_conf):
            label_parts.append(f"{cls_label} ({cls_conf:.2f})")
        else:
            label_parts.append(str(cls_label))
    
    return " - ".join(label_parts)


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
                    x1_target = 0
                    x2_target = min(img_width, target_size)
                elif x2_target > img_width:
                    # Shift left
                    x2_target = img_width
                    x1_target = max(0, img_width - target_size)
                
                if y1_target < 0:
                    # Shift down
                    y1_target = 0
                    y2_target = min(img_height, target_size)
                elif y2_target > img_height:
                    # Shift up
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
            img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_data}"
    except:
        return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='


def generate_modal_image(image_path, bbox_data, show_bbox=True, show_labels=True):
    """
    Generate high-resolution image for modal display.
    
    Args:
        image_path (str): Path to the image file
        bbox_data (dict): Bounding box data with x, y, width, height
        show_bbox (bool): Whether to draw bounding box
        show_labels (bool): Whether to draw detection/classification labels
        
    Returns:
        str: Base64 encoded image data URL
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='
        
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create high-resolution modal image (1000x1000)
            modal_size = (MODAL_IMAGE_SIZE, MODAL_IMAGE_SIZE)
            crop_info = None
            
            # Crop to bounding box if available
            if bbox_data and all(pd.notna([bbox_data['x'], bbox_data['y'], 
                                           bbox_data['width'], bbox_data['height']])):
                img_width, img_height = img.size
                x = int(bbox_data['x'] * img_width)
                y = int(bbox_data['y'] * img_height)
                w = int(bbox_data['width'] * img_width)
                h = int(bbox_data['height'] * img_height)
                
                # Same padding logic as grid thumbnails
                target_size = max(w, h) + (IMAGE_PADDING_PIXELS * 2)
                
                bbox_center_x = x + w // 2
                bbox_center_y = y + h // 2
                
                x1_target = bbox_center_x - target_size // 2
                y1_target = bbox_center_y - target_size // 2
                x2_target = x1_target + target_size
                y2_target = y1_target + target_size
                
                # Edge handling
                if x1_target < 0:
                    x1_target = 0
                    x2_target = min(img_width, target_size)
                elif x2_target > img_width:
                    x2_target = img_width
                    x1_target = max(0, img_width - target_size)
                
                if y1_target < 0:
                    y1_target = 0
                    y2_target = min(img_height, target_size)
                elif y2_target > img_height:
                    y2_target = img_height
                    y1_target = max(0, img_height - target_size)
                
                actual_crop_width = x2_target - x1_target
                actual_crop_height = y2_target - y1_target
                
                if actual_crop_width == target_size and actual_crop_height == target_size:
                    # Perfect fit
                    img = img.crop((x1_target, y1_target, x2_target, y2_target))
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
                    # Need background extension
                    cropped_img = img.crop((x1_target, y1_target, x2_target, y2_target))
                    blurred_crop = cropped_img.copy()
                    blurred_crop = blurred_crop.filter(ImageFilter.GaussianBlur(radius=IMAGE_BLUR_RADIUS))
                    blurred_background = blurred_crop.resize((target_size, target_size), Image.Resampling.LANCZOS)
                    
                    paste_x = (target_size - actual_crop_width) // 2
                    paste_y = (target_size - actual_crop_height) // 2
                    blurred_background.paste(cropped_img, (paste_x, paste_y))
                    img = blurred_background
                    
                    bbox_x_in_crop = (x - x1_target) + paste_x
                    bbox_y_in_crop = (y - y1_target) + paste_y
                    crop_info = {
                        'bbox_x_in_crop': bbox_x_in_crop,
                        'bbox_y_in_crop': bbox_y_in_crop,
                        'bbox_w': w,
                        'bbox_h': h,
                        'crop_size': target_size
                    }
            
            # Create high-resolution thumbnail
            img.thumbnail(modal_size, Image.Resampling.LANCZOS)
            thumbnail_size = img.size
            
            # Draw bounding box if requested
            if crop_info and show_bbox:
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
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=IMAGE_BBOX_COLOR, width=2)  # Thicker for modal
                
                # Add labels if requested (will be set separately)
                if show_labels and hasattr(generate_modal_image, '_current_label_text'):
                    label_text = generate_modal_image._current_label_text
                    
                    # Calculate font size for high-res image
                    font_size = max(12, min(24, thumbnail_size[0] // 35))
                    
                    try:
                        from PIL import ImageFont
                        font = ImageFont.load_default()
                    except:
                        font = None
                    
                    # Get text dimensions
                    if font:
                        bbox_text = draw.textbbox((0, 0), label_text, font=font)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]
                    else:
                        text_width = len(label_text) * (font_size // 2)
                        text_height = font_size
                    
                    # Position text
                    text_x = max(4, min(x1, thumbnail_size[0] - text_width - 8))
                    text_y = max(4, y1 - text_height - 6)
                    
                    if text_y < 4:
                        text_y = min(y2 + 6, thumbnail_size[1] - text_height - 4)
                    
                    # Draw background
                    padding = 4
                    bg_x1 = text_x - padding
                    bg_y1 = text_y - padding
                    bg_x2 = text_x + text_width + padding
                    bg_y2 = text_y + text_height + padding
                    
                    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 200))
                    
                    img_with_overlay = Image.alpha_composite(img.convert('RGBA'), overlay)
                    img = img_with_overlay.convert('RGB')
                    draw = ImageDraw.Draw(img)
                    
                    # Draw text
                    draw.text((text_x, text_y), label_text, fill='white', font=font)
            
            # Apply rounded corners
            if IMAGE_CORNER_RADIUS > 0:
                mask = Image.new('L', img.size, 0)
                mask_draw = ImageDraw.Draw(mask)
                scaled_radius = int(IMAGE_CORNER_RADIUS * (thumbnail_size[0] / 250))
                scaled_radius = max(8, min(scaled_radius, min(thumbnail_size) // 4))
                
                mask_draw.rounded_rectangle(
                    [(0, 0), (thumbnail_size[0]-1, thumbnail_size[1]-1)],
                    radius=scaled_radius,
                    fill=255
                )
                
                output = Image.new('RGBA', img.size, (255, 255, 255, 0))
                output.paste(img, (0, 0))
                output.putalpha(mask)
                
                final = Image.new('RGB', img.size, 'white')
                final.paste(output, (0, 0), output)
                img = final
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_data}"
            
    except Exception as e:
        return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='


def image_viewer_modal():
    """
    Display cached thumbnail in modal with metadata and navigation.
    Uses pre-generated thumbnails from the data browser cache.
    """
    import streamlit as st
    from PIL import Image, ImageDraw
    import os
    import pandas as pd
    import base64
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
        # Segmented control for bbox toggle
        bbox_options = ["Without Detection Box", "With Detection Box"]
        
        # Get user's last choice, defaulting to "With Detection Box"
        show_bbox = get_session_var("explore_results", "modal_show_bbox", True)
        default_option = bbox_options[1] if show_bbox else bbox_options[0]
        
        selected_option = st.segmented_control(
            "Display mode",
            options=bbox_options,
            default=default_option,
            key="bbox_toggle"
        )
        
        # Update bbox state if changed and save preference
        new_show_bbox = (selected_option == "With Detection Box")
        if new_show_bbox != show_bbox:
            set_session_var("explore_results", "modal_show_bbox", new_show_bbox)
            st.rerun()  # Force immediate UI update
        
        # Generate high-resolution modal image on-demand
        bbox_data = {
            'x': current_row.get('bbox_x'),
            'y': current_row.get('bbox_y'), 
            'width': current_row.get('bbox_width'),
            'height': current_row.get('bbox_height')
        }
        
        # Set label text for modal image generation
        det_label = current_row.get('detection_label', 'N/A')
        det_conf = current_row.get('detection_confidence')
        cls_label = current_row.get('classification_label')
        cls_conf = current_row.get('classification_confidence')
        label_text = format_detection_label(det_label, det_conf, cls_label, cls_conf)
        
        # Pass label text to generation function via function attribute (simple approach)
        generate_modal_image._current_label_text = label_text
        
        # Generate modal image at 1000x1000 resolution
        modal_image_url = generate_modal_image(
            image_path, 
            bbox_data, 
            show_bbox=show_bbox, 
            show_labels=True
        )
        
        # Display the high-resolution modal image
        if modal_image_url:
            st.markdown(f'<img src="{modal_image_url}" style="width: 100%; max-width: 600px; height: auto;">', unsafe_allow_html=True)
        else:
            st.error("Could not load image")
    
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
        run_id = current_row.get('run_id', 'N/A')
        st.write(f"• Location: {location_id}")
        st.write(f"• Run: {run_id}")
        
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