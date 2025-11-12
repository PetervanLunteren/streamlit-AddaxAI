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
from components.shortcut_utils import register_shortcuts, clear_shortcut_listeners

# Thumbnail generation constants
MODAL_IMAGE_SIZE = 1000  # High-resolution modal images (pixels) used for export/zoom
IMAGE_BACKGROUND_COLOR = (220, 227, 232)
IMAGE_PADDING_PIXELS = 100  # Padding around bbox in pixels (new padding system)
IMAGE_BLUR_RADIUS = 15  # Blur radius for background extension
IMAGE_CORNER_RADIUS = 4  # Corner radius for rounded thumbnails (pixels)
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
    
    return "\n".join(label_parts)


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
            
            # Create thumbnail (cap height to keep processing fast)
            max_width, max_height = max_size
            max_height = max(60, min(max_height, 180))
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
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
            
            # Create high-resolution modal image (square at MODAL_IMAGE_SIZE)
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
            
            # Store bbox coordinates for later drawing (after rounding)
            bbox_coords = None
            if crop_info and show_bbox:
                scale_x = thumbnail_size[0] / crop_info['crop_size']
                scale_y = thumbnail_size[1] / crop_info['crop_size']
                
                bbox_x_thumb = crop_info['bbox_x_in_crop'] * scale_x
                bbox_y_thumb = crop_info['bbox_y_in_crop'] * scale_y
                bbox_w_thumb = crop_info['bbox_w'] * scale_x
                bbox_h_thumb = crop_info['bbox_h'] * scale_y
                
                x1 = int(max(0, bbox_x_thumb))
                y1 = int(max(0, bbox_y_thumb))
                x2 = int(min(thumbnail_size[0]-1, bbox_x_thumb + bbox_w_thumb))
                y2 = int(min(thumbnail_size[1]-1, bbox_y_thumb + bbox_h_thumb))
                
                bbox_coords = (x1, y1, x2, y2)
            
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
            
            # Draw bounding box and labels AFTER rounding (fixed sizes for consistency)
            if bbox_coords and show_bbox:
                x1, y1, x2, y2 = bbox_coords
                draw = ImageDraw.Draw(img)
                
                # Calculate bbox size in final image
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_min_dim = min(bbox_width, bbox_height)
                
                # Scale line width based on bbox size (1-3 pixels)
                if bbox_min_dim < 100:  # Very small detection
                    line_width = 1
                elif bbox_min_dim < 300:  # Medium detection
                    line_width = 2
                else:  # Large detection
                    line_width = 3
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=IMAGE_BBOX_COLOR, width=line_width)
                
                # Add labels if requested
                if show_labels and hasattr(generate_modal_image, '_current_label_text'):
                    label_text = generate_modal_image._current_label_text
                    
                    # Scale font size based on bbox size (8-20px based on bbox dimension)
                    # Smaller bbox = smaller font
                    font_size = max(8, min(20, int(bbox_min_dim * 0.08)))
                    
                    try:
                        from PIL import ImageFont
                        font = None
                        for font_name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
                            try:
                                font = ImageFont.truetype(font_name, 15)
                                break
                            except Exception:
                                continue
                        if font is None:
                            font = ImageFont.load_default()
                    except:
                        font = None
                    
                    # Split text into lines
                    lines = label_text.split('\n')
                    
                    # Get dimensions for multiline text
                    max_width = 0
                    total_height = 0
                    line_heights = []
                    
                    for line in lines:
                        if font:
                            bbox_text = draw.textbbox((0, 0), line, font=font)
                            line_width = bbox_text[2] - bbox_text[0]
                            line_height = bbox_text[3] - bbox_text[1]
                        else:
                            line_width = len(line) * (font_size // 2)
                            line_height = font_size
                        
                        max_width = max(max_width, line_width)
                        line_heights.append(line_height)
                        total_height += line_height
                    
                    # Add spacing between lines
                    line_spacing = 2
                    total_height += (len(lines) - 1) * line_spacing
                    
                    # Position text (above bbox if possible, below if not)
                    text_x = max(4, min(x1, thumbnail_size[0] - max_width - 8))
                    text_y = max(4, y1 - total_height - 6)
                    
                    if text_y < 4:
                        text_y = min(y2 + 6, thumbnail_size[1] - total_height - 4)
                    
                    # Draw background
                    padding = 4
                    bg_x1 = text_x - padding
                    bg_y1 = text_y - padding
                    bg_x2 = text_x + max_width + padding
                    bg_y2 = text_y + total_height + padding
                    
                    # Draw semi-transparent background
                    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 200))
                    
                    img_with_overlay = Image.alpha_composite(img.convert('RGBA'), overlay)
                    img = img_with_overlay.convert('RGB')
                    draw = ImageDraw.Draw(img)
                    
                    # Draw multiline text
                    y_offset = text_y
                    for i, line in enumerate(lines):
                        draw.text((text_x, y_offset), line, fill='white', font=font)
                        y_offset += line_heights[i] + line_spacing
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_data}"
            
    except Exception as e:
        return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='


def image_to_base64_with_boxes(image_path, detection_details, max_height):
    """
    Generate a scaled image with all detections drawn for table thumbnails.

    Args:
        image_path (str): Absolute path to file
        detection_details (list): Detection dictionaries with normalized bbox values
        max_height (int): Target height of thumbnail

    Returns:
        str: Base64 data URL
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='

        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            orig_width, orig_height = img.size
            max_height = max(60, min(max_height, 180))

            if orig_height > max_height:
                scale = max_height / float(orig_height)
                new_width = int(orig_width * scale)
                img = img.resize((new_width, max_height), Image.Resampling.LANCZOS)
            else:
                new_width, max_height = orig_width, orig_height

            draw = ImageDraw.Draw(img)

            for detection in detection_details or []:
                bbox = detection.get("bbox") or {}
                x_norm = bbox.get("x")
                y_norm = bbox.get("y")
                w_norm = bbox.get("width")
                h_norm = bbox.get("height")
                if None in (x_norm, y_norm, w_norm, h_norm):
                    continue
                if any(pd.isna(value) for value in (x_norm, y_norm, w_norm, h_norm)):
                    continue

                x1 = int(x_norm * new_width)
                y1 = int(y_norm * max_height)
                x2 = int((x_norm + w_norm) * new_width)
                y2 = int((y_norm + h_norm) * max_height)

                line_width = max(1, int(min(new_width, max_height) * 0.003))
                draw.rectangle([x1, y1, x2, y2], outline=IMAGE_BBOX_COLOR, width=line_width)

            if IMAGE_CORNER_RADIUS > 0:
                mask = Image.new('L', img.size, 0)
                mask_draw = ImageDraw.Draw(mask)
                scaled_radius = max(5, min(int(IMAGE_CORNER_RADIUS), min(new_width, max_height) // 4))
                mask_draw.rounded_rectangle(
                    [(0, 0), (new_width - 1, max_height - 1)],
                    radius=scaled_radius,
                    fill=255
                )
                output = Image.new('RGBA', img.size, (255, 255, 255, 0))
                output.paste(img, (0, 0))
                output.putalpha(mask)
                final = Image.new('RGB', img.size, 'white')
                final.paste(output, (0, 0), output)
                img = final

            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_data}"
    except Exception:
        return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='

def generate_file_modal_image(image_path, detection_details, show_bbox=True, show_labels=True):
    """
    Generate full-image modal preview with all detections drawn.

    Args:
        image_path (str): Absolute path to media file.
        detection_details (list): List of detection dicts with bbox info.
        show_bbox (bool): Whether to draw bounding boxes.
        show_labels (bool): Whether to render labels next to boxes.

    Returns:
        str: Base64 encoded image suitable for HTML img tag.
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='

        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            orig_width, orig_height = img.size

            max_width = MODAL_IMAGE_SIZE
            if orig_width > max_width:
                scale = max_width / float(orig_width)
                display_width = max_width
                display_height = int(orig_height * scale)
                img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            else:
                display_width = orig_width
                display_height = orig_height

            draw = ImageDraw.Draw(img)

            if show_bbox and detection_details:
                for idx, detection in enumerate(detection_details):
                    bbox = detection.get("bbox") or {}
                    x_norm = bbox.get("x")
                    y_norm = bbox.get("y")
                    w_norm = bbox.get("width")
                    h_norm = bbox.get("height")

                    if None in (x_norm, y_norm, w_norm, h_norm):
                        continue

                    if pd.isna(x_norm) or pd.isna(y_norm) or pd.isna(w_norm) or pd.isna(h_norm):
                        continue

                    x1 = int(x_norm * display_width)
                    y1 = int(y_norm * display_height)
                    x2 = int((x_norm + w_norm) * display_width)
                    y2 = int((y_norm + h_norm) * display_height)

                    line_width = max(1, int(min(display_width, display_height) * 0.003))
                    draw.rectangle([x1, y1, x2, y2], outline=IMAGE_BBOX_COLOR, width=line_width)

                    if show_labels:
                        det_label = detection.get("detection_label") or "detection"
                        det_conf = detection.get("detection_confidence")
                        cls_label = detection.get("classification_label")
                        cls_conf = detection.get("classification_confidence")

                        lines = []
                        if det_label:
                            label = det_label
                            if det_conf is not None:
                                try:
                                    label = f"{label} ({float(det_conf):.2f})"
                                except (TypeError, ValueError):
                                    pass
                            lines.append(label)
                        if cls_label:
                            label = cls_label
                            if cls_conf is not None:
                                try:
                                    label = f"{label} ({float(cls_conf):.2f})"
                                except (TypeError, ValueError):
                                    pass
                            lines.append(label)

                        label_text = "\n".join(lines) if lines else f"detection {idx + 1}"

                        try:
                            from PIL import ImageFont
                            font = None
                            for font_name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
                                try:
                                    font = ImageFont.truetype(font_name, 20)
                                    break
                                except Exception:
                                    continue
                            if font is None:
                                font = ImageFont.load_default()
                        except Exception:
                            font = ImageFont.load_default()

                        text_width = text_height = 0
                        if font:
                            text_bbox = draw.multiline_textbbox((0, 0), label_text, font=font, spacing=2)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                        else:
                            text_width = len(label_text) * 6
                            text_height = 10

                        padding = 4
                        bg_x1 = x1 + padding
                        bg_y1 = max(0, y1 - text_height - (2 * padding))
                        bg_x2 = bg_x1 + text_width + (2 * padding)
                        bg_y2 = bg_y1 + text_height + (2 * padding)

                        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                        overlay_draw = ImageDraw.Draw(overlay)
                        overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 200))

                        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                        draw = ImageDraw.Draw(img)

                        draw.multiline_text(
                            (bg_x1 + padding, bg_y1 + padding),
                            label_text,
                            fill='white',
                            font=font,
                            spacing=2
                        )

            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
            img_data = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/jpeg;base64,{img_data}"
    except Exception:
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
    from components.ui_helpers import code_span
    
    # Get current modal state
    current_index = get_session_var("explore_results", "modal_current_image_index", 0)
    show_bbox = get_session_var("explore_results", "modal_show_bbox", True)
    
    # Get the filtered dataframe from session state
    df_filtered = st.session_state.get('observations_results', pd.DataFrame())
    
    if df_filtered.empty or current_index >= len(df_filtered):
        st.error("No image data available")
        if st.button("Close"):
            set_session_var("explore_results", "show_modal_image_viewer", False)
            st.rerun()
        clear_shortcut_listeners()
        return
    
    # Get current row data
    current_row = df_filtered.iloc[current_index]
    image_path = current_row.get('absolute_path', '')
    
    # Main layout: image on left, metadata on right
    img_col, meta_col = st.columns([1, 1])
    
    with img_col:
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
        
        # Generate modal image at MODAL_IMAGE_SIZE resolution
        modal_image_url = generate_modal_image(
            image_path, 
            bbox_data, 
            show_bbox=show_bbox, 
            show_labels=True
        )
        
        # Display the high-resolution modal image
        with st.container(border=True):
            if modal_image_url:
                st.markdown(
                    f'<img src="{modal_image_url}" style="width: 480px; height: auto;">',
                    unsafe_allow_html=True
                )
            else:
                st.error("Could not load image")
            st.write("")

    with meta_col:
        top_col_export, top_col_close = st.columns([1, 1])

        with top_col_export:
            if modal_image_url and modal_image_url.startswith('data:image/jpeg;base64,'):
                import base64
                base64_data = modal_image_url.split(',')[1]
                image_bytes = base64.b64decode(base64_data)

                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                download_filename = f"addaxai-{timestamp}.jpg"

                st.download_button(
                    label=":material/download: Export",
                    data=image_bytes,
                    file_name=download_filename,
                    mime="image/jpeg",
                    type="secondary",
                    key="modal_export_button",
                    width='stretch'
                )

        with top_col_close:
            if st.button(
                ":material/close: Close",
                type="secondary",
                width="stretch",
                key="modal_close_button",
                help="Closes the window (shortcut `Esc`)",
            ):
                clear_shortcut_listeners()
                set_session_var("explore_results", "show_modal_image_viewer", False)
                st.rerun()
        
        nav_col_prev, nav_col_next = st.columns([1, 1])

        with nav_col_prev:
            if st.button(
                ":material/chevron_left: Previous",
                disabled=(current_index <= 0),
                width="stretch",
                type="secondary",
                key="observation_modal_prev_button",
                help="Show previous row (shortcut `←`)",
            ):
                set_session_var("explore_results", "modal_current_image_index", current_index - 1)
                st.rerun()

        with nav_col_next:
            if st.button(
                "Next :material/chevron_right:",
                disabled=(current_index >= len(df_filtered) - 1),
                width="stretch",
                type="secondary",
                key="observation_modal_next_button",
                help="Show next row (shortcut `→`)",
            ):
                set_session_var("explore_results", "modal_current_image_index", current_index + 1)
                st.rerun()


        show_bbox = get_session_var("explore_results", "modal_show_bbox", True)
        bbox_options = ["hide", "show"]
        option_labels = {
            "hide": ":material/visibility_off: Hide",
            "show": ":material/visibility: Show",
        }

        selected_option = st.segmented_control(
            "Show box",
            options=bbox_options,
            default="show" if show_bbox else "hide",
            format_func=lambda opt: option_labels.get(opt, opt.title()),
            key="bbox_toggle",
            label_visibility="collapsed",
            width="stretch"
        )

        new_show_bbox = (selected_option == "show")
        if new_show_bbox != show_bbox:
            set_session_var("explore_results", "modal_show_bbox", new_show_bbox)
            st.rerun()


        with st.container(border=True):
            timestamp = current_row.get('timestamp', 'N/A')
            if timestamp != 'N/A':
                try:
                    from datetime import datetime
                    parsed_timestamp = datetime.strptime(timestamp, '%Y:%m:%d %H:%M:%S')
                    human_readable = parsed_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    timestamp_display = human_readable
                except:
                    timestamp_display = timestamp
            else:
                timestamp_display = timestamp

            metadata_rows = []
            metadata_rows.append(("Timestamp", timestamp_display))

            location_id = current_row.get('location_id', 'N/A')
            metadata_rows.append(("Location ID", location_id))

            rel_path = current_row.get('relative_path', 'N/A')
            if rel_path != 'N/A':
                filename = os.path.basename(rel_path)
                if len(filename) > 20:
                    display_filename = "..." + filename[-20:]
                else:
                    display_filename = filename
            else:
                display_filename = 'N/A'
            metadata_rows.append(("File", display_filename))

            metadata_rows.append(("Row", f"{current_index + 1} of {len(df_filtered)}"))

            det_label = current_row.get("detection_label") or "N/A"
            det_conf = current_row.get("detection_confidence")
            detection_display = det_label
            if det_conf is not None and not pd.isna(det_conf):
                detection_display = f"{det_label} {float(det_conf):.2f}"
            metadata_rows.append(("Detection", detection_display))

            cls_label = current_row.get("classification_label") or "N/A"
            cls_conf = current_row.get("classification_confidence")
            classification_display = cls_label
            if cls_conf is not None and not pd.isna(cls_conf):
                classification_display = f"{cls_label} {float(cls_conf):.2f}"
            metadata_rows.append(("Classification", classification_display))

            for label, value in metadata_rows:
                st.markdown(f"**{label}** {code_span(value)}", unsafe_allow_html=True)

        register_shortcuts(
            observation_modal_prev_button=["arrowleft"],
            observation_modal_next_button=["arrowright"],
            modal_close_button=["escape"],
        )


def image_viewer_modal_file():
    """
    Display file-level modal showing full image with all detections overlaid.
    """
    import streamlit as st
    import pandas as pd
    from utils.common import get_session_var, set_session_var
    from components.ui_helpers import code_span

    df_files = st.session_state.get("files_results", pd.DataFrame())
    current_index = get_session_var("explore_results", "modal_current_image_index", 0)
    show_bbox = get_session_var("explore_results", "modal_show_bbox", True)

    if df_files.empty or current_index >= len(df_files):
        st.error("No file data available")
        if st.button("Close"):
            set_session_var("explore_results", "show_modal_image_viewer", False)
            st.rerun()
        return

    current_row = df_files.iloc[current_index]
    image_path = current_row.get("absolute_path", "")
    detection_details = current_row.get("detection_details", []) or []

    modal_image_url = generate_file_modal_image(
        image_path,
        detection_details,
        show_bbox=show_bbox,
        show_labels=True,
    )

    img_col, meta_col = st.columns([1, 1])

    with img_col:
        with st.container(border=True):
            if modal_image_url:
                st.markdown(
                    f'<img src="{modal_image_url}" style="width: 480px; height: auto;">',
                    unsafe_allow_html=True,
                )
            else:
                st.error("Could not load image")
            st.write("")

    with meta_col:
        top_col_export, top_col_close = st.columns([1, 1])

        with top_col_export:
            if modal_image_url and modal_image_url.startswith("data:image/jpeg;base64,"):
                import base64
                from datetime import datetime

                base64_data = modal_image_url.split(",")[1]
                image_bytes = base64.b64decode(base64_data)

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                download_filename = f"addaxai-file-{timestamp}.jpg"

                st.download_button(
                    label=":material/download: Export",
                    data=image_bytes,
                    file_name=download_filename,
                    mime="image/jpeg",
                    type="secondary",
                    key="file_modal_export_button",
                    width='stretch',
                )

        with top_col_close:
            if st.button(
                ":material/close: Close",
                type="secondary",
                width="stretch",
                key="file_modal_close_button",
                help="Closes the window (shortcut `Esc`)",
            ):
                clear_shortcut_listeners()
                set_session_var("explore_results", "show_modal_image_viewer", False)
                st.rerun()

        nav_col_prev, nav_col_next = st.columns([1, 1])
        with nav_col_prev:
            if st.button(
                ":material/chevron_left: Previous",
                disabled=(current_index <= 0),
                width="stretch",
                type="secondary",
                key="file_modal_prev_button",
                help="Show previous row (shortcut `←`)",
            ):
                set_session_var("explore_results", "modal_current_image_index", current_index - 1)
                st.rerun()

        with nav_col_next:
            if st.button(
                "Next :material/chevron_right:",
                disabled=(current_index >= len(df_files) - 1),
                width="stretch",
                type="secondary",
                key="file_modal_next_button",
                help="Show next row (shortcut `→`)",
            ):
                set_session_var("explore_results", "modal_current_image_index", current_index + 1)
                st.rerun()


        bbox_options = ["hide", "show"]
        option_labels = {
            "hide": ":material/visibility_off: Hide",
            "show": ":material/visibility: Show",
        }

        selected_option = st.segmented_control(
            "Show boxes",
            options=bbox_options,
            default="show" if show_bbox else "hide",
            format_func=lambda opt: option_labels.get(opt, opt.title()),
            key="file_bbox_toggle",
            label_visibility="collapsed",
            width="stretch",
        )

        new_show_bbox = selected_option == "show"
        if new_show_bbox != show_bbox:
            set_session_var("explore_results", "modal_show_bbox", new_show_bbox)
            st.rerun()


        with st.container(border=True):
            raw_timestamp = current_row.get("timestamp")
            if raw_timestamp:
                try:
                    timestamp_display = pd.to_datetime(raw_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    timestamp_display = str(raw_timestamp)
            else:
                timestamp_display = "N/A"

            detections_summary = current_row.get("detections_summary", "N/A")
            classifications_summary = current_row.get("classifications_summary", "N/A")
            metadata_rows = [
                ("Timestamp", timestamp_display),
                ("Location ID", current_row.get("location_id", "N/A")),
                ("File", os.path.basename(current_row.get("relative_path", "")) or "N/A"),
                ("Row", f"{current_index + 1} of {len(df_files)}"),
                ("Detections", detections_summary),
                ("Classifications", classifications_summary),
            ]

            for label, value in metadata_rows:
                st.markdown(f"**{label}** {code_span(value)}", unsafe_allow_html=True)

        register_shortcuts(
            file_modal_prev_button=["arrowleft"],
            file_modal_next_button=["arrowright"],
            file_modal_close_button=["escape"],
        )

def classification_selector_modal(nodes, all_leaf_values):
    """
    Hierarchical species selector modal for data browser filtering.

    Args:
        nodes (list): Merged tree structure
        all_leaf_values (list): All unique species across all models
    """
    import streamlit as st
    from st_checkbox_tree import checkbox_tree
    from utils.common import get_session_var, set_session_var

    # Select all / none buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(":material/select_check_box: Select all", width='stretch', key="cls_select_all"):
            set_session_var("explore_results", "selected_classifications", all_leaf_values)
            st.rerun()
    with col2:
        if st.button(":material/check_box_outline_blank: Select none", width='stretch', key="cls_select_none"):
            set_session_var("explore_results", "selected_classifications", [])
            set_session_var("explore_results", "expanded_cls_nodes", [])
            st.rerun()

    # Get current state
    selected_nodes = get_session_var("explore_results", "selected_classifications", [])
    expanded_nodes = get_session_var("explore_results", "expanded_cls_nodes", [])

    # Tree widget
    with st.container(border=True, height=400):
        selected = checkbox_tree(
            nodes,
            check_model="leaf",
            checked=selected_nodes,
            expanded=expanded_nodes,
            show_expand_all=True,
            half_check_color="#086164",
            check_color="#086164",
            key="cls_tree_select",
            show_tree_lines=True,
            tree_line_color="#e9e9eb"
        )

    # Update state on change
    if selected:
        new_checked = selected.get("checked", [])
        new_expanded = selected.get("expanded", [])
        if new_checked != selected_nodes or new_expanded != expanded_nodes:
            set_session_var("explore_results", "selected_classifications", new_checked)
            set_session_var("explore_results", "expanded_cls_nodes", new_expanded)
            st.rerun()

    # Cancel / Apply buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(":material/cancel: Cancel", width='stretch', key="cls_cancel"):
            set_session_var("explore_results", "show_modal_cls_selector", False)
            st.rerun()
    with col2:
        if st.button(":material/check: Apply", width='stretch', type="primary", key="cls_apply"):
            set_session_var("explore_results", "show_modal_cls_selector", False)
            st.rerun()
