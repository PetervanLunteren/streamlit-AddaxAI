"""
AddaxAI Data Browser Utilities

Utility functions for the data browser page.
Provides image modal viewer, filter state management, and thumbnail generation for AgGrid data browsing.
"""

import json
import os
import base64
import io
import math
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd
from utils.config import ADDAXAI_ROOT, log
from utils.common import load_vars, update_vars
from components.shortcut_utils import register_shortcuts, clear_shortcut_listeners

# Thumbnail generation constants
MODAL_IMAGE_SIZE = 1280  # High-resolution modal images (pixels) used for export/zoom
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
            
            target_width, target_height = max_size
            target_width = max(1, int(target_width))
            target_height = max(1, int(target_height))
            target_ratio = target_width / target_height if target_height else 4 / 3

            crop_info = None
            img_width, img_height = img.size

            if bbox_data and all(pd.notna([
                bbox_data['x'], bbox_data['y'],
                bbox_data['width'], bbox_data['height']
            ])):
                x = float(bbox_data['x']) * img_width
                y = float(bbox_data['y']) * img_height
                w = max(1.0, float(bbox_data['width']) * img_width)
                h = max(1.0, float(bbox_data['height']) * img_height)

                padded_w = w + (IMAGE_PADDING_PIXELS * 2)
                padded_h = h + (IMAGE_PADDING_PIXELS * 2)

                crop_width = padded_w
                crop_height = padded_h
                current_ratio = crop_width / crop_height if crop_height else target_ratio
                if current_ratio > target_ratio:
                    crop_height = crop_width / target_ratio
                else:
                    crop_width = crop_height * target_ratio

                bbox_center_x = x + w / 2
                bbox_center_y = y + h / 2

                x1 = bbox_center_x - crop_width / 2
                y1 = bbox_center_y - crop_height / 2
                x2 = x1 + crop_width
                y2 = y1 + crop_height

                crop_box = (
                    int(math.floor(x1)),
                    int(math.floor(y1)),
                    int(math.ceil(x2)),
                    int(math.ceil(y2)),
                )
                img = img.crop(crop_box)

                crop_width = crop_box[2] - crop_box[0]
                crop_height = crop_box[3] - crop_box[1]
                bbox_x_in_crop = x - crop_box[0]
                bbox_y_in_crop = y - crop_box[1]
                crop_info = {
                    'bbox_x_in_crop': bbox_x_in_crop,
                    'bbox_y_in_crop': bbox_y_in_crop,
                    'bbox_w': w,
                    'bbox_h': h,
                    'crop_width': crop_width,
                    'crop_height': crop_height,
                }
            else:
                crop_width = img_width
                crop_height = img_height
                current_ratio = crop_width / crop_height if crop_height else target_ratio
                if current_ratio > target_ratio:
                    crop_height = crop_width / target_ratio
                else:
                    crop_width = crop_height * target_ratio

                x1 = (img_width - crop_width) / 2
                y1 = (img_height - crop_height) / 2
                x2 = x1 + crop_width
                y2 = y1 + crop_height
                crop_box = (
                    int(math.floor(x1)),
                    int(math.floor(y1)),
                    int(math.ceil(x2)),
                    int(math.ceil(y2)),
                )
                img = img.crop(crop_box)
                crop_width = crop_box[2] - crop_box[0]
                crop_height = crop_box[3] - crop_box[1]

            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            thumbnail_size = img.size

            if crop_info:
                draw = ImageDraw.Draw(img)
                crop_width = crop_info['crop_width']
                crop_height = crop_info['crop_height']
                scale_x = target_width / crop_width
                scale_y = target_height / crop_height

                bbox_x_thumb = crop_info['bbox_x_in_crop'] * scale_x
                bbox_y_thumb = crop_info['bbox_y_in_crop'] * scale_y
                bbox_w_thumb = crop_info['bbox_w'] * scale_x
                bbox_h_thumb = crop_info['bbox_h'] * scale_y

                x1 = int(max(0, bbox_x_thumb))
                y1 = int(max(0, bbox_y_thumb))
                x2 = int(min(target_width - 1, bbox_x_thumb + bbox_w_thumb))
                y2 = int(min(target_height - 1, bbox_y_thumb + bbox_h_thumb))

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
    Generate the modal preview image with the same 4:3, detection-centered crop
    used by the observation thumbnails.
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='

        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            modal_width = MODAL_IMAGE_SIZE
            modal_height = int(round(MODAL_IMAGE_SIZE * 3 / 4))
            target_ratio = modal_width / modal_height

            img_width, img_height = img.size
            crop_info = None

            def _has_bbox(data):
                return data and all(pd.notna([data['x'], data['y'], data['width'], data['height']]))

            if _has_bbox(bbox_data):
                x = float(bbox_data['x']) * img_width
                y = float(bbox_data['y']) * img_height
                w = max(1.0, float(bbox_data['width']) * img_width)
                h = max(1.0, float(bbox_data['height']) * img_height)

                padded_w = w + (IMAGE_PADDING_PIXELS * 2)
                padded_h = h + (IMAGE_PADDING_PIXELS * 2)

                crop_width = padded_w
                crop_height = padded_h
                current_ratio = crop_width / crop_height if crop_height else target_ratio
                if current_ratio > target_ratio:
                    crop_height = crop_width / target_ratio
                else:
                    crop_width = crop_height * target_ratio

                bbox_center_x = x + w / 2
                bbox_center_y = y + h / 2

                x1 = bbox_center_x - crop_width / 2
                y1 = bbox_center_y - crop_height / 2
                x2 = x1 + crop_width
                y2 = y1 + crop_height

                crop_box = (
                    int(math.floor(x1)),
                    int(math.floor(y1)),
                    int(math.ceil(x2)),
                    int(math.ceil(y2)),
                )
                img = img.crop(crop_box)

                crop_width = crop_box[2] - crop_box[0]
                crop_height = crop_box[3] - crop_box[1]
                bbox_x_in_crop = x - crop_box[0]
                bbox_y_in_crop = y - crop_box[1]
                crop_info = {
                    'bbox_x_in_crop': bbox_x_in_crop,
                    'bbox_y_in_crop': bbox_y_in_crop,
                    'bbox_w': w,
                    'bbox_h': h,
                    'crop_width': crop_width,
                    'crop_height': crop_height,
                }
            else:
                crop_width = img_width
                crop_height = img_height
                current_ratio = crop_width / crop_height if crop_height else target_ratio
                if current_ratio > target_ratio:
                    crop_height = crop_width / target_ratio
                else:
                    crop_width = crop_height * target_ratio

                x1 = (img_width - crop_width) / 2
                y1 = (img_height - crop_height) / 2
                x2 = x1 + crop_width
                y2 = y1 + crop_height
                crop_box = (
                    int(math.floor(x1)),
                    int(math.floor(y1)),
                    int(math.ceil(x2)),
                    int(math.ceil(y2)),
                )
                img = img.crop(crop_box)
                crop_width = crop_box[2] - crop_box[0]
                crop_height = crop_box[3] - crop_box[1]

            img = img.resize((modal_width, modal_height), Image.Resampling.LANCZOS)
            thumbnail_size = img.size

            bbox_coords = None
            if crop_info and show_bbox:
                scale_x = modal_width / crop_info['crop_width']
                scale_y = modal_height / crop_info['crop_height']

                bbox_x_thumb = crop_info['bbox_x_in_crop'] * scale_x
                bbox_y_thumb = crop_info['bbox_y_in_crop'] * scale_y
                bbox_w_thumb = crop_info['bbox_w'] * scale_x
                bbox_h_thumb = crop_info['bbox_h'] * scale_y

                x1 = int(max(0, bbox_x_thumb))
                y1 = int(max(0, bbox_y_thumb))
                x2 = int(min(modal_width - 1, bbox_x_thumb + bbox_w_thumb))
                y2 = int(min(modal_height - 1, bbox_y_thumb + bbox_h_thumb))

                bbox_coords = (x1, y1, x2, y2)

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

            if bbox_coords and show_bbox:
                x1, y1, x2, y2 = bbox_coords
                draw = ImageDraw.Draw(img)

                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_min_dim = max(1, min(bbox_width, bbox_height))

                if bbox_min_dim < 100:
                    line_width = 1
                elif bbox_min_dim < 300:
                    line_width = 2
                else:
                    line_width = 3

                draw.rectangle([x1, y1, x2, y2], outline=IMAGE_BBOX_COLOR, width=line_width)

                if show_labels and hasattr(generate_modal_image, '_current_label_text'):
                    label_text = generate_modal_image._current_label_text
                    font_size = max(10, min(24, int(bbox_min_dim * 0.08)))

                    try:
                        from PIL import ImageFont
                        font = None
                        for font_name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
                            try:
                                font = ImageFont.truetype(font_name, font_size)
                                break
                            except Exception:
                                continue
                        if font is None:
                            font = ImageFont.load_default()
                    except Exception:
                        font = None

                    lines = label_text.split('\n')
                    line_heights = []
                    max_width = 0
                    if font is not None:
                        for line in lines:
                            bbox_text = draw.textbbox((0, 0), line, font=font)
                            width = bbox_text[2] - bbox_text[0]
                            height = bbox_text[3] - bbox_text[1]
                            line_heights.append(height)
                            max_width = max(max_width, width)
                    else:
                        for line in lines:
                            width = len(line) * 6
                            height = 10
                            line_heights.append(height)
                            max_width = max(max_width, width)

                    line_spacing = 2
                    total_height = sum(line_heights) + line_spacing * (len(lines) - 1)

                    text_x = x1
                    text_y = max(4, y1 - total_height - 6)

                    if text_y < 4:
                        text_y = min(y2 + 6, thumbnail_size[1] - total_height - 4)

                    padding = 4
                    bg_x1 = text_x - padding
                    bg_y1 = text_y - padding
                    bg_x2 = text_x + max_width + padding
                    bg_y2 = text_y + total_height + padding

                    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 200))

                    img_with_overlay = Image.alpha_composite(img.convert('RGBA'), overlay)
                    img = img_with_overlay.convert('RGB')
                    draw = ImageDraw.Draw(img)

                    y_offset = text_y
                    for i, line in enumerate(lines):
                        draw.text((text_x, y_offset), line, fill='white', font=font)
                        y_offset += line_heights[i] + line_spacing

            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
            img_data = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/jpeg;base64,{img_data}"
    except Exception:
        return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='


def image_to_base64_with_boxes(image_path, detection_details, max_height, max_width=None):
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
            aspect_ratio = 4 / 3
            target_width = max_width if max_width else int(max_height * aspect_ratio)
            target_height = max_height
            # Only cap height for table thumbnails, not for high-res exports
            if max_height <= 250:
                target_height = max(60, min(target_height, 250))
                target_width = (
                    max(80, int(target_height * aspect_ratio))
                    if not max_width
                    else target_width
                )

            scale = min(target_width / orig_width, target_height / orig_height)
            new_width = max(1, int(orig_width * scale))
            new_height = max(1, int(orig_height * scale))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

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
                y1 = int(y_norm * new_height)  # Fixed: use new_height instead of max_height
                x2 = int((x_norm + w_norm) * new_width)
                y2 = int((y_norm + h_norm) * new_height)  # Fixed: use new_height instead of max_height

                line_width = max(1, int(min(new_width, max_height) * 0.003))
                draw.rectangle([x1, y1, x2, y2], outline=IMAGE_BBOX_COLOR, width=line_width)

            img = _fit_image_with_letterbox(img, target_width, target_height)

            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_data}"
    except Exception:
        return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='


def build_event_collage_base64(event_files, max_grid=4, thumb_height=180, thumb_width=None,
                                event_data=None, show_bbox=True):
    """
    Build a collage of event images with bounding boxes for event-specific detections only.

    Args:
        event_files: List of file dicts with relative_path and absolute_path
        max_grid: Maximum grid size (default 4x4)
        thumb_height: Thumbnail height in pixels
        thumb_width: Thumbnail width in pixels (default 4:3 ratio)
        event_data: Dict with event metadata for filtering detections:
            - run_id: Analysis run ID
            - project_id: Project ID
            - location_id: Location ID
            - start_timestamp: Event start time (ISO format)
            - end_timestamp: Event end time (ISO format)
            - dominant_species: Primary species in event

    Returns:
        Base64 encoded image string
    """
    if not event_files:
        return None

    import streamlit as st

    detections_df = st.session_state.get("observations_source_df", pd.DataFrame())

    # Filter detections to only those belonging to this specific event
    # Now we can use event_id for a simple, fast, accurate filter!
    if event_data and not detections_df.empty:
        event_id = event_data.get("event_id")
        if event_id and "event_id" in detections_df.columns:
            # Simple index-based lookup - much faster than complex filtering
            detections_df = detections_df[detections_df["event_id"] == event_id]

    if thumb_width is None:
        thumb_width = int(thumb_height * 4 / 3)

    total_files = len(event_files)
    if total_files == 0:
        return None

    # Use square grids (1x1, 2x2, 3x3, 4x4) for consistent aspect ratio
    grid_size = min(max_grid, max(1, math.ceil(math.sqrt(total_files))))
    slots = grid_size * grid_size

    collage = Image.new(
        "RGB",
        (grid_size * thumb_width, grid_size * thumb_height),
        color=(231, 239, 239),  # #e7efef
    )

    for idx, file_info in enumerate(event_files[:slots]):
        abs_path = file_info.get("absolute_path")
        rel_path = file_info.get("relative_path")
        if not abs_path or not os.path.exists(abs_path):
            continue

        detection_details = _lookup_detection_details(rel_path, detections_df) if show_bbox else []
        thumb_b64 = image_to_base64_with_boxes(
            abs_path,
            detection_details=detection_details,
            max_height=thumb_height,
            max_width=thumb_width,
        )
        thumb_img = _base64_to_image(thumb_b64)
        if thumb_img is None:
            continue
        row = idx // grid_size
        col = idx % grid_size
        collage.paste(thumb_img, (col * thumb_width, row * thumb_height))

    return _image_to_base64(collage)


def _fit_image_with_letterbox(image, box_width, box_height=None):
    if image is None:
        return None
    if box_height is None:
        box_height = box_width

    img = image.copy()
    img_width, img_height = img.size
    if img_width == 0 or img_height == 0:
        return None

    scale = min(box_width / img_width, box_height / img_height)
    new_width = max(1, int(img_width * scale))
    new_height = max(1, int(img_height * scale))
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (box_width, box_height), color=(231, 239, 239))  # #e7efef
    offset_x = (box_width - new_width) // 2
    offset_y = (box_height - new_height) // 2
    canvas.paste(img, (offset_x, offset_y))
    return canvas


def _lookup_detection_details(relative_path, detections_df):
    if relative_path is None or detections_df is None or detections_df.empty:
        return []

    matched = detections_df[detections_df["relative_path"] == relative_path]
    if matched.empty:
        return []

    details = []
    for _, det in matched.iterrows():
        bbox = {
            "x": det.get("bbox_x"),
            "y": det.get("bbox_y"),
            "width": det.get("bbox_width"),
            "height": det.get("bbox_height"),
        }
        if any(pd.isna(v) for v in bbox.values()):
            continue
        details.append({"bbox": bbox})

    return details


def _base64_to_image(b64_string):
    if not b64_string:
        return None
    if "," in b64_string:
        _, encoded = b64_string.split(",", 1)
    else:
        encoded = b64_string
    try:
        return Image.open(io.BytesIO(base64.b64decode(encoded)))
    except Exception:
        return None


def _image_to_base64(image):
    if image is None:
        return None
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"

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

        st.caption("Use shortcuts `←` `→` to navigate and `Esc` to close")

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

            # File
            abs_path = current_row.get('absolute_path', 'N/A')
            metadata_rows.append(("File", abs_path))

            # Location
            location_id = current_row.get('location_id', 'N/A')
            metadata_rows.append(("Location", location_id))

            # Timestamp
            metadata_rows.append(("Timestamp", timestamp_display))

            # Detection
            det_label = current_row.get("detection_label") or "N/A"
            det_conf = current_row.get("detection_confidence")
            detection_display = det_label
            if det_conf is not None and not pd.isna(det_conf):
                detection_display = f"{det_label} {float(det_conf):.2f}"
            metadata_rows.append(("Detection", detection_display))

            # Classification
            cls_label = current_row.get("classification_label") or "N/A"
            cls_conf = current_row.get("classification_confidence")
            classification_display = cls_label
            if cls_conf is not None and not pd.isna(cls_conf):
                classification_display = f"{cls_label} {float(cls_conf):.2f}"
            metadata_rows.append(("Classification", classification_display))

            # Row
            metadata_rows.append(("Row", f"{current_index + 1} of {len(df_filtered)}"))

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

        st.caption("Use shortcuts `←` `→` to navigate and `Esc` to close")

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
                ("File", current_row.get("absolute_path", "N/A")),
                ("Location", current_row.get("location_id", "N/A")),
                ("Timestamp", timestamp_display),
                ("Detections", detections_summary),
                ("Classifications", classifications_summary),
                ("Row", f"{current_index + 1} of {len(df_files)}"),
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
