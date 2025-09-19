"""
AddaxAI Explore Results Utilities

Utility functions for the explore results page including:
- Image thumbnail generation with base64 encoding
- Filter state persistence and management
- AgGrid configuration and options
- Error handling for missing/corrupted images
"""

import os
import json
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import streamlit as st
from utils.config import ADDAXAI_ROOT, log
from utils.common import load_vars, update_vars


def create_thumbnail_base64(image_path, size=(100, 100)):
    """
    Create a base64-encoded thumbnail from an image file.
    
    Args:
        image_path (str): Path to the source image
        size (tuple): Thumbnail size as (width, height)
        
    Returns:
        str: Base64-encoded image data URL or placeholder if image not found
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return create_placeholder_thumbnail(size, "No Image")
        
        # Open and create thumbnail
        with Image.open(image_path) as img:
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create thumbnail maintaining aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Create a new image with the exact requested size and center the thumbnail
            thumb = Image.new('RGB', size, (240, 240, 240))  # Light gray background
            
            # Center the thumbnail
            x = (size[0] - img.width) // 2
            y = (size[1] - img.height) // 2
            thumb.paste(img, (x, y))
            
            # Convert to base64
            buffer = io.BytesIO()
            thumb.save(buffer, format='JPEG', quality=85)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_data}"
            
    except Exception as e:
        log(f"Error creating thumbnail for {image_path}: {str(e)}")
        return create_placeholder_thumbnail(size, "Error")


def create_placeholder_thumbnail(size=(100, 100), text="No Image"):
    """
    Create a placeholder thumbnail with text.
    
    Args:
        size (tuple): Image size as (width, height)
        text (str): Text to display on placeholder
        
    Returns:
        str: Base64-encoded placeholder image data URL
    """
    try:
        # Create placeholder image
        img = Image.new('RGB', size, (200, 200, 200))  # Gray background
        draw = ImageDraw.Draw(img)
        
        # Try to use a font, fall back to default if not available
        try:
            font = ImageFont.truetype("Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, fill=(100, 100, 100), font=font)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_data = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_data}"
        
    except Exception as e:
        log(f"Error creating placeholder thumbnail: {str(e)}")
        # Return a minimal base64 image if all else fails
        return "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="


def load_filter_state():
    """
    Load saved filter and display state from persistent storage.
    
    Returns:
        dict: Filter state configuration
    """
    try:
        vars_data = load_vars("explore_results")
        return vars_data.get("filter_state", {
            "page_size": 50,
            "thumbnail_size": 100,
            "columns_visible": True,
            "grid_filters": {},
            "sort_model": []
        })
    except Exception as e:
        log(f"Error loading filter state: {str(e)}")
        return {
            "page_size": 50,
            "thumbnail_size": 100,
            "columns_visible": True,
            "grid_filters": {},
            "sort_model": []
        }


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


def configure_aggrid_options(df, page_size=50):
    """
    Configure AgGrid options for the results table.
    
    Args:
        df (pd.DataFrame): DataFrame to display
        page_size (int): Number of rows per page
        
    Returns:
        dict: AgGrid configuration options
    """
    # Column definitions with appropriate types and formatting
    column_defs = []
    
    for col in df.columns:
        col_def = {
            "field": col,
            "headerName": col.replace('_', ' ').title(),
            "sortable": True,
            "filter": True,
            "resizable": True,
        }
        
        # Special handling for specific columns
        if col == "thumbnail":
            col_def.update({
                "headerName": "Image",
                "cellRenderer": "agGroupCellRenderer",
                "cellRendererParams": {
                    "innerRenderer": "imageRenderer"
                },
                "width": 120,
                "filter": False,
                "sortable": False,
                "pinned": "left"
            })
        elif "confidence" in col:
            col_def.update({
                "type": "numericColumn",
                "valueFormatter": "value ? value.toFixed(3) : ''",
                "width": 100,
                "filter": "agNumberColumnFilter"
            })
        elif col in ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]:
            col_def.update({
                "type": "numericColumn",
                "valueFormatter": "value ? value.toFixed(4) : ''",
                "width": 90,
                "filter": "agNumberColumnFilter"
            })
        elif col in ["latitude", "longitude"]:
            col_def.update({
                "type": "numericColumn",
                "valueFormatter": "value ? value.toFixed(6) : ''",
                "width": 110,
                "filter": "agNumberColumnFilter"
            })
        elif col in ["image_width", "image_height"]:
            col_def.update({
                "type": "numericColumn",
                "width": 80,
                "filter": "agNumberColumnFilter"
            })
        elif col == "timestamp":
            col_def.update({
                "width": 150,
                "filter": "agDateColumnFilter"
            })
        elif col in ["absolute_path"]:
            col_def.update({
                "width": 300,
                "hide": True  # Hide by default due to length
            })
        else:
            col_def.update({
                "width": 120,
                "filter": "agTextColumnFilter"
            })
        
        column_defs.append(col_def)
    
    # Grid options
    grid_options = {
        "columnDefs": column_defs,
        "pagination": True,
        "paginationPageSize": page_size,
        "paginationPageSizeSelector": [25, 50, 100, 200],
        "defaultColDef": {
            "flex": 0,
            "minWidth": 80,
            "sortable": True,
            "filter": True,
            "floatingFilter": True,
        },
        "enableRangeSelection": True,
        "rowSelection": "single",
        "suppressRowClickSelection": False,
        "animateRows": True,
        "suppressColumnVirtualisation": False,
        "suppressRowVirtualisation": False,
        "rowHeight": 110,  # Accommodate thumbnail height
    }
    
    return grid_options


def add_thumbnails_to_dataframe(df, thumbnail_size=100):
    """
    Add thumbnail column to dataframe with base64-encoded images.
    
    Args:
        df (pd.DataFrame): DataFrame with absolute_path column
        thumbnail_size (int): Size of thumbnails in pixels
        
    Returns:
        pd.DataFrame: DataFrame with added thumbnail column
    """
    if df.empty:
        return df
    
    # Create thumbnail column
    size = (thumbnail_size, thumbnail_size)
    
    # Use progress bar for thumbnail generation
    with st.spinner("Generating thumbnails..."):
        df_copy = df.copy()
        # Create HTML img tags for AgGrid with allow_unsafe_jscode=True
        df_copy["thumbnail"] = df_copy["absolute_path"].apply(
            lambda path: f'<img src="{create_thumbnail_base64(path, size)}" width="{thumbnail_size}" height="{thumbnail_size}" style="border-radius: 4px; object-fit: cover; cursor: pointer;">'
        )
    
    return df_copy


def format_dataframe_for_display(df):
    """
    Format dataframe values for better display in AgGrid.
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Formatted dataframe
    """
    if df.empty:
        return df
    
    df_display = df.copy()
    
    # Format confidence values
    confidence_cols = [col for col in df_display.columns if "confidence" in col]
    for col in confidence_cols:
        df_display[col] = df_display[col].round(3)
    
    # Format bbox coordinates
    bbox_cols = [col for col in df_display.columns if col.startswith("bbox_")]
    for col in bbox_cols:
        df_display[col] = df_display[col].round(4)
    
    # Format lat/lon
    if "latitude" in df_display.columns:
        df_display["latitude"] = df_display["latitude"].round(6)
    if "longitude" in df_display.columns:
        df_display["longitude"] = df_display["longitude"].round(6)
    
    # Fill NaN values with empty strings for better display
    df_display = df_display.fillna("")
    
    return df_display