"""
AddaxAI Data Browser

Interactive data browser showing cropped images with detection labels using AgGrid.
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
from components import print_widget_label
from utils.data_browser_utils import (
    image_viewer_modal, image_to_base64_url,
    IMAGE_PADDING_PIXELS, IMAGE_BLUR_RADIUS, IMAGE_CORNER_RADIUS,
    IMAGE_BBOX_COLOR, IMAGE_QUALITY
)
from st_modal import Modal

def parse_timestamps(timestamp_series):
    """Parse timestamps in EXIF format: 2013:01:17 13:05:40"""
    return pd.to_datetime(timestamp_series, format='%Y:%m:%d %H:%M:%S')


def render_file_level_browser(files_df: pd.DataFrame):
    """Render the file-level data browser view using aggregated detections."""
    if files_df is None:
        st.error("File-level dataset was not initialized. Please restart the application.")
        st.stop()

    if files_df.empty:
        st.warning("No files containing detections were found.")
        st.stop()

    FILE_ROW_HEIGHT = 120
    DEFAULT_PAGE_SIZE = 20
    PAGE_SIZE_OPTIONS = [20, 50, 100]

    # Initialize pagination state
    if "file_grid_page_size" not in st.session_state:
        st.session_state.file_grid_page_size = DEFAULT_PAGE_SIZE
    if "file_grid_current_page" not in st.session_state:
        st.session_state.file_grid_current_page = 1

    page_size = st.session_state.file_grid_page_size
    current_page = st.session_state.file_grid_current_page

    total_rows = len(files_df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    if current_page > total_pages:
        current_page = total_pages
        st.session_state.file_grid_current_page = current_page

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    df_page = files_df.iloc[start_idx:end_idx].copy()

    thumbnail_cache_key = (
        f"files_page_{current_page}_size_{page_size}_height_{FILE_ROW_HEIGHT}"
    )
    if "file_thumbnail_cache" not in st.session_state:
        st.session_state.file_thumbnail_cache = {}

    thumbnails = st.session_state.file_thumbnail_cache.get(thumbnail_cache_key)

    if thumbnails is None:
        thumbnails = []
        for _, row in df_page.iterrows():
            image_url = image_to_base64_url(
                row.get("absolute_path"),
                bbox_data=None,
                max_size=(FILE_ROW_HEIGHT, FILE_ROW_HEIGHT),
            )
            thumbnails.append(image_url)
        st.session_state.file_thumbnail_cache.clear()
        st.session_state.file_thumbnail_cache[thumbnail_cache_key] = thumbnails

    display_rows = []
    for (idx, row), image_url in zip(df_page.iterrows(), thumbnails):
        timestamp_first = (
            pd.to_datetime(row.get("timestamp_first"))
            if row.get("timestamp_first")
            else None
        )
        timestamp_last = (
            pd.to_datetime(row.get("timestamp_last"))
            if row.get("timestamp_last")
            else None
        )

        detections_count_value = row.get("detections_count")
        detections_count = (
            int(detections_count_value)
            if detections_count_value is not None and not pd.isna(detections_count_value)
            else 0
        )

        classifications_count_value = row.get("classifications_count")
        classifications_count = (
            int(classifications_count_value)
            if classifications_count_value is not None and not pd.isna(classifications_count_value)
            else 0
        )

        display_rows.append(
            {
                "_df_index": idx,
                "image": image_url,
                "relative_path": row.get("relative_path") or "",
                "detections_count": detections_count,
                "detections_summary": row.get("detections_summary") or "",
                "classifications_count": classifications_count,
                "classifications_summary": row.get("classifications_summary") or "",
                "timestamp_first": timestamp_first.strftime("%Y-%m-%d %H:%M:%S")
                if timestamp_first is not None
                else "",
                "timestamp_last": timestamp_last.strftime("%Y-%m-%d %H:%M:%S")
                if timestamp_last is not None
                else "",
                "location_id": row.get("location_id") or "",
                "run_id": row.get("run_id") or "",
            }
        )

    display_df = pd.DataFrame(display_rows)

    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_column("_df_index", hide=True)

    gb.configure_column(
        "image",
        headerName="Image",
        width=FILE_ROW_HEIGHT + 20,
        autoHeight=True,
        cellRenderer=JsCode(
            f"""
            class ImageRenderer {{
                init(params) {{
                    const img = document.createElement('img');
                    if (params.value) {{
                        img.src = params.value;
                        img.style.width = '{FILE_ROW_HEIGHT}px';
                        img.style.height = '{FILE_ROW_HEIGHT}px';
                        img.style.objectFit = 'contain';
                        img.style.border = 'none';
                    }}
                    this.eGui = document.createElement('div');
                    this.eGui.appendChild(img);
                }}
                getGui() {{
                    return this.eGui;
                }}
            }}
            """
        ),
    )

    gb.configure_column("relative_path", headerName="File", width=220)
    gb.configure_column("detections_count", headerName="Detections", width=110, type="numericColumn")
    gb.configure_column("detections_summary", headerName="Detection labels", width=200)
    gb.configure_column("classifications_count", headerName="Classifications", width=130, type="numericColumn")
    gb.configure_column("classifications_summary", headerName="Classification labels", width=220)
    gb.configure_column("timestamp_first", headerName="First seen", width=160)
    gb.configure_column("timestamp_last", headerName="Last seen", width=160)
    gb.configure_column("location_id", headerName="Location", width=120)
    gb.configure_column("run_id", headerName="Run", width=120)

    gb.configure_default_column(
        resizable=True,
        sortable=False,
        filter=False,
        cellStyle={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-start'},
        headerClass='ag-left-aligned-header'
    )

    gb.configure_selection("none")
    gb.configure_grid_options(rowHeight=FILE_ROW_HEIGHT + 10)

    grid_options = gb.build()

    grid_height = (len(display_df) * (FILE_ROW_HEIGHT + 10)) + 55

    AgGrid(
        display_df,
        gridOptions=grid_options,
        height=grid_height,
        allow_unsafe_jscode=True,
        theme="streamlit",
        fit_columns_on_grid_load=False,
    )

    bottom_menu = st.columns((4, 1, 1))
    with bottom_menu[0]:
        st.markdown(
            f"Page **{current_page}** of **{total_pages}** "
            f"(Showing files {start_idx + 1}-{end_idx} of {total_rows:,})"
        )

    with bottom_menu[1]:
        new_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            step=1,
            value=current_page,
            key="file_grid_page_input",
        )
        if new_page != current_page:
            st.session_state.file_grid_current_page = new_page
            st.rerun()

    with bottom_menu[2]:
        new_page_size = st.selectbox(
            "Page Size",
            options=PAGE_SIZE_OPTIONS,
            index=PAGE_SIZE_OPTIONS.index(page_size),
            key="file_grid_page_size_input",
        )
        if new_page_size != page_size:
            st.session_state.file_grid_page_size = new_page_size
            st.session_state.file_grid_current_page = 1
            st.rerun()


# View selection configuration
BROWSER_VIEWS = [
    (":material/crop_free:", "Detections"),
    (":material/photo:", "Files"),
    (":material/event:", "Events"),
]
VIEW_LABELS = [label for _, label in BROWSER_VIEWS]
ICON_LOOKUP = {label: icon for icon, label in BROWSER_VIEWS}


def on_view_mode_change():
    """Persist the selected browse level in session state."""
    selection = st.session_state.get("data_browser_level")
    if selection:
        set_session_var("explore_results", "browser_view_mode", selection)


def render_view_mode_control(current_view):
    """Render segmented control row and return popover columns."""
    columns = st.columns((8, 1, 1, 1, 1, 1, 1))
    (
        level_col,
        spacer_one,
        spacer_two,
        sort_col,
        filter_col,
        export_col,
        settings_col,
    ) = columns

    with level_col:
        st.segmented_control(
            "Browse level",
            options=VIEW_LABELS,
            format_func=lambda label: f"{ICON_LOOKUP[label]} {label}",
            default=current_view,
            key="data_browser_level",
            selection_mode="single",
            label_visibility="collapsed",
            width="stretch",
            on_change=on_view_mode_change,
        )

    return sort_col, filter_col, export_col, settings_col

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

# Display settings (thumbnail generation constants are imported from utils)
DEFAULT_SIZE_OPTION = "medium"  # Default size selection

# Legacy padding settings (no longer used)
IMAGE_PADDING_PERCENT = 0.01
IMAGE_PADDING_MIN = 10

# Page config
st.set_page_config(layout="wide")

# Load filter settings from config
filter_config = load_vars("explore_results")
saved_settings = filter_config.get("aggrid_settings", {})

# Get saved sort settings or set defaults
saved_sort_column = saved_settings.get('sort_column', 'timestamp')
saved_sort_direction = saved_settings.get('sort_direction', '↓')  # Descending for newest first

# Check if detection results are available
if "results_detections" not in st.session_state:
    st.error("No detection results found. Please ensure detection data is loaded.")
    st.stop()

# Load data
df = st.session_state["results_detections"]

if df.empty:
    st.warning("No detection results found.")
    st.stop()

# Determine current view mode and render segmented control row
current_view = get_session_var("explore_results", "browser_view_mode", VIEW_LABELS[0])
set_session_var("explore_results", "browser_view_mode", current_view)

sort_col, filter_col, export_col, settings_col = render_view_mode_control(current_view)

# Reread in case selection changed during control rendering
current_view = get_session_var("explore_results", "browser_view_mode", VIEW_LABELS[0])

if current_view == "Files":
    files_df = st.session_state.get("results_files")
    render_file_level_browser(files_df)
    st.stop()
elif current_view == "Events":
    st.info("Event-level browser coming soon.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK IF MODAL SHOULD BE SHOWN - GUARD HEAVY CODE
# ═══════════════════════════════════════════════════════════════════════════════

# Check if image viewer modal should be displayed
if get_session_var("explore_results", "show_modal_image_viewer", False):
    # Only render the modal, skip all heavy grid processing
    modal_image_viewer = Modal(
        title="",
        key="image_viewer",
        show_close_button=False,
        show_title=False,
        show_divider=False
    )
    with modal_image_viewer.container():
        image_viewer_modal()
    st.stop()  # Don't render anything else

# Check if tree selector modal should be displayed
if get_session_var("explore_results", "show_tree_modal", False):
    from components.taxonomic_tree_selector import tree_selector_modal

    # Build classification universe from full dataset only
    raw_df = st.session_state.get("results_raw", df)
    if 'classification_label' in raw_df.columns:
        unique_classifications = sorted([
            cls for cls in raw_df['classification_label'].dropna().unique()
            if cls != 'N/A' and cls.strip() != ''
        ])
    else:
        unique_classifications = []

    # Get current selections from saved settings
    saved_selected_classifications = saved_settings.get('selected_classifications', unique_classifications)

    # Open modal using st-modal library
    tree_modal = Modal(
        title="Select Species",
        key="tree_selector_modal",
        show_close_button=False,
        show_title=False,
        show_divider=False
    )

    with tree_modal.container():
        # Render tree selector modal
        result = tree_selector_modal(
            available=unique_classifications,
            selected=saved_selected_classifications,
            key="explore_results_tree",
            title_mode="browser"
        )

        # Handle result
        if result is not None:  # Apply was clicked
            # Save selection
            new_settings = saved_settings.copy()
            new_settings['selected_classifications'] = result
            update_vars("explore_results", {"aggrid_settings": new_settings})

            # Close modal
            set_session_var("explore_results", "show_tree_modal", False)
            st.session_state.pop("explore_results_tree_dismissed", None)

            # Clear caches and rerun
            if 'aggrid_thumbnail_cache' in st.session_state:
                del st.session_state['aggrid_thumbnail_cache']
            if 'results_modified' in st.session_state:
                del st.session_state['results_modified']
            st.session_state.aggrid_current_page = 1
            st.rerun()
        elif st.session_state.pop("explore_results_tree_dismissed", None) == "cancel":
            set_session_var("explore_results", "show_tree_modal", False)

    st.stop()  # Don't render anything else

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GRID VIEW (Only rendered when modal is NOT open)
# ═══════════════════════════════════════════════════════════════════════════════

# Image size controls both row height and image column width
image_size_options = {
    size: {"height": height, "width": IMAGE_COLUMN_WIDTHS[size]}
    for size, height in ROW_HEIGHT_OPTIONS.items()
}

# ═══════════════════════════════════════════════════════════════════════════════
# VIEW LEVEL SWITCHER
# ═══════════════════════════════════════════════════════════════════════════════

BROWSER_VIEWS = [
    (":material/crop_free:", "Detections"),
    (":material/photo:", "Files"),
    (":material/event:", "Events"),
]
VIEW_LABELS = [label for _, label in BROWSER_VIEWS]
ICON_LOOKUP = {label: icon for icon, label in BROWSER_VIEWS}

# Store raw data (never changes)
if 'results_raw' not in st.session_state:
    st.session_state['results_raw'] = df.copy()

# Load filter settings for creating filtered dataframe
filter_config_early = load_vars("explore_results")
saved_settings_early = filter_config_early.get("aggrid_settings", {})

# Create or get modified dataframe (MOVED BEFORE CONTROLS)
if 'results_modified' not in st.session_state:
    # Start with full dataframe
    filtered_df = st.session_state['results_raw'].copy()

    # Apply date filter if set
    if ('date_start' in saved_settings_early and 'date_end' in saved_settings_early and
        saved_settings_early['date_start'] is not None and saved_settings_early['date_end'] is not None):
        if 'timestamp' in filtered_df.columns:
            df_timestamps = parse_timestamps(filtered_df['timestamp'])
            date_start = pd.to_datetime(saved_settings_early['date_start'])
            date_end = pd.to_datetime(saved_settings_early['date_end'])
            filtered_df = filtered_df[
                (df_timestamps >= date_start) &
                (df_timestamps <= date_end)
            ].copy()

    # Apply detection confidence filter if set
    if ('det_conf_min' in saved_settings_early and 'det_conf_max' in saved_settings_early):
        if 'detection_confidence' in filtered_df.columns:
            det_conf_min = saved_settings_early['det_conf_min']
            det_conf_max = saved_settings_early['det_conf_max']
            filtered_df = filtered_df[
                (filtered_df['detection_confidence'] >= det_conf_min) &
                (filtered_df['detection_confidence'] <= det_conf_max)
            ].copy()

    # Apply classification confidence filter if set
    if ('cls_conf_min' in saved_settings_early and 'cls_conf_max' in saved_settings_early):
        if 'classification_confidence' in filtered_df.columns:
            cls_conf_min = saved_settings_early['cls_conf_min']
            cls_conf_max = saved_settings_early['cls_conf_max']
            include_unclass = saved_settings_early.get('include_unclassified', True)

            if include_unclass:
                # Include both classified (in range) and unclassified (NaN) detections
                mask = (filtered_df['classification_confidence'].isna()) | \
                       ((filtered_df['classification_confidence'] >= cls_conf_min) &
                        (filtered_df['classification_confidence'] <= cls_conf_max))
            else:
                # Only classified detections in range
                mask = (filtered_df['classification_confidence'] >= cls_conf_min) & \
                       (filtered_df['classification_confidence'] <= cls_conf_max)

            filtered_df = filtered_df[mask].copy()

    # Apply detection types filter if set
    if 'selected_detection_types' in saved_settings_early:
        if 'detection_label' in filtered_df.columns:
            selected_det_types = saved_settings_early['selected_detection_types']
            if selected_det_types and len(selected_det_types) > 0:
                filtered_df = filtered_df[
                    filtered_df['detection_label'].isin(selected_det_types)
                ].copy()

    # Apply classification species filter if set
    if 'selected_classifications' in saved_settings_early:
        if 'classification_label' in filtered_df.columns:
            selected_classifications = saved_settings_early['selected_classifications']
            if selected_classifications and len(selected_classifications) > 0:
                # Include unclassified if include_unclassified is True
                include_unclass = saved_settings_early.get('include_unclassified', True)

                raw_classification_series = filtered_df['classification_label']
                classification_series = raw_classification_series.fillna("").astype(str)
                classification_series_norm = classification_series.str.strip()
                selected_norm = {cls.strip().lower() for cls in selected_classifications}

                if include_unclass:
                    # Include both selected classifications and unclassified (NaN or empty)
                    mask = (
                        raw_classification_series.isna() |
                        (classification_series_norm == '') |
                        (classification_series_norm.str.lower() == 'n/a') |
                        (classification_series_norm.str.lower().isin(selected_norm))
                    )
                else:
                    # Only selected classifications
                    mask = classification_series_norm.str.lower().isin(selected_norm)

                filtered_df = filtered_df[mask].copy()

    # Apply location filter if set
    if 'selected_locations' in saved_settings_early:
        if 'location_id' in filtered_df.columns:
            selected_locations = saved_settings_early['selected_locations']
            if selected_locations and len(selected_locations) > 0:
                filtered_df = filtered_df[
                    filtered_df['location_id'].isin(selected_locations)
                ].copy()

    # Apply run filter if set
    if 'selected_runs' in saved_settings_early:
        if 'run_id' in filtered_df.columns:
            selected_runs = saved_settings_early['selected_runs']
            if selected_runs and len(selected_runs) > 0:
                filtered_df = filtered_df[
                    filtered_df['run_id'].isin(selected_runs)
                ].copy()

    # Apply detection model filter if set
    if 'selected_detection_models' in saved_settings_early:
        if 'detection_model_id' in filtered_df.columns:
            selected_det_models = saved_settings_early['selected_detection_models']
            if selected_det_models and len(selected_det_models) > 0:
                filtered_df = filtered_df[
                    filtered_df['detection_model_id'].isin(selected_det_models)
                ].copy()

    # Apply classification model filter if set
    if 'selected_classification_models' in saved_settings_early:
        if 'classification_model_id' in filtered_df.columns:
            selected_cls_models = saved_settings_early['selected_classification_models']
            if selected_cls_models and len(selected_cls_models) > 0:
                filtered_df = filtered_df[
                    filtered_df['classification_model_id'].isin(selected_cls_models)
                ].copy()

    # Apply sorting
    if saved_sort_column in filtered_df.columns:
        ascending = (saved_sort_direction == '↑')

        try:
            # Handle different data types appropriately
            if saved_sort_column in ['detection_confidence', 'classification_confidence']:
                # For confidence columns, handle NaN values by putting them at the end
                filtered_df = filtered_df.sort_values(
                    by=saved_sort_column,
                    ascending=ascending,
                    na_position='last'
                )
            elif saved_sort_column == 'timestamp':
                # For timestamp, convert to datetime for proper sorting
                if 'timestamp' in filtered_df.columns:
                    # Parse timestamps for sorting
                    df_timestamps = parse_timestamps(filtered_df['timestamp'])
                    filtered_df = filtered_df.copy()
                    filtered_df['_temp_timestamp'] = df_timestamps
                    filtered_df = filtered_df.sort_values(
                        by='_temp_timestamp',
                        ascending=ascending,
                        na_position='last'
                    )
                    filtered_df = filtered_df.drop(columns=['_temp_timestamp'])
            else:
                # For other columns (text), sort normally
                filtered_df = filtered_df.sort_values(
                    by=saved_sort_column,
                    ascending=ascending,
                    na_position='last'
                )
        except Exception as e:
            # If sorting fails, continue without sorting and log the issue
            st.warning(f"Could not sort by {saved_sort_column}: {str(e)}")
    elif saved_sort_column != 'timestamp':
        # If the saved sort column doesn't exist, fall back to timestamp if available
        if 'timestamp' in filtered_df.columns:
            try:
                df_timestamps = parse_timestamps(filtered_df['timestamp'])
                filtered_df = filtered_df.copy()
                filtered_df['_temp_timestamp'] = df_timestamps
                filtered_df = filtered_df.sort_values(
                    by='_temp_timestamp',
                    ascending=False,  # Default to newest first
                    na_position='last'
                )
                filtered_df = filtered_df.drop(columns=['_temp_timestamp'])
            except Exception:
                pass  # If timestamp sorting fails, continue without sorting

    # Store filtered and sorted dataframe in session state
    st.session_state['results_modified'] = filtered_df

    # Clear thumbnail cache when results_modified changes
    if 'aggrid_thumbnail_cache' in st.session_state:
        del st.session_state['aggrid_thumbnail_cache']

# Use filtered dataframe for export
df_to_export = st.session_state['results_modified']

with sort_col:
    # Column sorting popover
    with st.popover(":material/swap_vert:", help="Sort", width="stretch"):
        with st.form("sort_form", border=False):
            # Sort container
            with st.container(border=True):
                print_widget_label("Sort by")

                # Available columns for sorting with friendly names
                sortable_columns = {
                    'detection_label': 'Detection',
                    'detection_confidence': 'Detection Confidence',
                    'classification_label': 'Classification',
                    'classification_confidence': 'Classification Confidence',
                    'timestamp': 'Timestamp',
                    'location_id': 'Location'
                }

                # All controls in one row
                sort_col1, sort_col2, sort_col3 = st.columns([3, 1, 1])

                with sort_col1:
                    # Column to sort dropdown
                    try:
                        default_index = list(sortable_columns.keys()).index(saved_sort_column)
                    except ValueError:
                        default_index = 0  # fallback to first option if saved column not found

                    sort_column = st.selectbox(
                        "Column to sort",
                        options=list(sortable_columns.keys()),
                        format_func=lambda x: sortable_columns[x],
                        index=default_index,
                        label_visibility="collapsed"
                    )

                with sort_col2:
                    # Sorting method (up or down)
                    sort_direction = st.segmented_control(
                        "Sorting method",
                        options=["↑", "↓"],
                        default=saved_sort_direction,
                        label_visibility="collapsed",
                        width="stretch"
                    )

                with sort_col3:
                    # Apply sorting button
                    if st.form_submit_button("Apply", width="stretch", type="primary"):
                        # Save sort settings
                        new_settings = saved_settings.copy()
                        new_settings.update({
                            'sort_column': sort_column,
                            'sort_direction': sort_direction
                        })
                        update_vars("explore_results", {"aggrid_settings": new_settings})

                        # Clear modified results to force re-processing with new sort
                        if 'results_modified' in st.session_state:
                            del st.session_state['results_modified']

                        # Clear thumbnail cache since sort order changed
                        if 'aggrid_thumbnail_cache' in st.session_state:
                            del st.session_state['aggrid_thumbnail_cache']

                        # Reset to page 1
                        st.session_state.aggrid_current_page = 1

                        st.rerun()

with filter_col:
    # Simple date filter popover
    with st.popover(":material/tune:", help="Filter", width="stretch"):
        # ═══════════════════════════════════════════════════════════════════════
        # FORM STARTS HERE
        # ═══════════════════════════════════════════════════════════════════════

        with st.form("date_filter_form", border=False):

            # Get unique detection types from current dataframe
            if 'detection_label' in df.columns:
                unique_detection_types = sorted([
                    det_type for det_type in df['detection_label'].dropna().unique()
                    if det_type.strip() != ''
                ])
            else:
                unique_detection_types = []  # No fallback - show only what exists

            # DETECTIONS SECTION
            with st.container(border=True):
                print_widget_label("Detections")

                # Detection types multiselect
                saved_detection_types = saved_settings.get('selected_detection_types', unique_detection_types)
                if unique_detection_types:
                    selected_detection_types = st.multiselect(
                        "Classes",
                        options=unique_detection_types,
                        default=saved_detection_types
                    )
                else:
                    selected_detection_types = []
                    st.info("No detection types available in the data")

                # Detection confidence range
                from utils.config import DEFAULT_DETECTION_CONFIDENCE_THRESHOLD
                saved_det_conf_min = saved_settings.get('det_conf_min', DEFAULT_DETECTION_CONFIDENCE_THRESHOLD)
                saved_det_conf_max = saved_settings.get('det_conf_max', 1.0)
                det_conf_range = st.slider(
                    "Confidence range",
                    min_value=0.01,
                    max_value=1.0,
                    value=(saved_det_conf_min, saved_det_conf_max),
                    step=0.01,
                    format="%.2f",
                    key="detection_confidence_slider"
                )

            # CLASSIFICATIONS SECTION (with modal tree selector)
            with st.container(border=True):
                print_widget_label("Classifications")

                st.markdown(
                    "<span style='color:#353640; font-size:0.85rem;'>Classes</span>",
                    unsafe_allow_html=True
                )

                # Get unique classifications from current dataframe
                if 'classification_label' in df.columns:
                    unique_classifications = sorted([
                        cls for cls in df['classification_label'].dropna().unique()
                        if cls != 'N/A' and cls.strip() != ''
                    ])
                else:
                    unique_classifications = []

                # Get current selections from saved settings
                saved_selected_classifications = saved_settings.get('selected_classifications', unique_classifications)

                selected_count = len(saved_selected_classifications)
                total_count = len(unique_classifications)

                col_btn, col_summary = st.columns([1, 3])
                with col_btn:
                    if st.form_submit_button(
                        ":material/pets: Select",
                        use_container_width=True,
                        key="select_species_button"
                    ):
                        set_session_var("explore_results", "show_tree_modal", True)
                        st.rerun()

                with col_summary:
                    if total_count > 0:
                        text = (
                            "You have selected "
                            f"<code style='color:#086164; font-family:monospace;'>{selected_count}</code>"
                            " of "
                            f"<code style='color:#086164; font-family:monospace;'>{total_count}</code> classes."
                        )
                        st.markdown(
                            f"""
                                <div style=\"background-color: #f0f2f6; padding: 7px; border-radius: 8px;\">
                                    &nbsp;&nbsp;{text}
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("No species classifications available in the current dataset.")

                # Classification confidence range
                saved_cls_conf_min = saved_settings.get('cls_conf_min', 0.01)
                saved_cls_conf_max = saved_settings.get('cls_conf_max', 1.0)
                cls_conf_range = st.slider(
                    "Confidence range",
                    min_value=0.01,
                    max_value=1.0,
                    value=(saved_cls_conf_min, saved_cls_conf_max),
                    step=0.01,
                    format="%.2f",
                    key="classification_confidence_slider"
                )

                # Checkbox for including unclassified detections
                saved_include_unclassified = saved_settings.get('include_unclassified', True)
                include_unclassified = st.checkbox(
                    "Include detections without classification",
                    value=saved_include_unclassified,
                    help="When checked, shows detections that haven't been classified for species"
                )
            
            # DATES SECTION
            with st.container(border=True):
                print_widget_label("Dates")

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
                    format="YYYY-MM-DD",
                    label_visibility="collapsed"
                )

            # LOCATIONS SECTION
            with st.container(border=True):
                print_widget_label("Locations")

                # Get unique locations from current dataframe
                if 'location_id' in df.columns:
                    unique_locations = sorted([
                        loc for loc in df['location_id'].dropna().unique()
                        if loc.strip() != '' and loc != 'NONE'
                    ])
                else:
                    unique_locations = []

                # Location multiselect
                saved_selected_locations = saved_settings.get('selected_locations', unique_locations)
                if unique_locations:
                    selected_locations = st.multiselect(
                        "Locations",
                        options=unique_locations,
                        default=saved_selected_locations,
                        help="Select specific locations to include",
                        label_visibility="collapsed"
                    )
                else:
                    selected_locations = []
                    st.info("No locations available in the data")

            # RUNS SECTION
            with st.container(border=True):
                print_widget_label("Runs")

                # Get unique runs from current dataframe
                if 'run_id' in df.columns:
                    unique_runs = sorted([
                        run for run in df['run_id'].dropna().unique()
                        if run.strip() != ''
                    ])
                else:
                    unique_runs = []

                # Run multiselect
                saved_selected_runs = [
                    run for run in saved_settings.get('selected_runs', unique_runs)
                    if run in unique_runs
                ]
                if unique_runs:
                    selected_runs = st.multiselect(
                        "Runs",
                        options=unique_runs,
                        default=saved_selected_runs,
                        help="Select specific runs to include",
                        label_visibility="collapsed"
                    )
                else:
                    selected_runs = []
                    st.info("No runs available in the data")

            # MODELS SECTION
            with st.container(border=True):
                print_widget_label("Models")

                # Get unique detection models from current dataframe
                if 'detection_model_id' in df.columns:
                    unique_detection_models = sorted([
                        model for model in df['detection_model_id'].dropna().unique()
                        if model.strip() != '' and model != 'unknown'
                    ])
                else:
                    unique_detection_models = []

                # Detection model multiselect
                saved_selected_det_models = [
                    model for model in saved_settings.get('selected_detection_models', unique_detection_models)
                    if model in unique_detection_models
                ]
                if unique_detection_models:
                    selected_detection_models = st.multiselect(
                        "Detection models",
                        options=unique_detection_models,
                        default=saved_selected_det_models,
                        help="Select specific detection models to include"
                    )
                else:
                    selected_detection_models = []
                    st.info("No detection models available in the data")

                # Get unique classification models from current dataframe
                if 'classification_model_id' in df.columns:
                    unique_classification_models = sorted([
                        model for model in df['classification_model_id'].dropna().unique()
                        if model.strip() != '' and model != 'unknown'
                    ])
                else:
                    unique_classification_models = []

                # Classification model multiselect
                saved_selected_cls_models = [
                    model for model in saved_settings.get('selected_classification_models', unique_classification_models)
                    if model in unique_classification_models
                ]
                if unique_classification_models:
                    selected_classification_models = st.multiselect(
                        "Classification models",
                        options=unique_classification_models,
                        default=saved_selected_cls_models,
                        help="Select specific classification models to include"
                    )
                else:
                    selected_classification_models = []
                    st.info("No classification models available in the data")

            # Buttons
            col1, col2 = st.columns([1, 1])
            with col1:
                clear_all = st.form_submit_button("Clear all", width="stretch")
            with col2:
                apply_filter = st.form_submit_button("Apply", width="stretch", type="primary")
            
            if apply_filter:
                # Use current selections from saved settings
                # (Modal selections are saved when Apply is clicked in the modal)
                current_cls_selections = saved_settings.get('selected_classifications', [])

                # Save filter settings to config
                filter_settings = {
                    "aggrid_settings": {
                        "date_start": date_range[0].isoformat(),
                        "date_end": date_range[1].isoformat(),
                        "det_conf_min": det_conf_range[0],
                        "det_conf_max": det_conf_range[1],
                        "cls_conf_min": cls_conf_range[0],
                        "cls_conf_max": cls_conf_range[1],
                        "include_unclassified": include_unclassified,
                        "selected_detection_types": selected_detection_types,
                        "selected_classifications": current_cls_selections,
                        "selected_locations": selected_locations,
                        "selected_runs": selected_runs,
                        "selected_detection_models": selected_detection_models,
                        "selected_classification_models": selected_classification_models,
                        "image_size": saved_settings.get('image_size', 'medium')
                    }
                }
                update_vars("explore_results", filter_settings)
                
                # Clear caches
                if 'aggrid_thumbnail_cache' in st.session_state:
                    del st.session_state['aggrid_thumbnail_cache']
                if 'aggrid_last_cache_key' in st.session_state:
                    del st.session_state['aggrid_last_cache_key']
                if 'results_modified' in st.session_state:
                    del st.session_state['results_modified']
                
                # Reset to first page
                st.session_state.aggrid_current_page = 1
                
                st.rerun()
            
            if clear_all:
                # Get default classification list (all species)
                default_classifications = unique_classifications

                # Clear all filters to default ranges
                filter_settings = {
                    "aggrid_settings": {
                        "date_start": min_date.date().isoformat(),
                        "date_end": max_date.date().isoformat(),
                        "det_conf_min": DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
                        "det_conf_max": 1.0,
                        "cls_conf_min": 0.01,
                        "cls_conf_max": 1.0,
                        "include_unclassified": True,
                        "selected_detection_types": unique_detection_types,
                        "selected_classifications": default_classifications,
                        "selected_locations": unique_locations,
                        "selected_runs": unique_runs,
                        "selected_detection_models": unique_detection_models,
                        "selected_classification_models": unique_classification_models,
                        "image_size": saved_settings.get('image_size', 'medium')
                    }
                }
                update_vars("explore_results", filter_settings)
                
                # Clear caches
                if 'aggrid_thumbnail_cache' in st.session_state:
                    del st.session_state['aggrid_thumbnail_cache']
                if 'aggrid_last_cache_key' in st.session_state:
                    del st.session_state['aggrid_last_cache_key']
                if 'results_modified' in st.session_state:
                    del st.session_state['results_modified']
                
                # Reset to first page
                st.session_state.aggrid_current_page = 1
                
                st.rerun()

with export_col:
    # Export popover with material icon
    with st.popover(":material/download:", help="Export", width="stretch"):
        # Generate timestamp for filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Export formats container
        with st.container(border=True):
            print_widget_label(
                "Export formats",
                help_text="Exports the current selection, taking into account all applied filters and sorting"
            )

            # Four download buttons in one row
            download_col1, download_col2, download_col3, download_col4 = st.columns(4)
        
        with download_col1:
            # XLSX Download Button
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_to_export.to_excel(writer, index=False, sheet_name='Filtered_Data')
            xlsx_data = output.getvalue()

            st.download_button(
                label="XLSX",
                data=xlsx_data,
                file_name=f"addaxai-{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch"
            )

        with download_col2:
            # CSV Download Button
            csv_data = df_to_export.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="CSV",
                data=csv_data,
                file_name=f"addaxai-{timestamp}.csv",
                mime="text/csv",
                width="stretch"
            )

        with download_col3:
            # TSV Download Button
            tsv_data = df_to_export.to_csv(index=False, sep='\t').encode('utf-8')

            st.download_button(
                label="TSV",
                data=tsv_data,
                file_name=f"addaxai-{timestamp}.tsv",
                mime="text/tab-separated-values",
                width="stretch"
            )

        with download_col4:
            # JSON Download Button
            json_data = df_to_export.to_json(orient='records', indent=2).encode('utf-8')

            st.download_button(
                label="JSON",
                data=json_data,
                file_name=f"addaxai-{timestamp}.json",
                mime="application/json",
                width="stretch"
            )

with settings_col:
    # Settings popover with material icon
    with st.popover(":material/settings:", help="Settings", width="stretch"):
        # Settings container
        with st.container(border=True):
            print_widget_label("Image size")

            # Get the saved or default value for the segmented control
            default_image_size = saved_settings.get('image_size', DEFAULT_SIZE_OPTION)

            selected_size = st.segmented_control(
                "Image Size",
                options=list(image_size_options.keys()),
                default=default_image_size,
                key="aggrid_image_size_control",
                label_visibility="collapsed",
                width="stretch"
            )

            # Save the setting when changed
            if selected_size != default_image_size:
                filter_settings = {
                    "aggrid_settings": {
                        "date_start": saved_settings.get('date_start', ''),
                        "date_end": saved_settings.get('date_end', ''),
                        "det_conf_min": saved_settings.get('det_conf_min', 0.0),
                        "det_conf_max": saved_settings.get('det_conf_max', 1.0),
                        "cls_conf_min": saved_settings.get('cls_conf_min', 0.0),
                        "cls_conf_max": saved_settings.get('cls_conf_max', 1.0),
                        "include_unclassified": saved_settings.get('include_unclassified', True),
                        "selected_detection_types": saved_settings.get('selected_detection_types', []),
                        "selected_classifications": saved_settings.get('selected_classifications', []),
                        "selected_locations": saved_settings.get('selected_locations', []),
                        "selected_runs": saved_settings.get('selected_runs', []),
                        "selected_detection_models": saved_settings.get('selected_detection_models', []),
                        "selected_classification_models": saved_settings.get('selected_classification_models', []),
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

# Use filtered dataframe for all operations (already created earlier)
df_filtered = df_to_export

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

# Function now imported from utils.data_browser_utils

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
            '_df_index': idx,  # Store original dataframe index for modal navigation
            'image': image_url,  # Image column (was image_url in explore_results)
            'relative_path': row.get('relative_path') if pd.notna(row.get('relative_path')) else '',
            'detection_label': row.get('detection_label') if pd.notna(row.get('detection_label')) else '',
            'detection_confidence': round(float(row.get('detection_confidence')), 2) if pd.notna(row.get('detection_confidence')) else None,
            'classification_label': row.get('classification_label') if pd.notna(row.get('classification_label')) else '',
            'classification_confidence': round(float(row.get('classification_confidence')), 2) if pd.notna(row.get('classification_confidence')) else None,
            'timestamp': parse_timestamps(pd.Series([row.get('timestamp')])).iloc[0].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row.get('timestamp')) else '',
            'project_id': row.get('project_id') if pd.notna(row.get('project_id')) else '',
            'location_id': row.get('location_id') if pd.notna(row.get('location_id')) else '',
            'run_id': row.get('run_id') if pd.notna(row.get('run_id')) else '',
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

# Hide the internal index column
gb.configure_column("_df_index", hide=True)

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
gb.configure_column("run_id", headerName="Run ID", width=110, filter=False, sortable=False)
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
    'run_id',
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
gb.configure_grid_options(rowHeight=current_row_height + 10)

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
# AgGrid uses exact image height as row height with padding for breathing room
row_height = current_row_height + 10  # More padding for better spacing
header_height = 35  # Actual header height
buffer_height = 20  # Extra buffer to prevent internal scrollbar
grid_height = (actual_rows * row_height) + header_height + buffer_height  # Exact size based on rows with buffer

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
    
    # Get the original dataframe index we stored
    original_df_index = selected_row.get('_df_index')
    
    # Find the position in the full filtered dataframe
    if original_df_index is not None and 'results_modified' in st.session_state:
        df_filtered = st.session_state['results_modified']
        
        # Convert the original index to position in the filtered dataframe
        try:
            modal_index = df_filtered.index.get_loc(original_df_index)
        except KeyError:
            # Fallback: if index not found, use direct position
            modal_index = list(df_filtered.index).index(original_df_index)
        
        # Set modal state and open
        set_session_var("explore_results", "modal_current_image_index", modal_index)
        set_session_var("explore_results", "show_modal_image_viewer", True)
        st.rerun()

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
