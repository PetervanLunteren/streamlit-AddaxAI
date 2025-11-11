"""
AddaxAI Data Browser coordinator page.
"""

from __future__ import annotations

import streamlit as st

import streamlit as st

from utils.common import (
    load_vars,
    load_app_settings,
    set_session_var,
    get_session_var,
    update_vars,
)
from utils.config import DEFAULT_DETECTION_CONFIDENCE_THRESHOLD
from utils.data_browser_helpers import (
    render_filter_popover,
    compute_filtered_detections,
    handle_tree_selector_modal,
    render_export_popover,
)
from components import print_widget_label

from pages import data_browser_observations as observations_view
from pages import data_browser_files as files_view

st.set_page_config(layout="wide")

BROWSER_VIEWS = [
    (":material/crop_free:", "Observations"),
    (":material/photo:", "Files"),
    (":material/event:", "Events"),
]
VIEW_LABELS = [label for _, label in BROWSER_VIEWS]
ICON_LOOKUP = {label: icon for icon, label in BROWSER_VIEWS}


def on_view_mode_change():
    selection = st.session_state.get("data_browser_level")
    if selection:
        set_session_var("explore_results", "browser_view_mode", selection)


def render_view_mode_control(current_view):
    columns = st.columns((2, 6, 1, 1, 1, 1, 1))
    (
        level_col,
        _spacer_one,
        _spacer_two,
        sort_col,
        filter_col,
        export_col,
        settings_col,
    ) = columns

    with level_col:
        st.segmented_control(
            "Level",
            options=VIEW_LABELS,
            format_func=lambda label: ICON_LOOKUP[label],
            default=current_view,
            key="data_browser_level",
            selection_mode="single",
            label_visibility="collapsed",
            width="stretch",
            on_change=on_view_mode_change,
        )

    return sort_col, filter_col, export_col, settings_col


def render_settings_popover(settings_col, saved_settings):
    row_height_options = {"small": 30, "medium": 100, "large": 250}
    image_size_options = {
        size: {"height": height, "width": int(height * 1.5)}
        for size, height in row_height_options.items()
    }

    with settings_col:
        with st.popover(":material/settings:", help="Settings", width="stretch"):
            with st.container(border=True):
                print_widget_label("Image size")
                default_image_size = saved_settings.get("image_size", "medium")
                selected_size = st.segmented_control(
                    "Image Size",
                    options=list(image_size_options.keys()),
                    default=default_image_size,
                    key="browser_image_size_control",
                    label_visibility="collapsed",
                    width="stretch",
                )

                if selected_size != default_image_size:
                    filter_settings = {"aggrid_settings": dict(saved_settings)}
                    filter_settings["aggrid_settings"]["image_size"] = selected_size
                    update_vars("explore_results", filter_settings)
                    for key in [
                        "aggrid_thumbnail_cache",
                        "aggrid_last_cache_key",
                        "file_thumbnail_cache",
                    ]:
                        st.session_state.pop(key, None)
                    st.rerun()


def render_data_browser_page():
    filter_config = load_vars("explore_results")
    saved_settings = filter_config.get("aggrid_settings", {})
    app_settings = load_app_settings()
    detection_import_threshold = float(
        app_settings.get("data_import", {}).get(
            "detection_conf_threshold", DEFAULT_DETECTION_CONFIDENCE_THRESHOLD
        )
    )
    detection_import_threshold = max(0.0, min(1.0, detection_import_threshold))

    saved_sort_column = saved_settings.get("sort_column", "timestamp")
    saved_sort_direction = saved_settings.get("sort_direction", "↓")

    if "results_observations" not in st.session_state:
        st.error("No detection results found. Please ensure detection data is loaded.")
        st.stop()

    df = st.session_state["results_observations"]
    if df.empty:
        st.warning("No detection results found.")
        st.stop()

    current_view = get_session_var("explore_results", "browser_view_mode", None)
    if current_view is None:
        current_view = VIEW_LABELS[0]
        set_session_var("explore_results", "browser_view_mode", current_view)

    sort_col, filter_col, export_col, settings_col = render_view_mode_control(current_view)
    current_view = get_session_var("explore_results", "browser_view_mode", VIEW_LABELS[0])

    render_settings_popover(settings_col, saved_settings)
    render_filter_popover(filter_col, df, saved_settings, detection_import_threshold)

    st.subheader(f"{ICON_LOOKUP[current_view]} {current_view}", divider="grey")

    filtered_detection_df = compute_filtered_detections(
        df=df,
        filter_settings=saved_settings,
        sort_column=saved_sort_column,
        sort_direction=saved_sort_direction,
        detection_import_threshold=detection_import_threshold,
    )

    handle_tree_selector_modal(df, saved_settings)

    if current_view == "Files":
        files_df = st.session_state.get("results_files")
        files_view.render_files_view(files_df, filtered_detection_df, export_col)
        return

    if current_view == "Events":
        st.info("Event-level browser coming soon.")
        render_export_popover(export_col, st.session_state.get("results_events"), "Events")
        return

    observations_view.render_observations_view(
        df,
        filtered_detection_df,
        sort_col,
        export_col,
        saved_settings,
        saved_sort_column,
        saved_sort_direction,
    )


render_data_browser_page()
