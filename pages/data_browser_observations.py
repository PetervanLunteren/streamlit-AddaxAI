"""
Observations view for the Data Browser.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode

from st_modal import Modal

from utils.data_browser_helpers import (
    render_observation_export_popover,
    parse_timestamps,
    render_sort_popover,
)
from utils.data_browser_utils import image_to_base64_url, image_viewer_modal
from utils.common import set_session_var, get_session_var, update_vars
from components import print_widget_label

# Local constants for the observation grid
ROW_HEIGHT_OPTIONS = {"small": 30, "medium": 100, "large": 250}
IMAGE_SIZE_RATIO = 1.5
IMAGE_COLUMN_WIDTHS = {
    size: int(height * IMAGE_SIZE_RATIO) for size, height in ROW_HEIGHT_OPTIONS.items()
}
IMAGE_SIZE_OPTIONS = {
    size: {"height": height, "width": IMAGE_COLUMN_WIDTHS[size]}
    for size, height in ROW_HEIGHT_OPTIONS.items()
}
DEFAULT_SIZE_OPTION = "medium"


def render_observations_view(
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    sort_col,
    export_col,
    saved_settings: dict,
    saved_sort_column: str,
    saved_sort_direction: str,
):
    """Render the detection-level (observations) table."""

    st.session_state["observations_results"] = filtered_df
    render_observation_export_popover(export_col, filtered_df)

    if get_session_var("explore_results", "show_modal_image_viewer", False):
        modal_source = get_session_var("explore_results", "modal_source", "observation")
        if modal_source != "file":
            modal_image_viewer = Modal(
                title="",
                key="image_viewer",
                show_close_button=False,
                show_title=False,
                show_divider=False,
            )
            with modal_image_viewer.container():
                image_viewer_modal()
            st.stop()

    observation_sortable_columns = [
        ("detection_label", "Detection"),
        ("detection_confidence", "Detection Confidence"),
        ("classification_label", "Classification"),
        ("classification_confidence", "Classification Confidence"),
        ("timestamp", "Timestamp"),
        ("location_id", "Location"),
    ]

    observation_sort_settings = {
        "column": saved_settings.get("sort_column", "timestamp"),
        "direction": saved_settings.get("sort_direction", "↓"),
    }
    saved_sort_column = observation_sort_settings["column"]
    saved_sort_direction = observation_sort_settings["direction"]

    def save_observation_sort(new_sort):
        updated_settings = saved_settings.copy()
        updated_settings.update(
            {
                "sort_column": new_sort["column"],
                "sort_direction": new_sort["direction"],
            }
        )
        update_vars("explore_results", {"aggrid_settings": updated_settings})

    def on_observation_sort_applied():
        for key in [
            "observations_results",
            "observations_thumbnail_cache",
            "observations_last_cache_key",
        ]:
            st.session_state.pop(key, None)
        st.session_state.observations_current_page = 1
        st.rerun()

    render_sort_popover(
        sort_col,
        storage_key="observations",
        current_settings=observation_sort_settings,
        save_settings_fn=save_observation_sort,
        sortable_columns=observation_sortable_columns,
        on_apply_fn=on_observation_sort_applied,
    )

    selected_size = st.session_state.get(
        "browser_image_size_control", DEFAULT_SIZE_OPTION
    )
    size_config = IMAGE_SIZE_OPTIONS.get(
        selected_size, IMAGE_SIZE_OPTIONS[DEFAULT_SIZE_OPTION]
    )
    current_row_height = size_config["height"]
    current_image_width = size_config["width"]

    if filtered_df.empty:
        st.warning("No results match the current filters.")
        st.stop()

    DEFAULT_PAGE_SIZE = 20
    PAGE_SIZE_OPTIONS = [20, 50, 100]
    if "observations_page_size" not in st.session_state:
        st.session_state.observations_page_size = DEFAULT_PAGE_SIZE
    if "observations_current_page" not in st.session_state:
        st.session_state.observations_current_page = 1

    total_rows = len(filtered_df)
    total_pages = max(
        1,
        (total_rows + st.session_state.observations_page_size - 1)
        // st.session_state.observations_page_size,
    )

    if st.session_state.observations_current_page > total_pages:
        st.session_state.observations_current_page = total_pages

    start_idx = (st.session_state.observations_current_page - 1) * st.session_state.observations_page_size
    end_idx = min(start_idx + st.session_state.observations_page_size, total_rows)

    cache_key = (
        f"page_{st.session_state.observations_current_page}_"
        f"pagesize_{st.session_state.observations_page_size}_imgsize_{selected_size}"
    )

    if "observations_thumbnail_cache" not in st.session_state:
        st.session_state.observations_thumbnail_cache = {}

    if (
        "observations_last_cache_key" not in st.session_state
        or st.session_state.observations_last_cache_key != cache_key
    ):
        st.session_state.observations_thumbnail_cache = {}
        st.session_state.observations_last_cache_key = cache_key

    df_page = filtered_df.iloc[start_idx:end_idx].copy()

    with st.spinner("Processing images for current page..."):
        if cache_key in st.session_state.observations_thumbnail_cache:
            image_urls = st.session_state.observations_thumbnail_cache[cache_key]
        else:
            image_urls = []
            for _, row in df_page.iterrows():
                bbox_data = {
                    "x": row.get("bbox_x"),
                    "y": row.get("bbox_y"),
                    "width": row.get("bbox_width"),
                    "height": row.get("bbox_height"),
                }
                thumbnail_size = (current_row_height, current_row_height)
                image_url = image_to_base64_url(
                    row.get("absolute_path"), bbox_data, max_size=thumbnail_size
                )
                image_urls.append(image_url)
            st.session_state.observations_thumbnail_cache[cache_key] = image_urls

        display_data = []
        for (idx, row), image_url in zip(df_page.iterrows(), image_urls):
            timestamp_value = ""
            if row.get("timestamp"):
                timestamp_value = (
                    parse_timestamps(pd.Series([row.get("timestamp")]))
                    .iloc[0]
                    .strftime("%Y-%m-%d %H:%M:%S")
                )

            display_data.append(
                {
                    "_df_index": idx,
                    "image": image_url,
                    "relative_path": row.get("relative_path") or "",
                    "detection_label": row.get("detection_label") or "",
                    "detection_confidence": round(
                        float(row.get("detection_confidence")), 2
                    )
                    if pd.notna(row.get("detection_confidence"))
                    else None,
                    "classification_label": row.get("classification_label") or "",
                    "classification_confidence": round(
                        float(row.get("classification_confidence")), 2
                    )
                    if pd.notna(row.get("classification_confidence"))
                    else None,
                    "timestamp": timestamp_value,
                    "project_id": row.get("project_id") or "",
                    "location_id": row.get("location_id") or "",
                }
            )

    display_df = pd.DataFrame(display_data)
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_column("_df_index", hide=True)

    image_jscode = JsCode(
        f"""
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
                    img.addEventListener('click', (e) => {{
                        e.stopPropagation();
                        params.node.setSelected(true);
                        window.lastImageClick = true;
                    }});
                }}
                this.eGui = document.createElement('div');
                this.eGui.style.display = 'flex';
                this.eGui.style.justifyContent = 'center';
                this.eGui.style.alignItems = 'center';
                this.eGui.appendChild(img);
            }}
            getGui() {{
                return this.eGui;
            }}
        }}
        """
    )

    gb.configure_column(
        "image",
        headerName="",
        cellRenderer=image_jscode,
        width=current_image_width + 20,
        autoHeight=True,
    )

    flex_columns = [
        ("relative_path", "File"),
        ("detection_label", "Detection"),
        ("classification_label", "Classification"),
        ("timestamp", "Timestamp"),
        ("location_id", "Location"),
    ]
    for col_id, header in flex_columns:
        gb.configure_column(
            col_id,
            headerName=header,
            flex=1,
            filter=False,
            sortable=False,
        )

    for col_id, header in [
        ("detection_confidence", "Detection confidence"),
        ("classification_confidence", "Classification confidence"),
    ]:
        gb.configure_column(
            col_id,
            headerName=header,
            flex=1,
            filter=False,
            sortable=False,
            valueFormatter="x.toFixed(2)",
            type="numericColumn",
            headerClass="ag-left-aligned-header",
        )

    gb.configure_column("project_id", headerName="Project ID", width=100, hide=True)

    gb.configure_default_column(
        resizable=True,
        sortable=False,
        filter=False,
        cellStyle={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "flex-start",
        },
        headerClass="ag-left-aligned-header",
    )
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_grid_options(
        rowHeight=current_row_height + 10,
        domLayout="normal",
        autoSizeStrategy={
            "type": "fitGridWidth",
            "defaultMinWidth": 100,
        },
    )
    grid_options = gb.build()

    row_height = current_row_height + 10
    header_height = 35
    buffer_height = 20
    grid_height = (len(display_df) * row_height) + header_height + buffer_height

    grid_response = AgGrid(
        display_df,
        gridOptions=grid_options,
        height=grid_height,
        allow_unsafe_jscode=True,
        theme="streamlit",
        update_on=["selectionChanged"],
    )

    if grid_response["selected_rows"] is not None and len(grid_response["selected_rows"]) > 0:
        selected_row = grid_response["selected_rows"].iloc[0]
        original_df_index = selected_row.get("_df_index")
        df_filtered = st.session_state.get("observations_results")
        if original_df_index is not None and df_filtered is not None:
            try:
                modal_index = df_filtered.index.get_loc(original_df_index)
            except KeyError:
                modal_index = list(df_filtered.index).index(original_df_index)

            set_session_var("explore_results", "modal_current_image_index", modal_index)
            set_session_var("explore_results", "modal_source", "observation")
            set_session_var("explore_results", "show_modal_image_viewer", True)
            st.rerun()

    bottom_menu = st.columns((4, 1, 1))
    with bottom_menu[0]:
        st.markdown(
            f"Page **{st.session_state.observations_current_page}** of **{total_pages}** "
            f"(Showing rows {start_idx + 1}-{end_idx} of {total_rows:,})"
        )

    with bottom_menu[1]:
        new_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            step=1,
            value=st.session_state.observations_current_page,
            key="observations_page_input",
        )
        if new_page != st.session_state.observations_current_page:
            st.session_state.observations_current_page = new_page
            st.rerun()

    with bottom_menu[2]:
        new_page_size = st.selectbox(
            "Page Size",
            options=PAGE_SIZE_OPTIONS,
            index=PAGE_SIZE_OPTIONS.index(st.session_state.observations_page_size),
            key="observations_page_size_input",
        )
        if new_page_size != st.session_state.observations_page_size:
            st.session_state.observations_page_size = new_page_size
            st.session_state.observations_current_page = 1
            st.rerun()
