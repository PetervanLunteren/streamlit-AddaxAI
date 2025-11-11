"""
Files view for the Data Browser.
"""

from __future__ import annotations

import hashlib
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode
import pandas as pd
from st_modal import Modal

from utils.data_browser_helpers import filter_files_by_detections, render_export_popover
from utils.data_browser_utils import image_to_base64_with_boxes, image_viewer_modal_file
from utils.common import set_session_var, get_session_var

DEFAULT_PAGE_SIZE = 20
PAGE_SIZE_OPTIONS = [20, 50, 100]


def render_files_view(
    files_df: pd.DataFrame,
    filtered_detections: pd.DataFrame,
    export_col,
):
    """Render the file-level browser with shared filters applied."""
    if files_df is None:
        st.error("File-level dataset was not initialized. Please restart the application.")
        st.stop()

    if get_session_var("explore_results", "show_modal_image_viewer", False) and get_session_var("explore_results", "modal_source", "observation") == "file":
        modal_image_viewer = Modal(
            title="",
            key="file_image_viewer",
            show_close_button=False,
            show_title=False,
            show_divider=False
        )
        with modal_image_viewer.container():
            image_viewer_modal_file()

    filtered_files_df = filter_files_by_detections(files_df, filtered_detections)
    render_file_level_browser(filtered_files_df)
    render_export_popover(export_col, filtered_files_df, "Files")


def render_file_level_browser(files_df: pd.DataFrame):
    """Render the file-level data browser view using aggregated detections."""
    if files_df is None:
        st.error("File-level dataset was not initialized. Please restart the application.")
        st.stop()

    if files_df.empty:
        st.warning("No files containing detections were found.")
        st.stop()

    files_df_reset = files_df.reset_index(drop=True)
    st.session_state["results_files_filtered"] = files_df_reset

    image_size_setting = st.session_state.get(
        "browser_image_size_control",
        st.session_state.get("aggrid_image_size_control", "medium"),
    )
    row_height_options = {"small": 30, "medium": 100, "large": 250}
    image_size_options = {
        size: {"height": height, "width": int(height * 1.5)}
        for size, height in row_height_options.items()
    }
    size_config = image_size_options.get(image_size_setting, image_size_options["medium"])
    file_row_height = size_config["height"]

    if "file_grid_page_size" not in st.session_state:
        st.session_state.file_grid_page_size = DEFAULT_PAGE_SIZE
    if "file_grid_current_page" not in st.session_state:
        st.session_state.file_grid_current_page = 1

    page_size = st.session_state.file_grid_page_size
    current_page = st.session_state.file_grid_current_page
    total_rows = len(files_df_reset)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    if current_page > total_pages:
        current_page = total_pages
        st.session_state.file_grid_current_page = current_page

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    df_page = files_df_reset.iloc[start_idx:end_idx].copy()

    page_signature_src = "|".join(
        df_page["relative_path"].fillna("").astype(str)
        + df_page["run_id"].fillna("").astype(str)
    )
    signature = hashlib.sha1(page_signature_src.encode("utf-8")).hexdigest()
    thumbnail_cache_key = (
        f"files_page_{current_page}_size_{page_size}_height_{file_row_height}_"
        f"setting_{image_size_setting}_sig_{signature}"
    )
    if "file_thumbnail_cache" not in st.session_state:
        st.session_state.file_thumbnail_cache = {}

    thumbnails = st.session_state.file_thumbnail_cache.get(thumbnail_cache_key)

    if thumbnails is None:
        thumbnails = []
        for _, row in df_page.iterrows():
            detection_details = row.get("detection_details") or []
            image_url = image_to_base64_with_boxes(
                row.get("absolute_path"),
                detection_details=detection_details,
                max_height=file_row_height,
            )
            thumbnails.append(image_url)
        st.session_state.file_thumbnail_cache.clear()
        st.session_state.file_thumbnail_cache[thumbnail_cache_key] = thumbnails

    display_rows = []
    for (idx, row), image_url in zip(df_page.iterrows(), thumbnails):
        raw_timestamp = row.get("timestamp")
        timestamp_value = pd.to_datetime(raw_timestamp) if raw_timestamp else None

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

        timestamp_display = (
            timestamp_value.strftime("%Y-%m-%d %H:%M:%S") if timestamp_value else ""
        )

        display_rows.append(
            {
                "_df_index": idx,
                "image": image_url,
                "relative_path": row.get("relative_path") or "",
                "detections": row.get("detections_summary") or "",
                "classifications": row.get("classifications_summary") or "",
                "count": detections_count,
                "timestamp": timestamp_display,
                "location_id": row.get("location_id") or "",
            }
        )


    display_df = pd.DataFrame(display_rows)

    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_column("_df_index", hide=True)

    gb.configure_column(
        "image",
        headerName="",
        width=file_row_height + 40,
        minWidth=file_row_height + 40,
        autoHeight=True,
        cellRenderer=JsCode(
            f"""
            class ImageRenderer {{
                init(params) {{
                    const img = document.createElement('img');
                    if (params.value) {{
                        img.src = params.value;
                        img.style.width = '{file_row_height}px';
                        img.style.height = '{file_row_height}px';
                        img.style.objectFit = 'contain';
                        img.style.border = 'none';
                        img.style.cursor = 'pointer';
                        img.addEventListener('click', (e) => {{
                            e.stopPropagation();
                            params.node.setSelected(true);
                        }});
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

    flex_columns = [
        ("relative_path", "File"),
        ("detections", "Detections"),
        ("classifications", "Classifications"),
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

    gb.configure_column(
        "count",
        headerName="Count",
        flex=1,
        type="numericColumn",
        headerClass="ag-left-aligned-header",
    )

    gb.configure_default_column(
        resizable=True,
        sortable=False,
        filter=False,
        cellStyle={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-start'},
        headerClass='ag-left-aligned-header'
    )
    gb.configure_selection(selection_mode='single', use_checkbox=True)
    gb.configure_grid_options(
        rowHeight=file_row_height + 10,
        domLayout="normal",
    )

    grid_options = gb.build()
    grid_height = (len(display_df) * (file_row_height + 10)) + 55

    grid_response = AgGrid(
        display_df,
        gridOptions=grid_options,
        height=grid_height,
        allow_unsafe_jscode=True,
        theme="streamlit",
        fit_columns_on_grid_load=True,
        update_on=['selectionChanged'],
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

    if (
        grid_response["selected_rows"] is not None
        and len(grid_response["selected_rows"]) > 0
    ):
        selected_row = grid_response["selected_rows"].iloc[0]
        original_index = selected_row.get("_df_index")

        df_filtered_files = st.session_state.get("results_files_filtered", pd.DataFrame())
        if not df_filtered_files.empty and original_index is not None:
            try:
                modal_index = df_filtered_files.index.get_loc(original_index)
            except KeyError:
                modal_index = int(original_index) if isinstance(original_index, (int, float)) else 0
        else:
            modal_index = 0

        set_session_var("explore_results", "modal_current_image_index", modal_index)
        set_session_var("explore_results", "modal_source", "file")
        set_session_var("explore_results", "show_modal_image_viewer", True)
        st.rerun()
