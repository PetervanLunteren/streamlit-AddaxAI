"""
Events view for the Data Browser.
"""

from __future__ import annotations

import base64
import math
from io import BytesIO
from typing import List

import pandas as pd
import streamlit as st
from PIL import Image
from st_aggrid import GridOptionsBuilder, AgGrid
from st_aggrid.shared import JsCode
from st_modal import Modal

from utils.common import set_session_var, get_session_var
from utils.data_browser_helpers import render_sort_popover, render_export_popover
from utils.data_browser_utils import build_event_collage_base64

PLACEHOLDER_IMAGE = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)


DEFAULT_PAGE_SIZE = 20
PAGE_SIZE_OPTIONS = [20, 50, 100]


def render_events_view(events_df: pd.DataFrame, export_col, sort_col):
    if events_df is None:
        st.error("Event dataset not initialized. Please restart the application.")
        st.stop()

    if events_df.empty:
        st.warning("No events available.")
        st.stop()

    sortable_columns = [
        ("detection_label", "Detection"),
        ("classification_label", "Classification"),
        ("start_timestamp", "Timestamp"),
        ("duration_seconds", "Duration"),
        ("image_count", "Images"),
        ("individual_count", "Individuals"),
        ("location_id", "Location"),
    ]

    sort_settings = {
        "column": st.session_state.get("events_sort_column", "start_timestamp"),
        "direction": st.session_state.get("events_sort_direction", "↓"),
    }

    def save_event_sort(new_sort):
        st.session_state["events_sort_column"] = new_sort["column"]
        st.session_state["events_sort_direction"] = new_sort["direction"]

    def on_event_sort_applied():
        for key in ["events_results", "events_thumbnail_cache"]:
            st.session_state.pop(key, None)
        st.session_state["events_current_page"] = 1
        st.rerun()

    render_sort_popover(
        sort_col,
        storage_key="events",
        current_settings=sort_settings,
        save_settings_fn=save_event_sort,
        sortable_columns=sortable_columns,
        on_apply_fn=on_event_sort_applied,
    )

    df_sorted = sort_events_dataframe(events_df, sort_settings["column"], sort_settings["direction"])
    st.session_state["events_results"] = df_sorted

    if get_session_var("explore_results", "show_modal_event_viewer", False):
        show_event_modal()
        st.stop()

    image_size_setting = st.session_state.get(
        "browser_image_size_control",
        "medium",
    )
    row_height_options = {"small": 30, "medium": 100, "large": 250}
    thumb_height_options = row_height_options
    thumb_width_options = {size: int(height * 4 / 3) for size, height in row_height_options.items()}

    current_row_height = row_height_options.get(image_size_setting, row_height_options["medium"])
    current_thumb_height = thumb_height_options.get(image_size_setting, thumb_height_options["medium"])
    current_thumb_width = thumb_width_options.get(image_size_setting, thumb_width_options["medium"])

    if "events_page_size" not in st.session_state:
        st.session_state["events_page_size"] = DEFAULT_PAGE_SIZE
    if "events_current_page" not in st.session_state:
        st.session_state["events_current_page"] = 1

    page_size = st.session_state["events_page_size"]
    current_page = st.session_state["events_current_page"]
    total_rows = len(df_sorted)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    if current_page > total_pages:
        current_page = total_pages
        st.session_state["events_current_page"] = current_page

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    df_page = df_sorted.iloc[start_idx:end_idx].reset_index()
    df_page = df_page.rename(columns={"index": "_df_index"})

    is_new_state = False
    if st.session_state.get("events_last_thumb_height") != current_thumb_height:
        is_new_state = True
        st.session_state["events_last_thumb_height"] = current_thumb_height
    if st.session_state.get("events_last_page") != current_page:
        is_new_state = True
        st.session_state["events_last_page"] = current_page

    with st.container():
        render_event_table(
            df_page,
            base_row_height=current_row_height,
            thumb_height=current_thumb_height,
            thumb_width=current_thumb_width,
            reset_cache=is_new_state,
        )
    render_export_popover(export_col, df_sorted, "Events")

    bottom_menu = st.columns((4, 1, 1))
    with bottom_menu[0]:
        st.markdown(
            f"Page **{current_page}** of **{total_pages}** "
            f"(Showing events {start_idx + 1}-{end_idx} of {total_rows:,})"
        )

    with bottom_menu[1]:
        new_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            step=1,
            value=current_page,
            key="events_page_input",
        )
        if new_page != current_page:
            st.session_state["events_current_page"] = new_page
            st.rerun()

    with bottom_menu[2]:
        new_page_size = st.selectbox(
            "Page Size",
            options=PAGE_SIZE_OPTIONS,
            index=PAGE_SIZE_OPTIONS.index(page_size),
            key="events_page_size_input",
        )
        if new_page_size != page_size:
            st.session_state["events_page_size"] = new_page_size
            st.session_state["events_current_page"] = 1
            st.rerun()


def sort_events_dataframe(df: pd.DataFrame, column: str, direction: str) -> pd.DataFrame:
    if df is None or df.empty or column not in df.columns:
        return df

    ascending = direction == "↑"
    if column in ("start_timestamp", "end_timestamp"):
        df = df.copy()
        df["_temp_ts"] = pd.to_datetime(df[column], errors="coerce")
        df = df.sort_values(by="_temp_ts", ascending=ascending, na_position="last")
        df = df.drop(columns=["_temp_ts"])
        return df

    return df.sort_values(by=column, ascending=ascending, na_position="last")


def render_event_table(
    df_page: pd.DataFrame,
    base_row_height: int,
    thumb_height: int,
    thumb_width: int,
    reset_cache: bool = False,
):
    if df_page.empty:
        st.warning("No events match the current filters.")
        st.stop()

    display_df = df_page.copy()
    if reset_cache:
        st.session_state.pop("events_collage_cache", None)

    display_df["collage_image"] = display_df.apply(
        lambda row: get_cached_event_collage(
            row.get("event_id"),
            row.get("event_files") or [],
            thumb_height,
            thumb_width,
            event_data={"event_id": row.get("event_id")},  # Now we only need event_id!
        ),
        axis=1,
    )

    preferred_order = [
        "collage_image",
        "individual_count",
        "detection_label",
        "classification_label",
        "duration_seconds",
        "image_count",
        "start_timestamp",
        "location_id",
        "end_timestamp",
        "detections_count",
        "classifications_count",
    ]
    ordered_cols = [col for col in preferred_order if col in display_df.columns]
    ordered_cols += [col for col in display_df.columns if col not in ordered_cols]
    display_df = display_df[ordered_cols]

    image_width = int(base_row_height * 1.5)

    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_column("_df_index", hide=True)
    gb.configure_column("file_paths", hide=True)
    gb.configure_column("event_files", hide=True)

    # Hide technical/detailed columns (user-requested)
    gb.configure_column("event_id", hide=True)
    gb.configure_column("project_id", hide=True)
    gb.configure_column("run_id", hide=True)
    gb.configure_column("detection_ids", hide=True)
    gb.configure_column("file_ids", hide=True)
    gb.configure_column("max_detection_conf", hide=True)
    gb.configure_column("max_classification_conf", hide=True)
    gb.configure_column("latitude", hide=True)
    gb.configure_column("longitude", hide=True)
    gb.configure_column("end_timestamp", hide=True)
    gb.configure_column("detections_count", hide=True)
    gb.configure_column("classifications_count", hide=True)

    collage_renderer = JsCode(
        """
        class EventCollageRenderer {
            init(params) {
                const img = document.createElement('img');
                if (params.value) {
                    img.src = params.value;
                    img.style.width = '""" + str(thumb_width) + """px';
                    img.style.height = '""" + str(thumb_height) + """px';
                    img.style.objectFit = 'cover';
                    img.style.borderRadius = '6px';
                    img.style.cursor = 'pointer';
                    img.addEventListener('click', (e) => {
                        e.stopPropagation();
                        params.node.setSelected(true);
                    });
                }
                this.eGui = document.createElement('div');
                this.eGui.style.display = 'flex';
                this.eGui.style.justifyContent = 'center';
                this.eGui.style.alignItems = 'center';
                this.eGui.appendChild(img);
            }
            getGui() {
                return this.eGui;
            }
        }
        """
    )

    gb.configure_column(
        "collage_image",
        headerName="",
        cellRenderer=collage_renderer,
        width=thumb_width + 20,
        minWidth=thumb_width + 20,
        suppressNavigable=True,
    )

    gb.configure_column("detection_label", headerName="Detection", flex=1)
    gb.configure_column("classification_label", headerName="Classification", flex=1)
    gb.configure_column("start_timestamp", headerName="Timestamp", flex=1)
    gb.configure_column("duration_seconds", headerName="Duration (s)", type="numericColumn", flex=1, headerClass="ag-left-aligned-header")
    gb.configure_column("image_count", headerName="Files", type="numericColumn", flex=1, headerClass="ag-left-aligned-header")
    gb.configure_column("detections_count", headerName="Detections", type="numericColumn", flex=1, headerClass="ag-left-aligned-header")
    gb.configure_column("individual_count", headerName="Individuals", type="numericColumn", flex=1, headerClass="ag-left-aligned-header")
    gb.configure_column("classifications_count", headerName="Classifications", type="numericColumn", flex=1, headerClass="ag-left-aligned-header")
    gb.configure_column("location_id", headerName="Location", flex=1)

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

    gb.configure_selection(selection_mode="single", use_checkbox=False)

    row_height = base_row_height + 10
    gb.configure_grid_options(domLayout="normal", rowHeight=row_height)

    header_height = 35
    buffer_height = 20
    grid_height = (len(display_df) * row_height) + header_height + buffer_height

    grid_response = AgGrid(
        display_df,
        gridOptions=gb.build(),
        height=grid_height,
        update_on=["selectionChanged"],
        theme="streamlit",
        allow_unsafe_jscode=True,
    )

    if grid_response["selected_rows"] is not None and len(grid_response["selected_rows"]) > 0:
        selected_row = grid_response["selected_rows"].iloc[0]
        df_filtered = st.session_state.get("events_results", pd.DataFrame())
        if not df_filtered.empty:
            original_index = selected_row.get("_df_index")
            if original_index is None:
                original_index = 0
            try:
                modal_index = df_filtered.index.get_loc(original_index)
            except KeyError:
                modal_index = int(original_index) if isinstance(original_index, (int, float)) else 0

            set_session_var("explore_results", "modal_current_event_index", modal_index)
            set_session_var("explore_results", "modal_source", "event")
            set_session_var("explore_results", "show_modal_event_viewer", True)
            st.rerun()


def show_event_modal():
    if not get_session_var("explore_results", "show_modal_event_viewer", False):
        return

    events_df = st.session_state.get("events_results", pd.DataFrame())
    if events_df.empty:
        return

    current_index = get_session_var("explore_results", "modal_current_event_index", 0)
    current_index = max(0, min(current_index, len(events_df) - 1))
    current_row = events_df.iloc[current_index]

    modal = Modal(
        title="",
        key="event_modal",
        show_close_button=False,
        show_title=False,
        show_divider=False,
    )

    with modal.container():
        event_files = current_row.get("event_files") or []
        show_bbox = get_session_var("explore_results", "modal_show_bbox", True)

        collage_b64 = get_cached_event_collage(
            current_row.get("event_id"),
            event_files,
            thumb_height=240,
            thumb_width=320,
            event_data={"event_id": current_row.get("event_id")},
            show_bbox=show_bbox,
        )

        img_col, meta_col = st.columns([1, 1])

        with img_col:
            with st.container(border=True):
                if collage_b64:
                    st.markdown(
                        f'<img src="{collage_b64}" style="width: 480px; height: auto;">',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("Could not load collage")
                st.write("")

            st.caption("Use shortcuts `←` `→` to navigate and `Esc` to close")

        with meta_col:
            top_col_export, top_col_close = st.columns([1, 1])

            with top_col_export:
                # Generate high-res collage on-demand and provide download
                with st.spinner("Generating high-resolution collage..."):
                    # Generate fresh collage at export resolution (1280×960)
                    high_res_collage = build_event_collage_base64(
                        event_files,
                        thumb_height=960,   # 4x modal display resolution
                        thumb_width=1280,   # Maintain 4:3 ratio
                        event_data={"event_id": current_row.get("event_id")},
                        show_bbox=show_bbox,
                    )

                    if high_res_collage and high_res_collage.startswith("data:image/"):
                        import base64
                        from datetime import datetime

                        base64_data = high_res_collage.split(",")[1]
                        image_bytes = base64.b64decode(base64_data)

                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        download_filename = f"addaxai-event-{timestamp}.png"

                        st.download_button(
                            label=":material/download: Export",
                            data=image_bytes,
                            file_name=download_filename,
                            mime="image/png",
                            type="secondary",
                            key="event_modal_export_button",
                            width="stretch",
                        )

            with top_col_close:
                if st.button(
                    ":material/close: Close",
                    type="secondary",
                    width="stretch",
                    key="event_modal_close_button",
                ):
                    set_session_var("explore_results", "modal_source", None)
                    set_session_var("explore_results", "show_modal_event_viewer", False)
                    st.rerun()

            nav_col_prev, nav_col_next = st.columns([1, 1])

            with nav_col_prev:
                if st.button(
                    ":material/chevron_left: Previous",
                    disabled=(current_index <= 0),
                    width="stretch",
                    type="secondary",
                    key="event_modal_prev_button",
                ):
                    set_session_var("explore_results", "modal_current_event_index", current_index - 1)
                    st.rerun()

            with nav_col_next:
                if st.button(
                    "Next :material/chevron_right:",
                    disabled=(current_index >= len(events_df) - 1),
                    width="stretch",
                    type="secondary",
                    key="event_modal_next_button",
                ):
                    set_session_var("explore_results", "modal_current_event_index", current_index + 1)
                    st.rerun()

            # Show/Hide bboxes toggle
            show_bbox = get_session_var("explore_results", "modal_show_bbox", True)
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
                key="event_bbox_toggle",
                label_visibility="collapsed",
                width="stretch",
            )

            new_show_bbox = selected_option == "show"
            if new_show_bbox != show_bbox:
                set_session_var("explore_results", "modal_show_bbox", new_show_bbox)
                st.rerun()

            with st.container(border=True):
                from components.ui_helpers import code_span

                # Format timestamp
                raw_timestamp = current_row.get("start_timestamp")
                if raw_timestamp:
                    try:
                        timestamp_display = pd.to_datetime(raw_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        timestamp_display = str(raw_timestamp)
                else:
                    timestamp_display = "N/A"

                # Format individual count
                individual_count = current_row.get('individual_count')
                individual_display = str(individual_count) if individual_count is not None else 'N/A'

                # Format duration
                duration = current_row.get('duration_seconds')
                duration_display = f"{duration}s" if duration is not None else 'N/A'

                metadata_rows = [
                    ("Individuals", individual_display),
                    ("Detection", current_row.get("detection_label", "N/A")),
                    ("Classification", current_row.get("classification_label", "N/A")),
                    ("Timestamp", timestamp_display),
                    ("Duration", duration_display),
                    ("Location", current_row.get("location_id", "N/A")),
                    ("Row", f"{current_index + 1} of {len(events_df)}"),
                ]

                for label, value in metadata_rows:
                    st.markdown(f"**{label}** {code_span(value)}", unsafe_allow_html=True)

        # Register keyboard shortcuts
        from components.shortcut_utils import register_shortcuts
        register_shortcuts(
            event_modal_prev_button=["arrowleft"],
            event_modal_next_button=["arrowright"],
            event_modal_close_button=["escape"],
        )


def get_cached_event_collage(event_id, event_files, thumb_height, thumb_width=None, event_data=None, show_bbox=True):
    cache = st.session_state.setdefault("events_collage_cache", {})
    identifier = event_id or "unknown"
    cache_key = f"{identifier}_{thumb_height}_{thumb_width or thumb_height}_bbox{show_bbox}"
    if cache_key not in cache:
        cache[cache_key] = (
            build_event_collage_base64(
                event_files,
                thumb_height=thumb_height,
                thumb_width=thumb_width,
                event_data=event_data,
                show_bbox=show_bbox,
            )
            or PLACEHOLDER_IMAGE
        )
    return cache[cache_key]
