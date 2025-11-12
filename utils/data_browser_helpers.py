"""
Reusable utilities for the Data Browser views.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import streamlit as st

from st_modal import Modal

from components import print_widget_label
from collections import Counter

from utils.common import update_vars, set_session_var, get_session_var


def render_sort_popover(
    sort_col,
    *,
    storage_key: str,
    current_settings: Dict[str, str],
    save_settings_fn,
    sortable_columns: List[tuple[str, str]],
    on_apply_fn=None,
):
    """Render a reusable sort popover for any data browser view."""

    columns_lookup = {col_id: label for col_id, label in sortable_columns}
    column_order = [col_id for col_id, _ in sortable_columns]
    saved_sort_column = current_settings.get("column", column_order[0])
    saved_sort_direction = current_settings.get("direction", "↓")

    with sort_col:
        with st.popover(":material/swap_vert:", help="Sort", width="stretch"):
            with st.form(f"sort_form_{storage_key}", border=False):
                with st.container(border=True):
                    print_widget_label("Sort by")
                    col_field, col_dir, col_btn = st.columns([3, 1, 1])

                    with col_field:
                        try:
                            default_index = column_order.index(saved_sort_column)
                        except ValueError:
                            default_index = 0

                        selected_column = st.selectbox(
                            "Column to sort",
                            options=column_order,
                            format_func=lambda x: columns_lookup.get(x, x),
                            index=default_index,
                            label_visibility="collapsed",
                        )

                    with col_dir:
                        selected_direction = st.segmented_control(
                            "Sorting method",
                            options=["↑", "↓"],
                            default=saved_sort_direction,
                            label_visibility="collapsed",
                            width="stretch",
                        )

                    with col_btn:
                        if st.form_submit_button("Apply", width="stretch", type="primary"):
                            save_settings_fn(
                                {
                                    "column": selected_column,
                                    "direction": selected_direction,
                                }
                            )
                            if on_apply_fn:
                                on_apply_fn()


def parse_timestamps(timestamp_series: pd.Series) -> pd.Series:
    """Parse timestamps in EXIF format: 2013:01:17 13:05:40."""
    return pd.to_datetime(timestamp_series, format="%Y:%m:%d %H:%M:%S")


def compute_filtered_detections(
    df: pd.DataFrame,
    filter_settings: dict,
    sort_column: str,
    sort_direction: str,
    detection_import_threshold: float,
) -> pd.DataFrame:
    """Apply shared filters/sorting to the detection-level dataframe."""
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])

    filtered_df = df.copy()

    date_start = filter_settings.get("date_start")
    date_end = filter_settings.get("date_end")
    if date_start and date_end and "timestamp" in filtered_df.columns:
        df_timestamps = parse_timestamps(filtered_df["timestamp"])
        start_dt = pd.to_datetime(date_start)
        end_dt = pd.to_datetime(date_end)
        filtered_df = filtered_df[
            (df_timestamps >= start_dt) & (df_timestamps <= end_dt)
        ].copy()

    det_conf_min = filter_settings.get("det_conf_min")
    det_conf_max = filter_settings.get("det_conf_max")
    if (
        det_conf_min is not None
        and det_conf_max is not None
        and "detection_confidence" in filtered_df.columns
    ):
        det_conf_min = max(float(det_conf_min), detection_import_threshold)
        filtered_df = filtered_df[
            (filtered_df["detection_confidence"] >= det_conf_min)
            & (filtered_df["detection_confidence"] <= float(det_conf_max))
        ].copy()

    cls_conf_min = filter_settings.get("cls_conf_min")
    cls_conf_max = filter_settings.get("cls_conf_max")
    include_unclass = filter_settings.get("include_unclassified", True)
    if (
        cls_conf_min is not None
        and cls_conf_max is not None
        and "classification_confidence" in filtered_df.columns
    ):
        cls_conf_min = float(cls_conf_min)
        cls_conf_max = float(cls_conf_max)
        if include_unclass:
            mask = (
                filtered_df["classification_confidence"].isna()
                | (
                    (filtered_df["classification_confidence"] >= cls_conf_min)
                    & (filtered_df["classification_confidence"] <= cls_conf_max)
                )
            )
        else:
            mask = (
                (filtered_df["classification_confidence"] >= cls_conf_min)
                & (filtered_df["classification_confidence"] <= cls_conf_max)
            )
        filtered_df = filtered_df[mask].copy()

    selected_det_types = filter_settings.get("selected_detection_types", [])
    if selected_det_types and "detection_label" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["detection_label"].isin(selected_det_types)
        ].copy()

    selected_classifications = filter_settings.get("selected_classifications", [])
    if selected_classifications and "classification_label" in filtered_df.columns:
        raw_series = filtered_df["classification_label"]
        classification_series = raw_series.fillna("").astype(str)
        classification_series_norm = classification_series.str.strip()
        selected_norm = {cls.strip().lower() for cls in selected_classifications}
        if include_unclass:
            mask = (
                raw_series.isna()
                | (classification_series_norm == "")
                | (classification_series_norm.str.lower() == "n/a")
                | (classification_series_norm.str.lower().isin(selected_norm))
            )
        else:
            mask = classification_series_norm.str.lower().isin(selected_norm)
        filtered_df = filtered_df[mask].copy()

    selected_locations = filter_settings.get("selected_locations", [])
    if selected_locations and "location_id" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["location_id"].isin(selected_locations)
        ].copy()

    selected_det_models = filter_settings.get("selected_detection_models", [])
    if selected_det_models and "detection_model_id" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["detection_model_id"].isin(selected_det_models)
        ].copy()

    selected_cls_models = filter_settings.get("selected_classification_models", [])
    if selected_cls_models and "classification_model_id" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["classification_model_id"].isin(selected_cls_models)
        ].copy()

    if sort_column in filtered_df.columns:
        ascending = sort_direction == "↑"
        try:
            if sort_column in ["detection_confidence", "classification_confidence"]:
                filtered_df = filtered_df.sort_values(
                    by=sort_column, ascending=ascending, na_position="last"
                )
            elif sort_column == "timestamp":
                df_timestamps = parse_timestamps(filtered_df["timestamp"])
                filtered_df = filtered_df.copy()
                filtered_df["_temp_timestamp"] = df_timestamps
                filtered_df = filtered_df.sort_values(
                    by="_temp_timestamp", ascending=ascending, na_position="last"
                ).drop(columns=["_temp_timestamp"])
            else:
                filtered_df = filtered_df.sort_values(
                    by=sort_column, ascending=ascending, na_position="last"
                )
        except Exception as exc:  # pragma: no cover - guard rail
            st.warning(f"Could not sort by {sort_column}: {exc}")
    elif sort_column != "timestamp" and "timestamp" in filtered_df.columns:
        try:
            df_timestamps = parse_timestamps(filtered_df["timestamp"])
            filtered_df = filtered_df.copy()
            filtered_df["_temp_timestamp"] = df_timestamps
            filtered_df = filtered_df.sort_values(
                by="_temp_timestamp", ascending=False, na_position="last"
            ).drop(columns=["_temp_timestamp"])
        except Exception:
            pass

    return filtered_df


def render_filter_popover(
    filter_col,
    df: pd.DataFrame,
    saved_settings: dict,
    detection_import_threshold: float,
):
    """Render the shared filter popover UI for all data browser views."""
    with filter_col:
        with st.popover(":material/tune:", help="Filter", width="stretch"):
            with st.form("data_browser_filter_form", border=False):
                if "detection_label" in df.columns:
                    unique_detection_types = sorted(
                        [
                            det.strip()
                            for det in df["detection_label"].dropna().astype(str).unique()
                            if det.strip()
                        ]
                    )
                else:
                    unique_detection_types = []

                with st.container(border=True):
                    print_widget_label("Detections")
                    saved_detection_types = saved_settings.get(
                        "selected_detection_types", unique_detection_types
                    )
                    if unique_detection_types:
                        selected_detection_types = st.multiselect(
                            "Classes",
                            options=unique_detection_types,
                            default=saved_detection_types,
                        )
                    else:
                        selected_detection_types = []
                        st.info("No detection types available in the data")

                    saved_det_conf_min = max(
                        saved_settings.get("det_conf_min", detection_import_threshold),
                        detection_import_threshold,
                    )
                    saved_det_conf_max = max(
                        saved_settings.get("det_conf_max", 1.0), saved_det_conf_min
                    )
                    det_conf_range = st.slider(
                        "Confidence range",
                        min_value=float(detection_import_threshold),
                        max_value=1.0,
                        value=(saved_det_conf_min, saved_det_conf_max),
                        step=0.01,
                        format="%.2f",
                        key="detection_confidence_slider",
                        help=(
                            "This table only shows detections loaded during startup. "
                            "Adjust Data import settings and reload the app to change the minimum threshold."
                        ),
                    )

                with st.container(border=True):
                    print_widget_label("Classifications")
                    st.markdown(
                        "<span style='color:#353640; font-size:0.85rem;'>Classes</span>",
                        unsafe_allow_html=True,
                    )

                    if "classification_label" in df.columns:
                        unique_classifications = sorted(
                            [
                                cls
                                for cls in df["classification_label"].dropna().unique()
                                if cls != "N/A" and cls.strip()
                            ]
                        )
                    else:
                        unique_classifications = []

                    saved_selected_classifications = saved_settings.get(
                        "selected_classifications", unique_classifications
                    )

                    col_btn, col_summary = st.columns([1, 3])
                    with col_btn:
                        if st.form_submit_button(
                            ":material/pets: Select",
                            use_container_width=True,
                            key="select_species_button",
                        ):
                            set_session_var("explore_results", "show_tree_modal", True)
                            st.rerun()

                    with col_summary:
                        total_count = len(unique_classifications)
                        selected_count = len(saved_selected_classifications)
                        if total_count > 0:
                            st.markdown(
                                f"""
                                <div style="background-color:#f0f2f6;padding:7px;border-radius:8px;">
                                    &nbsp;&nbsp;You have selected
                                    <code style='color:#086164'>{selected_count}</code>
                                    of
                                    <code style='color:#086164'>{total_count}</code> classes.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.info("No species classifications available in the data.")

                    saved_cls_conf_min = saved_settings.get("cls_conf_min", 0.01)
                    saved_cls_conf_max = saved_settings.get("cls_conf_max", 1.0)
                    cls_conf_range = st.slider(
                        "Confidence range",
                        min_value=0.0,
                        max_value=1.0,
                        value=(saved_cls_conf_min, saved_cls_conf_max),
                        step=0.01,
                        format="%.2f",
                        key="classification_confidence_slider",
                    )

                    include_unclassified = st.checkbox(
                        "Include unclassified observations",
                        value=saved_settings.get("include_unclassified", True),
                    )

                with st.container(border=True):
                    print_widget_label("Date range")
                    if "timestamp" in df.columns and not df["timestamp"].dropna().empty:
                        timestamps = parse_timestamps(df["timestamp"])
                        min_timestamp = timestamps.min()
                        max_timestamp = timestamps.max()
                    else:
                        min_timestamp = max_timestamp = None

                    if min_timestamp is None or pd.isna(min_timestamp):
                        min_date_dt = datetime.now() - timedelta(days=365)
                    else:
                        min_date_dt = min_timestamp

                    if max_timestamp is None or pd.isna(max_timestamp):
                        max_date_dt = datetime.now()
                    else:
                        max_date_dt = max_timestamp

                    if min_date_dt > max_date_dt:
                        min_date_dt = max_date_dt

                    min_date = min_date_dt.date()
                    max_date = max_date_dt.date()

                    saved_start_str = saved_settings.get("date_start", min_date.isoformat())
                    saved_end_str = saved_settings.get("date_end", max_date.isoformat())

                    try:
                        saved_start = datetime.fromisoformat(saved_start_str).date()
                    except ValueError:
                        saved_start = min_date
                    try:
                        saved_end = datetime.fromisoformat(saved_end_str).date()
                    except ValueError:
                        saved_end = max_date

                    if saved_start > saved_end:
                        saved_start = min_date
                        saved_end = max_date

                    date_range = st.date_input(
                        "Captured between",
                        value=(saved_start, saved_end),
                        format="YYYY/MM/DD",
                        label_visibility="collapsed",
                    )

                with st.container(border=True):
                    print_widget_label("Locations")
                    if "location_id" in df.columns:
                        unique_locations = sorted(
                            [
                                loc
                                for loc in df["location_id"].dropna().unique()
                                if loc.strip() and loc != "NONE"
                            ]
                        )
                    else:
                        unique_locations = []

                    saved_locations = [
                        loc
                        for loc in saved_settings.get(
                            "selected_locations", unique_locations
                        )
                        if loc in unique_locations
                    ]
                    if unique_locations:
                        selected_locations = st.multiselect(
                            "Locations",
                            options=unique_locations,
                            default=saved_locations or unique_locations,
                            label_visibility="collapsed",
                        )
                    else:
                        selected_locations = []
                        st.info("No locations available in the data.")

                apply_filters = st.form_submit_button(
                    "Apply filters", use_container_width=True, type="primary"
                )
                clear_filters = st.form_submit_button(
                    "Clear filters", use_container_width=True
                )

                if apply_filters:
                    new_settings = saved_settings.copy()
                    new_settings.update(
                        {
                            "selected_detection_types": selected_detection_types,
                            "det_conf_min": det_conf_range[0],
                            "det_conf_max": det_conf_range[1],
                            "selected_classifications": saved_selected_classifications,
                            "cls_conf_min": cls_conf_range[0],
                            "cls_conf_max": cls_conf_range[1],
                            "include_unclassified": include_unclassified,
                            "date_start": date_range[0].isoformat(),
                            "date_end": date_range[1].isoformat(),
                            "selected_locations": selected_locations,
                        }
                    )
                    update_vars("explore_results", {"aggrid_settings": new_settings})
                    for key in [
                        "results_modified",
                        "aggrid_thumbnail_cache",
                        "aggrid_last_cache_key",
                    ]:
                        st.session_state.pop(key, None)
                    st.session_state.aggrid_current_page = 1
                    st.rerun()

                if clear_filters:
                    min_iso = min_date.isoformat()
                    max_iso = max_date.isoformat()
                    default_settings = {
                        "selected_detection_types": unique_detection_types,
                        "det_conf_min": detection_import_threshold,
                        "det_conf_max": 1.0,
                        "selected_classifications": unique_classifications,
                        "cls_conf_min": 0.01,
                        "cls_conf_max": 1.0,
                        "include_unclassified": True,
                        "date_start": min_iso,
                        "date_end": max_iso,
                        "selected_locations": unique_locations,
                    }
                    update_vars("explore_results", {"aggrid_settings": default_settings})
                    for key in [
                        "results_modified",
                        "aggrid_thumbnail_cache",
                        "aggrid_last_cache_key",
                    ]:
                        st.session_state.pop(key, None)
                    st.session_state.aggrid_current_page = 1
                    st.rerun()


def handle_tree_selector_modal(df: pd.DataFrame, saved_settings: dict):
    """Open the taxonomy tree selector when requested."""
    if not get_session_var("explore_results", "show_tree_modal", False):
        return

    from components.taxonomic_tree_selector import tree_selector_modal

    if "classification_label" in df.columns:
        unique_classifications = sorted(
            [
                cls
                for cls in df["classification_label"].dropna().unique()
                if cls != "N/A" and cls.strip()
            ]
        )
    else:
        unique_classifications = []

    saved_selected_classifications = saved_settings.get(
        "selected_classifications", unique_classifications
    )

    tree_modal = Modal(
        title="Select Species",
        key="tree_selector_modal",
        show_close_button=False,
        show_title=False,
        show_divider=False,
    )

    with tree_modal.container():
        result = tree_selector_modal(
            available=unique_classifications,
            selected=saved_selected_classifications,
            key="explore_results_tree",
            title_mode="browser",
        )

        if result is not None:
            new_settings = saved_settings.copy()
            new_settings["selected_classifications"] = result
            update_vars("explore_results", {"aggrid_settings": new_settings})
            set_session_var("explore_results", "show_tree_modal", False)
            st.session_state.pop("explore_results_tree_dismissed", None)
            for key in ["aggrid_thumbnail_cache", "results_modified"]:
                st.session_state.pop(key, None)
            st.session_state.aggrid_current_page = 1
            st.rerun()
        elif st.session_state.pop("explore_results_tree_dismissed", None) == "cancel":
            set_session_var("explore_results", "show_tree_modal", False)

    st.stop()


def filter_files_by_detections(
    files_df: pd.DataFrame, detections_df: pd.DataFrame
) -> pd.DataFrame:
    """Filter aggregated files based on filtered detections."""
    if files_df is None:
        return None
    if files_df.empty:
        return files_df
    if detections_df is None or detections_df.empty:
        return files_df.iloc[0:0]

    key_cols = ["project_id", "location_id", "run_id", "relative_path"]
    if all(col in detections_df.columns for col in key_cols):
        detections_keys = (
            detections_df[key_cols].astype(str).agg("|".join, axis=1).dropna().unique()
        )
        files_df = files_df.copy()
        file_keys = files_df[key_cols].astype(str).agg("|".join, axis=1)
        return files_df[file_keys.isin(detections_keys)].copy()

    if "absolute_path" in detections_df.columns and "absolute_path" in files_df.columns:
        valid_paths = detections_df["absolute_path"].dropna().astype(str).unique()
        return files_df[files_df["absolute_path"].astype(str).isin(valid_paths)].copy()

    return files_df


def render_observation_export_popover(export_col, df_to_export: pd.DataFrame):
    """Render the custom export popover for the observation table."""
    if df_to_export is None or df_to_export.empty:
        return

    with export_col:
        with st.popover(":material/download:", help="Export", width="stretch"):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            with st.container(border=True):
                print_widget_label(
                    "Export formats",
                    help_text="Exports the current selection, taking into account all applied filters and sorting",
                )
                download_col1, download_col2, download_col3, download_col4 = st.columns(4)

            with download_col1:
                import io

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_to_export.to_excel(writer, index=False, sheet_name="Filtered_Data")
                st.download_button(
                    label="XLSX",
                    data=output.getvalue(),
                    file_name=f"addaxai-observations-{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )

            with download_col2:
                st.download_button(
                    label="CSV",
                    data=df_to_export.to_csv(index=False).encode("utf-8"),
                    file_name=f"addaxai-observations-{timestamp}.csv",
                    mime="text/csv",
                    width="stretch",
                )

            with download_col3:
                st.download_button(
                    label="TSV",
                    data=df_to_export.to_csv(index=False, sep="\t").encode("utf-8"),
                    file_name=f"addaxai-observations-{timestamp}.tsv",
                    mime="text/tab-separated-values",
                    width="stretch",
                )

            with download_col4:
                st.download_button(
                    label="JSON",
                    data=df_to_export.to_json(orient="records", indent=2).encode("utf-8"),
                    file_name=f"addaxai-observations-{timestamp}.json",
                    mime="application/json",
                    width="stretch",
                )


def render_export_popover(export_col, data_frame: pd.DataFrame, label_prefix: str):
    """Generic export popover used by the files/events views."""
    if data_frame is None or data_frame.empty:
        return

    with export_col:
        with st.popover(":material/download:", help="Export", width="stretch"):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            with st.container(border=True):
                print_widget_label(
                    "Export formats",
                    help_text=f"Exports the current selection for {label_prefix}.",
                )
                download_col1, download_col2, download_col3, download_col4 = st.columns(4)

            with download_col1:
                import io

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    data_frame.to_excel(writer, index=False, sheet_name="Data")
                st.download_button(
                    label="XLSX",
                    data=output.getvalue(),
                    file_name=f"addaxai-{label_prefix.lower()}-{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )

            with download_col2:
                st.download_button(
                    label="CSV",
                    data=data_frame.to_csv(index=False).encode("utf-8"),
                    file_name=f"addaxai-{label_prefix.lower()}-{timestamp}.csv",
                    mime="text/csv",
                    width="stretch",
                )

            with download_col3:
                st.download_button(
                    label="TSV",
                    data=data_frame.to_csv(index=False, sep="\t").encode("utf-8"),
                    file_name=f"addaxai-{label_prefix.lower()}-{timestamp}.tsv",
                    mime="text/tab-separated-values",
                    width="stretch",
                )

            with download_col4:
                st.download_button(
                    label="JSON",
                    data=data_frame.to_json(orient="records", indent=2).encode("utf-8"),
                    file_name=f"addaxai-{label_prefix.lower()}-{timestamp}.json",
                    mime="application/json",
                    width="stretch",
                )
