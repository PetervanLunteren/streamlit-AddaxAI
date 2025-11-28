"""
AddaxAI Data Loading Utilities

Functions for loading and processing data from completed runs including:
- Detection results from addaxai-run.json files
- Project and run metadata aggregation
- DataFrame creation for analysis and visualization
"""

import os
import json
import math
from datetime import datetime
from collections import Counter
import pandas as pd
import streamlit as st
from utils.config import log
from utils.common import load_app_settings

def decode_gps_info(gps_info):
    """
    Decode EXIF GPSInfo dictionary to latitude/longitude decimal degrees.
    
    Args:
        gps_info (dict): GPSInfo dictionary from EXIF data
        
    Returns:
        tuple: (latitude, longitude) in decimal degrees, or (None, None) if invalid
    """
    try:
        # Get GPS coordinates from GPSInfo
        gps_lat = gps_info.get("GPSLatitude")
        gps_lat_ref = gps_info.get("GPSLatitudeRef") 
        gps_lon = gps_info.get("GPSLongitude")
        gps_lon_ref = gps_info.get("GPSLongitudeRef")
        
        # Check if we have actual GPS coordinate values (not just EXIF tag IDs)
        if (isinstance(gps_lat, (int, float)) and gps_lat <= 10) or \
           (isinstance(gps_lon, (int, float)) and gps_lon <= 10):
            # These look like EXIF tag IDs, not actual coordinates
            return None, None
            
        if not all([gps_lat, gps_lat_ref, gps_lon, gps_lon_ref]):
            return None, None
            
        # Convert GPS coordinates to decimal degrees
        # GPS coordinates are typically stored as [degrees, minutes, seconds]
        if isinstance(gps_lat, list) and len(gps_lat) == 3:
            lat = float(gps_lat[0]) + float(gps_lat[1])/60.0 + float(gps_lat[2])/3600.0
        else:
            lat = float(gps_lat)
            
        if isinstance(gps_lon, list) and len(gps_lon) == 3:
            lon = float(gps_lon[0]) + float(gps_lon[1])/60.0 + float(gps_lon[2])/3600.0
        else:
            lon = float(gps_lon)
            
        # Apply hemisphere references
        if gps_lat_ref in ['S', 'South', 2]:
            lat = -lat
        if gps_lon_ref in ['W', 'West', 4]:
            lon = -lon
            
        return lat, lon
        
    except (KeyError, ValueError, TypeError):
        return None, None


def load_model_taxonomy(model_id, classification_category_descriptions=None, classification_categories=None):
    """
    Load taxonomy information for a classification model.

    Args:
        model_id (str): Classification model ID
        classification_category_descriptions (dict, optional): Run-level taxonomy mapping (used for SpeciesNet models)
        classification_categories (dict, optional): Category labels from run JSON (used for SpeciesNet models)

    Returns:
        dict: {
            "tree": [...],           # Hierarchical tree structure
            "leaf_values": [...],    # All species (model_class values)
            "taxon_mapping": [...]   # Raw CSV data as list of dicts
        } or None if loading fails
    """
    from utils.analysis_utils import build_taxon_tree, get_all_leaf_values, load_taxon_mapping

    try:
        # Check if model is NONE or unknown
        if not model_id or model_id in ["NONE", "unknown"]:
            return None

        is_speciesnet = "SPECIESNET" in model_id.upper()

        if is_speciesnet:
            if not classification_category_descriptions:
                log(f"SpeciesNet model {model_id} is missing classification_category_descriptions; skipping taxonomy load")
                return None

            label_lookup = {str(k): v for k, v in (classification_categories or {}).items()}
            deduped_entries = {}

            for category_id, taxonomy_string in classification_category_descriptions.items():
                taxonomy_string = (taxonomy_string or "").strip()
                if not taxonomy_string:
                    continue

                parts = [p.strip() for p in taxonomy_string.split(';')]
                if len(parts) < 5:
                    parts.extend([""] * (5 - len(parts)))

                class_name, order_name, family_name, genus_name, species_name = parts[:5]
                label = (label_lookup.get(str(category_id)) or "").strip()
                fallback = next((name for name in [label, species_name, genus_name, family_name, order_name, class_name] if name), None)
                model_class = fallback or f"{model_id}_{category_id}"

                deduped_entries[model_class] = {
                    "model_class": model_class,
                    "class": class_name.lower(),
                    "order": order_name.lower(),
                    "family": family_name.lower(),
                    "genus": genus_name.lower(),
                    "species": species_name.lower()
                }

            taxon_mapping = list(deduped_entries.values())
            if not taxon_mapping:
                log(f"No embedded taxonomy entries found for SpeciesNet model {model_id}")
                return None

            log(f"Using embedded taxonomy for SpeciesNet model {model_id}: {len(taxon_mapping)} entries")

        else:
            try:
                taxon_mapping = load_taxon_mapping(model_id)
                log(f"Using taxonomy.csv for model {model_id}")
            except FileNotFoundError:
                log(f"taxonomy.csv not found for {model_id}")
                return None

        tree = build_taxon_tree(taxon_mapping)
        leaf_values = get_all_leaf_values(tree)

        return {
            "tree": tree,
            "leaf_values": leaf_values,
            "taxon_mapping": taxon_mapping
        }

    except Exception as e:
        log(f"Error loading taxonomy for model {model_id}: {str(e)}")
        return None


def merge_taxonomies(taxonomy_dict):
    """
    Merge multiple model taxonomies into a unified tree.

    Args:
        taxonomy_dict (dict): {
            "model_africa": {"tree": [...], "leaf_values": [...], "taxon_mapping": [...]},
            "model_asia": {...}
        }

    Returns:
        dict: {
            "merged_tree": [...],           # Unified tree structure
            "all_leaf_values": [...],       # All unique species
        }
    """
    from utils.analysis_utils import build_taxon_tree, get_all_leaf_values

    try:
        if not taxonomy_dict:
            return {
                "merged_tree": [],
                "all_leaf_values": []
            }

        # Collect all taxon_mapping entries from all models
        all_taxon_entries = []
        seen_model_classes = set()

        for model_id, taxonomy_data in taxonomy_dict.items():
            taxon_mapping = taxonomy_data.get("taxon_mapping", [])

            for entry in taxon_mapping:
                model_class = entry.get("model_class", "").strip()

                # Deduplicate by model_class
                if model_class and model_class not in seen_model_classes:
                    seen_model_classes.add(model_class)

                    normalized_entry = {
                        "model_class": model_class,
                        "class": (entry.get("class") or "").strip().lower(),
                        "order": (entry.get("order") or "").strip().lower(),
                        "family": (entry.get("family") or "").strip().lower(),
                        "genus": (entry.get("genus") or "").strip().lower(),
                        "species": (entry.get("species") or "").strip().lower(),
                    }

                    all_taxon_entries.append(normalized_entry)

        # Build unified tree from merged entries
        if all_taxon_entries:
            merged_tree = build_taxon_tree(all_taxon_entries)
            all_leaf_values = get_all_leaf_values(merged_tree)
        else:
            merged_tree = []
            all_leaf_values = []

        return {
            "merged_tree": merged_tree,
            "all_leaf_values": all_leaf_values
        }

    except Exception as e:
        log(f"Error merging taxonomies: {str(e)}")
        return {
            "merged_tree": [],
            "all_leaf_values": []
        }


def load_detection_results_dataframe():
    """
    Load all detection results from completed runs into a pandas DataFrame.
    
    Loops through all projects and runs in the MAP_JSON file, reads each 
    addaxai-run.json file, and flattens all detections into a single dataframe.
    
    Returns:
        pd.DataFrame: Detection results with columns:
            - project_id: Project identifier
            - location_id: Location identifier  
            - run_id: Run identifier
            - absolute_path: Full path to image file
            - relative_path: Image filename only
            - detection_label: Detection category (string)
            - detection_confidence: Detection confidence score (0-1)
            - classification_label: Species classification (string, may be None)
            - classification_confidence: Classification confidence (0-1, may be None)
            - bbox_x: Bounding box x coordinate (normalized 0-1)
            - bbox_y: Bounding box y coordinate (normalized 0-1) 
            - bbox_width: Bounding box width (normalized 0-1)
            - bbox_height: Bounding box height (normalized 0-1)
            - timestamp: Image capture datetime
            - latitude: GPS latitude (may be None)
            - longitude: GPS longitude (may be None)
            - image_width: Image width in pixels
            - image_height: Image height in pixels
            - detection_model_id: Model used for detection
            - classification_model_id: Model used for classification
    """
    
    results_list = []
    detection_keys = set()
    error_count = 0
    classification_models_seen = set()
    model_taxonomy_descriptions = {}
    model_category_labels = {}

    try:
        app_settings = load_app_settings()
        detection_threshold = float(
            app_settings.get("data_import", {}).get("detection_conf_threshold", 0.0)
        )
        detection_threshold = max(0.0, min(1.0, detection_threshold))
        # Load MAP_JSON from session state
        map_file_path = st.session_state["shared"]["MAP_FILE_PATH"]

        if not os.path.exists(map_file_path):
            log(f"MAP_JSON file not found: {map_file_path}")
            return pd.DataFrame()

        with open(map_file_path, 'r') as f:
            map_data = json.load(f)
            
        # Loop through all projects, locations, and runs
        for project_id, project_data in map_data.get("projects", {}).items():
            for location_id, location_data in project_data.get("locations", {}).items():
                for run_id, run_data in location_data.get("runs", {}).items():
                    
                    run_folder = run_data.get("folder")
                    if not run_folder:
                        continue
                        
                    run_json_path = os.path.join(run_folder, "addaxai-run.json")
                    
                    if not os.path.exists(run_json_path):
                        log(f"Run JSON not found: {run_json_path}")
                        error_count += 1
                        continue
                        
                    try:
                        # Load run JSON
                        with open(run_json_path, 'r') as f:
                            run_json = json.load(f)
                            
                        # Extract metadata
                        metadata = run_json.get("addaxai_metadata", {})
                        detection_model_id = metadata.get("selected_det_modelID", "unknown")
                        classification_model_id = metadata.get("selected_cls_modelID", "unknown")

                        # Track classification models encountered
                        if classification_model_id and classification_model_id not in ["NONE", "unknown"]:
                            classification_models_seen.add(classification_model_id)

                            if "SPECIESNET" in classification_model_id.upper():
                                category_descriptions = run_json.get("classification_category_descriptions", {})
                                if category_descriptions:
                                    existing_descriptions = model_taxonomy_descriptions.setdefault(classification_model_id, {})
                                    existing_descriptions.update(category_descriptions)

                                category_labels = run_json.get("classification_categories", {})
                                if category_labels:
                                    existing_labels = model_category_labels.setdefault(classification_model_id, {})
                                    existing_labels.update(category_labels)

                        # Get category mapping dictionaries
                        detection_categories = run_json.get("detection_categories", {})
                        classification_categories = run_json.get("classification_categories", {})
                        
                        # Get location coordinates - use location data for deployments, image EXIF for non-deployments
                        if location_id == "NONE":
                            # For non-deployments, GPS will be extracted from individual images
                            default_latitude = None
                            default_longitude = None
                        else:
                            # For deployments, use location coordinates from MAP_JSON
                            default_latitude = location_data.get("lat")
                            default_longitude = location_data.get("lon")
                        
                        # Process each image and its detections
                        for image_data in run_json.get("images", []):
                            image_filename = image_data.get("file")
                            image_datetime = image_data.get("datetime")
                            image_width = image_data.get("width")
                            image_height = image_data.get("height")

                            absolute_path = os.path.join(run_folder, image_filename) if image_filename else None

                            # Read file_id from JSON (required - no fallback)
                            file_id = image_data.get("file_id")
                            
                            # For non-deployments, try to get GPS from image EXIF data
                            if location_id == "NONE":
                                # Extract GPS from GPSInfo if available
                                gps_info = image_data.get("exif_metadata", {}).get("GPSInfo")
                                if gps_info:
                                    # Decode GPSInfo to lat/lng
                                    image_latitude, image_longitude = decode_gps_info(gps_info)
                                else:
                                    image_latitude = None
                                    image_longitude = None
                            else:
                                # For deployments, use location GPS
                                image_latitude = default_latitude
                                image_longitude = default_longitude
                            valid_detection_found = False
                            
                            # Process each detection in the image
                            for detection in image_data.get("detections", []):

                                # Extract detection data - no defaults, should always be present
                                detection_category_id = detection["category"]
                                detection_conf = detection["conf"]
                                if detection_conf < detection_threshold:
                                    continue
                                bbox = detection["bbox"]

                                # Read detection_id from JSON (required - no fallback)
                                detection_id = detection.get("detection_id")
                                
                                # Map detection category ID to actual label
                                detection_label = detection_categories[detection_category_id]
                                
                                # Extract classification data - may not be present if cls_modelID was "NONE"
                                if "classifications" in detection and detection["classifications"]:
                                    classifications = detection["classifications"]
                                    # Take the first classification (highest confidence)
                                    classification_id = classifications[0][0]
                                    classification_conf = float(classifications[0][1])
                                    # Map classification ID to actual label
                                    classification_label = classification_categories[classification_id]
                                else:
                                    # No classification performed
                                    classification_id = None
                                    classification_conf = None
                                    classification_label = None
                                
                                # Create row data
                                row = {
                                    'detection_id': detection_id,  # NEW: Unique detection ID
                                    'file_id': file_id,  # NEW: Unique file ID
                                    'project_id': project_id,
                                    'location_id': location_id,
                                    'run_id': run_id,
                                    'absolute_path': absolute_path,
                                    'relative_path': image_filename,
                                    'run_json_path': run_json_path,
                                    'detection_label': detection_label,
                                    'detection_confidence': float(detection_conf),
                                    'classification_label': classification_label,
                                    'classification_confidence': classification_conf,
                                    'bbox_x': float(bbox[0]) if len(bbox) >= 1 else 0.0,
                                    'bbox_y': float(bbox[1]) if len(bbox) >= 2 else 0.0,
                                    'bbox_width': float(bbox[2]) if len(bbox) >= 3 else 0.0,
                                    'bbox_height': float(bbox[3]) if len(bbox) >= 4 else 0.0,
                                    'timestamp': image_datetime,
                                    'latitude': image_latitude,
                                    'longitude': image_longitude,
                                    'image_width': image_width,
                                    'image_height': image_height,
                                    'detection_model_id': detection_model_id,
                                    'classification_model_id': classification_model_id,
                                    'is_empty_file': False,
                                }
                                
                                results_list.append(row)
                                detection_keys.add((project_id, location_id, run_id, image_filename))
                                valid_detection_found = True

                            if not valid_detection_found:
                                results_list.append({
                                    'detection_id': None,  # No detection for empty files
                                    'file_id': file_id,  # File still has an ID
                                    'project_id': project_id,
                                    'location_id': location_id,
                                    'run_id': run_id,
                                    'absolute_path': absolute_path,
                                    'relative_path': image_filename,
                                    'run_json_path': run_json_path,
                                    'detection_label': None,
                                    'detection_confidence': None,
                                    'classification_label': None,
                                    'classification_confidence': None,
                                    'bbox_x': None,
                                    'bbox_y': None,
                                    'bbox_width': None,
                                    'bbox_height': None,
                                    'timestamp': image_datetime,
                                    'latitude': image_latitude,
                                    'longitude': image_longitude,
                                    'image_width': image_width,
                                    'image_height': image_height,
                                    'detection_model_id': detection_model_id,
                                    'classification_model_id': classification_model_id
                                })
                                detection_keys.add((project_id, location_id, run_id, image_filename))
                                
                    except json.JSONDecodeError as e:
                        log(f"Invalid JSON in {run_json_path}: {str(e)}")
                        error_count += 1
                        continue
                    except Exception as e:
                        log(f"Error processing {run_json_path}: {str(e)}")
                        error_count += 1
                        continue
                        
        # Create DataFrame and ensure required schema even when empty
        df = pd.DataFrame(results_list)
        if df.empty:
            expected_columns = [
                "detection_id",
                "file_id",
                "project_id",
                "location_id",
                "run_id",
                "absolute_path",
                "relative_path",
                "run_json_path",
                "detection_label",
                "detection_confidence",
                "classification_label",
                "classification_confidence",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
                "timestamp",
                "latitude",
                "longitude",
                "image_width",
                "image_height",
                "detection_model_id",
                "classification_model_id",
            ]
            df = pd.DataFrame(columns=expected_columns)

        # Append empty files that had no detections
        empty_rows = []
        for project_id, project_data in map_data.get("projects", {}).items():
            for location_id, location_data in project_data.get("locations", {}).items():
                for run_id, run_data in location_data.get("runs", {}).items():
                    run_folder = run_data.get("folder")
                    if not run_folder:
                        continue
                    metadata = run_data.get("addaxai_metadata", {})
                    det_model_id = metadata.get("selected_det_modelID", "unknown")
                    cls_model_id = metadata.get("selected_cls_modelID", "unknown")
                    for image_data in run_data.get("images", []):
                        image_filename = image_data.get("file")
                        absolute_path = os.path.join(run_folder, image_filename) if image_filename else None
                        relative_path = image_filename
                        key = (project_id, location_id, run_id, relative_path)
                        if key in detection_keys:
                            continue
                        if not image_data.get("detections") or key not in detection_keys:
                            empty_rows.append({
                                "project_id": project_id,
                                "location_id": location_id,
                                "run_id": run_id,
                                "absolute_path": absolute_path,
                                "relative_path": relative_path,
                                "run_json_path": run_json_path,
                                "detection_label": None,
                                "detection_confidence": None,
                                "classification_label": None,
                                "classification_confidence": None,
                                "bbox_x": None,
                                "bbox_y": None,
                                "bbox_width": None,
                                "bbox_height": None,
                                "timestamp": image_data.get("datetime"),
                                "latitude": None,
                                "longitude": None,
                                "image_width": image_data.get("width"),
                                "image_height": image_data.get("height"),
                                "detection_model_id": det_model_id,
                                "classification_model_id": cls_model_id,
                                "is_empty_file": True,
                            })

        if empty_rows:
            empty_df = pd.DataFrame(empty_rows)
            df = pd.concat([df, empty_df], ignore_index=True)

        # Load taxonomy for each classification model encountered
        taxonomy_dict = {}
        if classification_models_seen:
            log(f"Loading taxonomy for {len(classification_models_seen)} classification model(s)...")

            for model_id in classification_models_seen:
                taxonomy_data = load_model_taxonomy(
                    model_id,
                    classification_category_descriptions=model_taxonomy_descriptions.get(model_id),
                    classification_categories=model_category_labels.get(model_id)
                )
                if taxonomy_data:
                    taxonomy_dict[model_id] = taxonomy_data
                    log(f"  ✓ Loaded taxonomy for {model_id}: {len(taxonomy_data['leaf_values'])} species")
                else:
                    log(f"  ✗ Failed to load taxonomy for {model_id}")

        # Merge all taxonomies into unified tree
        if taxonomy_dict:
            merged_taxonomy = merge_taxonomies(taxonomy_dict)
            st.session_state["taxonomy_cache"] = merged_taxonomy
            log(f"Merged taxonomy: {len(merged_taxonomy['all_leaf_values'])} unique species total")
        else:
            st.session_state["taxonomy_cache"] = None
            if classification_models_seen:
                log("No taxonomies loaded - tree selector will not be available")

        # Log summary
        if len(df) > 0:
            log(f"Successfully loaded {len(df)} detections from {len(df['run_id'].unique())} runs")
        else:
            log("No detection results found in any runs")

        if error_count > 0:
            log(f"Encountered {error_count} errors while loading detection results")

        return df
        
    except Exception as e:
        log(f"Fatal error in load_detection_results_dataframe: {str(e)}")
        return pd.DataFrame()


FILE_AGGREGATION_COLUMNS = [
    # Identity
    "file_id",
    # Hierarchy
    "project_id",
    "location_id",
    "run_id",
    # Links
    "event_ids",
    "detection_ids",
    # Content
    "absolute_path",
    "detections_count",
    "detections_summary",
    "classifications_count",
    "classifications_summary",
    # Temporal
    "timestamp",
    # Spatial
    "latitude",
    "longitude",
    # Technical
    "image_width",
    "image_height",
    # Kept for internal use but will be removed from exports
    "relative_path",
    "run_json_path",
    "detection_details",
]


def aggregate_detections_to_files(detections_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate detection-level results into a file-level DataFrame.

    Each row represents a single media file with summary metrics and the raw
    detection details preserved for drill-down views.

    Args:
        detections_df (pd.DataFrame): Detection-level dataframe generated by
            load_detection_results_dataframe.

    Returns:
        pd.DataFrame: File-level dataframe with helper columns and detection details.

    Raises:
        ValueError: If required detection columns are missing.
    """
    required_columns = {
        "project_id",
        "location_id",
        "run_id",
        "absolute_path",
        "relative_path",
        "timestamp",
        "detection_label",
        "detection_confidence",
        "classification_label",
        "classification_confidence",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "image_width",
        "image_height",
        "latitude",
        "longitude",
    }

    if detections_df is None:
        raise ValueError("Detections dataframe is None.")

    missing_columns = required_columns - set(detections_df.columns)
    if missing_columns:
        raise ValueError(
            f"Detections dataframe is missing required columns: {sorted(missing_columns)}"
        )

    if detections_df.empty:
        return pd.DataFrame(columns=FILE_AGGREGATION_COLUMNS)

    group_columns = [
        "project_id",
        "location_id",
        "run_id",
        "absolute_path",
        "relative_path",
    ]

    aggregated_files = []

    # Group by file identity to consolidate detections per media asset
    for group_keys, group in detections_df.groupby(group_columns, dropna=False):
        (
            project_id,
            location_id,
            run_id,
            absolute_path,
            relative_path,
        ) = group_keys

        real_rows = group[group["detection_label"].notna()]
        detections_count = int(len(real_rows))

        detection_labels = [
            str(label).strip()
            for label in real_rows["detection_label"]
            if str(label).strip()
        ]
        if detection_labels:
            detection_counter = Counter(detection_labels)
            detections_summary = ", ".join(
                f"{label} ({count})" for label, count in detection_counter.items()
            )
        else:
            detections_summary = "None"

        classification_labels = []
        for value in real_rows["classification_label"]:
            label = str(value).strip()
            if not label or label.upper() == "N/A":
                continue
            classification_labels.append(label)

        if classification_labels:
            classification_counter = Counter(classification_labels)
            classifications_summary = ", ".join(
                f"{label} ({count})" for label, count in classification_counter.items()
            )
            classifications_count = sum(classification_counter.values())
        else:
            classifications_summary = "None"
            classifications_count = 0

        detection_details = []
        for _, detection_row in real_rows.iterrows():
            detection_details.append(
                {
                    "detection_label": detection_row.get("detection_label"),
                    "detection_confidence": (
                        float(detection_row["detection_confidence"])
                        if pd.notna(detection_row["detection_confidence"])
                        else None
                    ),
                    "classification_label": detection_row.get("classification_label"),
                    "classification_confidence": (
                        float(detection_row["classification_confidence"])
                        if pd.notna(detection_row["classification_confidence"])
                        else None
                    ),
                    "bbox": {
                        "x": float(detection_row["bbox_x"])
                        if pd.notna(detection_row["bbox_x"])
                        else None,
                        "y": float(detection_row["bbox_y"])
                        if pd.notna(detection_row["bbox_y"])
                        else None,
                        "width": float(detection_row["bbox_width"])
                        if pd.notna(detection_row["bbox_width"])
                        else None,
                        "height": float(detection_row["bbox_height"])
                        if pd.notna(detection_row["bbox_height"])
                        else None,
                    },
                }
            )

        timestamp_series = pd.to_datetime(
            group["timestamp"],
            format="%Y:%m:%d %H:%M:%S",
            errors="coerce",
        )
        timestamp_value = None
        if not timestamp_series.empty:
            timestamp_value = timestamp_series.min()
            if pd.notna(timestamp_value):
                timestamp_value = timestamp_value.isoformat()

        # Resolve single-value columns with sensible fallbacks
        def first_non_null(series):
            series_no_na = series.dropna()
            return series_no_na.iloc[0] if not series_no_na.empty else None

        image_width = first_non_null(group["image_width"])
        image_height = first_non_null(group["image_height"])
        latitude = first_non_null(group["latitude"])
        longitude = first_non_null(group["longitude"])
        run_json_path = first_non_null(group["run_json_path"])
        file_id = first_non_null(group["file_id"])

        # Collect unique event_ids for this file (if event_id column exists)
        event_ids = None
        if "event_id" in group.columns:
            unique_event_ids = group["event_id"].dropna().unique()
            if len(unique_event_ids) > 0:
                event_ids = ",".join(sorted(str(eid) for eid in unique_event_ids))

        # Collect unique detection_ids for this file (if detection_id column exists)
        detection_ids = None
        if "detection_id" in group.columns:
            unique_detection_ids = group["detection_id"].dropna().unique()
            if len(unique_detection_ids) > 0:
                detection_ids = ",".join(sorted(str(did) for did in unique_detection_ids))

        aggregated_files.append(
            {
                # Identity
                "file_id": file_id,
                # Hierarchy
                "project_id": project_id,
                "location_id": location_id,
                "run_id": run_id,
                # Links
                "event_ids": event_ids,
                "detection_ids": detection_ids,
                # Content
                "absolute_path": absolute_path,
                "detections_count": detections_count,
                "detections_summary": detections_summary,
                "classifications_count": classifications_count,
                "classifications_summary": classifications_summary,
                # Temporal
                "timestamp": timestamp_value,
                # Spatial
                "latitude": latitude,
                "longitude": longitude,
                # Technical
                "image_width": image_width,
                "image_height": image_height,
                # Internal
                "relative_path": relative_path,
                "run_json_path": run_json_path,
                "detection_details": detection_details,
            }
        )

    return pd.DataFrame(aggregated_files, columns=FILE_AGGREGATION_COLUMNS)


def aggregate_detections_to_events(
    detections_df: pd.DataFrame,
    time_gap_seconds: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate detection-level records into event-level summaries based on a time gap threshold.

    Each event represents a burst of detections captured within `time_gap_seconds` of each other
    for the same project/location/run combination.

    Also assigns event_id to each detection in the input detections_df.

    Returns:
        tuple: (events_df, detections_df_with_event_ids)
            - events_df: Event-level aggregations
            - detections_df_with_event_ids: Original detections with event_id column added
    """
    if detections_df is None or detections_df.empty:
        empty_events = pd.DataFrame(
            columns=[
                # Identity
                "event_id",
                # Hierarchy
                "project_id",
                "location_id",
                "run_id",
                # Links
                "detection_ids",
                "file_ids",
                # Content
                "detection_label",
                "classification_label",
                "detections_count",
                "classifications_count",
                "image_count",
                "max_detection_conf",
                "max_classification_conf",
                # Temporal
                "start_timestamp",
                "end_timestamp",
                "duration_seconds",
                # Spatial
                "latitude",
                "longitude",
                # Internal
                "file_paths",
                "event_files",
            ]
        )
        empty_detections = detections_df.copy() if detections_df is not None else pd.DataFrame()
        if not empty_detections.empty:
            empty_detections["event_id"] = None
        return empty_events, empty_detections

    # Keep original df to preserve all rows (including empty files)
    original_detections_df = detections_df.copy()
    original_detections_df["event_id"] = None

    # Filter out empty files and rows with no valid labels for event aggregation only
    detections_df = detections_df.copy()

    # Remove is_empty_file rows
    if "is_empty_file" in detections_df.columns:
        detections_df = detections_df[detections_df["is_empty_file"] != True].copy()

    # Also remove rows where BOTH detection_label AND classification_label are null/empty
    # (these can't create meaningful events)
    if "detection_label" in detections_df.columns and "classification_label" in detections_df.columns:
        has_detection = detections_df["detection_label"].notna() & (detections_df["detection_label"] != "")
        has_classification = detections_df["classification_label"].notna() & (detections_df["classification_label"] != "")
        detections_df = detections_df[has_detection | has_classification].copy()

    if detections_df.empty:
        # No actual detections to create events from
        return pd.DataFrame(columns=list(empty_events.columns)), original_detections_df

    detections_df["_timestamp_dt"] = detections_df["timestamp"].apply(_parse_detection_timestamp)
    detections_df = detections_df.sort_values(by=["project_id", "location_id", "run_id", "_timestamp_dt"])

    # Initialize event_id column (will be filled as we create events)
    detections_df["event_id"] = None

    grouped = detections_df.groupby(["project_id", "location_id", "run_id"], dropna=False, sort=False)

    events = []

    for (project_id, location_id, run_id), group in grouped:
        # Keep original index so we can map event_ids back to original dataframe
        group = group.sort_values("_timestamp_dt")

        # Step 1: Identify time windows based on time gaps
        time_windows = []
        current_window_indices = []
        last_timestamp = None

        for idx, row in group.iterrows():
            timestamp_dt = row.get("_timestamp_dt")

            if last_timestamp is None:
                # First detection starts first window
                current_window_indices.append(idx)
                last_timestamp = timestamp_dt
            else:
                gap_seconds = None
                if timestamp_dt is not None and last_timestamp is not None:
                    gap_seconds = (timestamp_dt - last_timestamp).total_seconds()

                if gap_seconds is not None and gap_seconds > time_gap_seconds:
                    # Time gap exceeded - close current window and start new one
                    time_windows.append(current_window_indices)
                    current_window_indices = [idx]
                else:
                    # Continue current window
                    current_window_indices.append(idx)

                last_timestamp = timestamp_dt

        # Add final window
        if current_window_indices:
            time_windows.append(current_window_indices)

        # Step 2: Within each time window, group by label and create events
        for window_indices in time_windows:
            window_rows = group.loc[window_indices]

            # Group detections in this window by classification (if available) or detection label
            label_groups = {}
            for idx, row in window_rows.iterrows():
                classification_label = _normalize_species_label(row.get("classification_label"))
                detection_label = _normalize_species_label(row.get("detection_label"))
                # Use classification for grouping if available, otherwise use detection label
                grouping_label = classification_label if classification_label != "unclassified" else detection_label

                if grouping_label not in label_groups:
                    label_groups[grouping_label] = []
                label_groups[grouping_label].append((idx, row))

            # Create one event per unique label in this window
            for grouping_label, label_detections in label_groups.items():
                indices = [idx for idx, _ in label_detections]
                rows = [row for _, row in label_detections]

                # Create event for this label
                event_record = _finalize_event(rows, project_id, location_id, run_id)
                event_id = event_record["event_id"]

                # Tag all detections of this label with the event_id
                for det_idx in indices:
                    detections_df.at[det_idx, "event_id"] = event_id

                events.append(event_record)

    events_df = pd.DataFrame(events)

    # Merge event_ids back into original dataframe (preserving empty file rows)
    # Update original_detections_df with event_ids from detections_df
    if not detections_df.empty and "event_id" in detections_df.columns:
        # Use index to map event_ids back to original dataframe
        event_id_map = detections_df["event_id"].to_dict()
        for idx, event_id in event_id_map.items():
            if idx in original_detections_df.index:
                original_detections_df.at[idx, "event_id"] = event_id

    # Return both events and detections (now with event_id column)
    return events_df, original_detections_df


def _finalize_event(rows, project_id, location_id, run_id):
    """Helper to summarize a list of detection rows into a single event record."""
    df_event = pd.DataFrame(rows)
    timestamps = pd.to_datetime(df_event.get("_timestamp_dt"), errors="coerce").dropna()
    start_ts = timestamps.min() if not timestamps.empty else None
    end_ts = timestamps.max() if not timestamps.empty else start_ts
    duration_seconds = (
        (end_ts - start_ts).total_seconds() if start_ts is not None and end_ts is not None else 0
    )

    # Get the single detection_label (should all be same since we grouped by it)
    detection_series = df_event["detection_label"].dropna().astype(str).str.strip()
    detection_label = detection_series.mode().iloc[0] if not detection_series.empty else None

    # Get the single classification_label (should all be same, or null if detection-only)
    classification_series = df_event["classification_label"].dropna().astype(str).str.strip()
    classification_label = classification_series.mode().iloc[0] if not classification_series.empty else None

    relative_series = df_event.get("relative_path")
    absolute_series = df_event.get("absolute_path")
    relative_paths = relative_series.dropna().astype(str).tolist() if relative_series is not None else []

    path_map = {}
    if relative_series is not None and absolute_series is not None:
        for rel, abs_path in zip(relative_series.tolist(), absolute_series.tolist()):
            rel = str(rel) if rel is not None else None
            if rel and rel not in path_map:
                path_map[rel] = abs_path
    unique_relative_paths = []
    seen = set()
    for rel in relative_paths:
        if rel and rel not in seen:
            seen.add(rel)
            unique_relative_paths.append(rel)
    event_files = [
        {
            "relative_path": rel,
            "absolute_path": path_map.get(rel),
        }
        for rel in unique_relative_paths
    ]

    # Generate deterministic UUID based on event characteristics
    # This ensures the same event gets the same UUID across app restarts
    import uuid
    import hashlib

    # Create a unique identifier string from event characteristics
    # Use classification if available, otherwise detection
    event_species = classification_label if classification_label else detection_label
    event_identifier = f"{run_id}_{start_ts.isoformat() if start_ts is not None else 'none'}_{event_species}"

    # Generate UUID5 (deterministic) from the identifier
    event_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, event_identifier))

    # Collect detection_ids for this event (if detection_id column exists)
    detection_ids = None
    if "detection_id" in df_event.columns:
        unique_detection_ids = df_event["detection_id"].dropna().unique()
        if len(unique_detection_ids) > 0:
            detection_ids = ",".join(sorted(str(did) for did in unique_detection_ids))

    # Collect file_ids for this event (if file_id column exists)
    file_ids = None
    if "file_id" in df_event.columns:
        unique_file_ids = df_event["file_id"].dropna().unique()
        if len(unique_file_ids) > 0:
            file_ids = ",".join(sorted(str(fid) for fid in unique_file_ids))

    # Calculate individual_count using max-per-image method
    # Count detections per file, then take the maximum
    individual_count = None
    if "file_id" in df_event.columns and not df_event.empty:
        # Group by file_id and count detections per file
        detections_per_file = df_event.groupby("file_id").size()
        if not detections_per_file.empty:
            individual_count = int(detections_per_file.max())

    return {
        # Identity
        "event_id": event_uuid,
        # Hierarchy
        "project_id": project_id,
        "location_id": location_id,
        "run_id": run_id,
        # Links
        "detection_ids": detection_ids,
        "file_ids": file_ids,
        # Content
        "detection_label": detection_label,
        "classification_label": classification_label,
        "detections_count": len(df_event),
        "individual_count": individual_count,
        "classifications_count": df_event["classification_label"].notna().sum(),
        "image_count": len(unique_relative_paths),  # Fixed: count unique files, not detection rows
        "max_detection_conf": df_event["detection_confidence"].max(skipna=True),
        "max_classification_conf": df_event["classification_confidence"].max(skipna=True),
        # Temporal
        "start_timestamp": start_ts.isoformat() if start_ts is not None else None,
        "end_timestamp": end_ts.isoformat() if end_ts is not None else None,
        "duration_seconds": duration_seconds,
        # Spatial
        "latitude": df_event["latitude"].dropna().iloc[0] if "latitude" in df_event and not df_event["latitude"].dropna().empty else None,
        "longitude": df_event["longitude"].dropna().iloc[0] if "longitude" in df_event and not df_event["longitude"].dropna().empty else None,
        # Internal
        "file_paths": unique_relative_paths,
        "event_files": event_files,
    }


def _parse_detection_timestamp(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    value_str = str(value).strip()
    if not value_str:
        return None

    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value_str, fmt)
        except ValueError:
            continue

    return pd.to_datetime(value_str, errors="coerce")


def _normalize_species_label(label):
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return "unclassified"
    label_str = str(label).strip()
    return label_str.lower() if label_str else "unclassified"
