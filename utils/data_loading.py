"""
AddaxAI Data Loading Utilities

Functions for loading and processing data from completed runs including:
- Detection results from addaxai-run.json files
- Project and run metadata aggregation
- DataFrame creation for analysis and visualization
"""

import os
import json
import pandas as pd
import streamlit as st
from utils.config import log

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


def load_model_taxonomy(model_id, classification_category_descriptions=None):
    """
    Load taxonomy information for a classification model.

    Tries to load from classification_category_descriptions first (from SpeciesNet),
    then falls back to taxon-mapping.csv for other models.

    Args:
        model_id (str): Classification model ID
        classification_category_descriptions (dict, optional): Taxonomy mapping from JSON
            Format: {"0": "mammalia;carnivora;felidae;panthera;leo", ...}

    Returns:
        dict: {
            "tree": [...],           # Hierarchical tree structure
            "leaf_values": [...],    # All species (model_class values)
            "taxon_mapping": [...]   # Raw CSV data as list of dicts
        } or None if loading fails
    """
    from utils.config import ADDAXAI_ROOT
    from utils.analysis_utils import build_taxon_tree, get_all_leaf_values

    try:
        # Check if model is NONE or unknown
        if not model_id or model_id in ["NONE", "unknown"]:
            return None

        # Try to use classification_category_descriptions from JSON first (SpeciesNet)
        if classification_category_descriptions:
            log(f"Using taxonomy from classification_category_descriptions for {model_id}")

            # Convert SpeciesNet format to taxon_mapping format
            # SpeciesNet format: {"0": "mammalia;carnivora;felidae;panthera;leo"}
            # Need to convert to: [{"model_class": "leo", "level_class": "class mammalia", ...}]
            taxon_mapping = []
            for category_id, taxonomy_string in classification_category_descriptions.items():
                # Skip empty or whitespace-only strings
                if not taxonomy_string or not taxonomy_string.strip():
                    continue

                # Split taxonomy string: "mammalia;carnivora;felidae;panthera;leo"
                parts = [p.strip() for p in taxonomy_string.split(';')]

                # Skip entries that don't have at least class and species (parts[0] and parts[4])
                if len(parts) < 5 or not parts[0] or not parts[4]:
                    log(f"Skipping malformed taxonomy entry: {taxonomy_string}")
                    continue

                # Assume format is: class;order;family;genus;species
                # build_taxon_tree expects level_* fields with "class ", "order ", etc. prefixes
                entry = {
                    "model_class": parts[4],  # species
                    "level_class": f"class {parts[0]}",
                    "level_order": f"order {parts[1]}" if parts[1] else "",
                    "level_family": f"family {parts[2]}" if parts[2] else "",
                    "level_genus": f"genus {parts[3]}" if parts[3] else "",
                    "level_species": f"species {parts[4]}"
                }
                taxon_mapping.append(entry)

            if taxon_mapping:
                tree = build_taxon_tree(taxon_mapping)
                leaf_values = get_all_leaf_values(tree)

                return {
                    "tree": tree,
                    "leaf_values": leaf_values,
                    "taxon_mapping": taxon_mapping
                }

        # Fallback: Load from taxon-mapping.csv
        mapping_path = os.path.join(
            ADDAXAI_ROOT, "models", "cls", model_id, "taxon-mapping.csv"
        )

        if not os.path.exists(mapping_path):
            log(f"No taxon-mapping.csv found for model: {model_id}")
            return None

        log(f"Using taxonomy from taxon-mapping.csv for {model_id}")

        # Load and parse taxonomy
        taxon_df = pd.read_csv(mapping_path)
        taxon_mapping = taxon_df.to_dict('records')

        # Build tree structure
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

                    # Normalize taxonomic levels to lowercase for consistent merging
                    # This ensures "class Mammalia" and "class mammalia" are treated as the same node
                    normalized_entry = entry.copy()
                    for level_key in ["level_class", "level_order", "level_family", "level_genus", "level_species"]:
                        if level_key in normalized_entry and normalized_entry[level_key]:
                            normalized_entry[level_key] = normalized_entry[level_key].lower()

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
    error_count = 0
    classification_models_seen = set()
    speciesnet_taxonomy_descriptions = {}  # Track classification_category_descriptions for SpeciesNet only

    try:
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

                            # For SpeciesNet models, capture taxonomy descriptions from JSON
                            if "SPECIESNET" in classification_model_id.upper():
                                category_descriptions = run_json.get("classification_category_descriptions", {})
                                if category_descriptions and classification_model_id not in speciesnet_taxonomy_descriptions:
                                    speciesnet_taxonomy_descriptions[classification_model_id] = category_descriptions

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
                            
                            # Process each detection in the image
                            for detection in image_data.get("detections", []):
                                
                                # Extract detection data - no defaults, should always be present
                                detection_category_id = detection["category"]
                                detection_conf = detection["conf"]
                                bbox = detection["bbox"]
                                
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
                                    'project_id': project_id,
                                    'location_id': location_id,
                                    'run_id': run_id,
                                    'absolute_path': absolute_path,
                                    'relative_path': image_filename,
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
                                    'classification_model_id': classification_model_id
                                }
                                
                                results_list.append(row)
                                
                    except json.JSONDecodeError as e:
                        log(f"Invalid JSON in {run_json_path}: {str(e)}")
                        error_count += 1
                        continue
                    except Exception as e:
                        log(f"Error processing {run_json_path}: {str(e)}")
                        error_count += 1
                        continue
                        
        # Create DataFrame
        df = pd.DataFrame(results_list)

        # Load taxonomy for each classification model encountered
        taxonomy_dict = {}
        if classification_models_seen:
            log(f"Loading taxonomy for {len(classification_models_seen)} classification model(s)...")

            for model_id in classification_models_seen:
                # For SpeciesNet, pass taxonomy descriptions from JSON
                category_descriptions = speciesnet_taxonomy_descriptions.get(model_id)
                taxonomy_data = load_model_taxonomy(model_id, classification_category_descriptions=category_descriptions)
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