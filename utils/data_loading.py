"""
AddaxAI Data Loading Utilities

Functions for loading and processing data from completed deployments including:
- Detection results from addaxai-deployment.json files
- Project and deployment metadata aggregation
- DataFrame creation for analysis and visualization
"""

import os
import json
import pandas as pd
import streamlit as st
from utils.config import log


def load_detection_results_dataframe():
    """
    Load all detection results from completed deployments into a pandas DataFrame.
    
    Loops through all projects and deployments in the MAP_JSON file, reads each 
    addaxai-deployment.json file, and flattens all detections into a single dataframe.
    
    Returns:
        pd.DataFrame: Detection results with columns:
            - project_id: Project identifier
            - location_id: Location identifier  
            - deployment_id: Deployment identifier
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
    
    try:
        # Load MAP_JSON from session state
        map_file_path = st.session_state["shared"]["MAP_FILE_PATH"]
        
        if not os.path.exists(map_file_path):
            log(f"MAP_JSON file not found: {map_file_path}")
            return pd.DataFrame()
            
        with open(map_file_path, 'r') as f:
            map_data = json.load(f)
            
        # Loop through all projects, locations, and deployments
        for project_id, project_data in map_data.get("projects", {}).items():
            for location_id, location_data in project_data.get("locations", {}).items():
                for deployment_id, deployment_data in location_data.get("deployments", {}).items():
                    
                    deployment_folder = deployment_data.get("folder")
                    if not deployment_folder:
                        continue
                        
                    deployment_json_path = os.path.join(deployment_folder, "addaxai-deployment.json")
                    
                    if not os.path.exists(deployment_json_path):
                        log(f"Deployment JSON not found: {deployment_json_path}")
                        error_count += 1
                        continue
                        
                    try:
                        # Load deployment JSON
                        with open(deployment_json_path, 'r') as f:
                            deployment_json = json.load(f)
                            
                        # Extract metadata
                        metadata = deployment_json.get("addaxai_metadata", {})
                        detection_model_id = metadata.get("selected_det_modelID", "unknown")
                        classification_model_id = metadata.get("selected_cls_modelID", "unknown")
                        
                        # Get location coordinates from MAP_JSON
                        latitude = location_data.get("lat")
                        longitude = location_data.get("lon")
                        
                        # Process each image and its detections
                        for image_data in deployment_json.get("images", []):
                            image_filename = image_data.get("file")
                            image_datetime = image_data.get("datetime")
                            image_width = image_data.get("width")
                            image_height = image_data.get("height")
                            
                            absolute_path = os.path.join(deployment_folder, image_filename) if image_filename else None
                            
                            # Process each detection in the image
                            for detection in image_data.get("detections", []):
                                
                                # Extract detection data
                                detection_category = detection.get("category", "unknown")
                                detection_conf = detection.get("conf", 0.0)
                                bbox = detection.get("bbox", [0, 0, 0, 0])
                                
                                # Extract classification data (may not exist)
                                classification_label = None
                                classification_conf = None
                                classifications = detection.get("classifications", [])
                                if classifications and len(classifications) > 0:
                                    # Take the first classification (highest confidence)
                                    classification_label = str(classifications[0][0])
                                    classification_conf = float(classifications[0][1])
                                
                                # Create row data
                                row = {
                                    'project_id': project_id,
                                    'location_id': location_id,
                                    'deployment_id': deployment_id,
                                    'absolute_path': absolute_path,
                                    'relative_path': image_filename,
                                    'detection_label': str(detection_category),
                                    'detection_confidence': float(detection_conf),
                                    'classification_label': classification_label,
                                    'classification_confidence': classification_conf,
                                    'bbox_x': float(bbox[0]) if len(bbox) >= 1 else 0.0,
                                    'bbox_y': float(bbox[1]) if len(bbox) >= 2 else 0.0,
                                    'bbox_width': float(bbox[2]) if len(bbox) >= 3 else 0.0,
                                    'bbox_height': float(bbox[3]) if len(bbox) >= 4 else 0.0,
                                    'timestamp': image_datetime,
                                    'latitude': latitude,
                                    'longitude': longitude,
                                    'image_width': image_width,
                                    'image_height': image_height,
                                    'detection_model_id': detection_model_id,
                                    'classification_model_id': classification_model_id
                                }
                                
                                results_list.append(row)
                                
                    except json.JSONDecodeError as e:
                        log(f"Invalid JSON in {deployment_json_path}: {str(e)}")
                        error_count += 1
                        continue
                    except Exception as e:
                        log(f"Error processing {deployment_json_path}: {str(e)}")
                        error_count += 1
                        continue
                        
        # Create DataFrame
        df = pd.DataFrame(results_list)
        
        # Log summary
        if len(df) > 0:
            log(f"Successfully loaded {len(df)} detections from {len(df['deployment_id'].unique())} deployments")
        else:
            log("No detection results found in any deployments")
            
        if error_count > 0:
            log(f"Encountered {error_count} errors while loading detection results")
            
        return df
        
    except Exception as e:
        log(f"Fatal error in load_detection_results_dataframe: {str(e)}")
        return pd.DataFrame()