
"""
AddaxAI Data Verification Utilities

Functions for validating and organizing deployment data:
- Location-deployment hierarchical structures
- Data integrity checks for camera trap datasets
- UI components for data exploration and verification
"""

import streamlit as st
# from st_aggrid import AgGrid  # UNUSED: Vulture detected unused import
import pandas as pd
# from st_aggrid import GridOptionsBuilder  # UNUSED: Vulture detected unused import
import os
from components import print_widget_label

def get_location_deployment_nested_dict(df):
    """
    Convert deployment DataFrame to hierarchical location-deployment structure.
    
    Args:
        df (pd.DataFrame): DataFrame with 'location' and 'deployment' columns
        
    Returns:
        list: Hierarchical nodes for checkbox tree widget with locations as parents
              and deployments as children
    """
    
    d = (
        df[['location', 'deployment']]
        .drop_duplicates()
        .groupby('location')['deployment']
        .unique()
        .apply(list)
        .to_dict()
    )
    
    nodes = []
    for loc_key, deploy_list in d.items():
        node = {
            "label": loc_key, 
            "value": loc_key, 
        }
        # If there are deploy items, add them as children nodes
        if deploy_list:
            children = []
            for deploy_id in deploy_list:
                children.append({
                    "label": deploy_id,
                    "value": deploy_id
                })
            node["children"] = children
        nodes.append(node)
    return nodes