

import streamlit as st
from st_aggrid import AgGrid
import pandas as pd
from st_aggrid import GridOptionsBuilder
import os
from backend.utils import print_widget_label

def get_location_deployment_nested_dict(df):
    
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