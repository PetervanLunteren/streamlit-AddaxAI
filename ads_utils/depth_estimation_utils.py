import streamlit as st
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from ads_utils.data_verification import get_location_deployment_nested_dict

@dataclass
class Calibration:
    x: int
    y: int
    dist: float

def deployment_selector():
    """
    Creates a selectbox to choose a deployment.
    """
    data = get_location_deployment_nested_dict()

    options = []
    for project, locations in data.items():
        for location, deployments in locations.items():
            for deployment, info in deployments.items():
                options.append(f"{project}/{location}/{deployment}")

    return st.selectbox("Select a deployment", options)

def get_calibration_points(deployment: str) -> List[Calibration]:
    """
    Retrieves the calibration points for a given deployment.
    """
    calibration_file = os.path.join("deployments", deployment, "calibration.json")
    if os.path.exists(calibration_file):
        with open(calibration_file, 'r') as f:
            data = json.load(f)
            return [Calibration(**p) for p in data]
    return []

def save_calibration_points(deployment: str, points: List[Calibration]):
    """
    Saves the calibration points for a given deployment.
    """
    deployment_dir = os.path.join("deployments", deployment)
    os.makedirs(deployment_dir, exist_ok=True)
    calibration_file = os.path.join(deployment_dir, "calibration.json")
    with open(calibration_file, 'w') as f:
        json.dump([asdict(p) for p in points], f, indent=4)

def debug_deployment_selector(example_data: Dict[str, Any]):
    """
    Creates a selectbox to choose a deployment from example data.
    """
    options = []
    for project, locations in example_data.items():
        for location, deployments in locations.items():
            for deployment, info in deployments.items():
                options.append(f"{project}/{location}/{deployment}")

    return st.selectbox("Select a deployment", options)
