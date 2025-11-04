"""
Global Taxonomy Loader

Loads all taxonomy data once at app startup and stores in session state.
This provides a single source of truth for species taxonomy across all tools.
"""

import os
import json
import streamlit as st
from utils.config import log
from utils.data_loading import load_model_taxonomy


def load_global_taxonomy():
    """
    Load all taxonomy data once and store in st.session_state["taxonomy"].

    This function:
    1. Reads all classification models referenced by completed runs
    2. Loads taxonomy data from each model's taxonomy.csv (SpeciesNet models fall back to embedded run metadata)
    3. Builds a flat dictionary of all unique species with their taxonomy

    Stores result in st.session_state["taxonomy"] as:
    {
        "lion": {
            "class": "mammalia",
            "order": "carnivora",
            "family": "felidae",
            "genus": "panthera",
            "species": "leo"
        },
        ...
    }

    Returns:
        dict: The taxonomy dictionary (also stored in session state)
    """

    # Check if already loaded
    if "taxonomy" in st.session_state:
        return st.session_state["taxonomy"]

    log("Loading global taxonomy data...")

    taxonomy = {}
    classification_models_seen = set()
    model_taxonomy_descriptions = {}
    model_category_labels = {}

    try:
        # Load MAP_JSON from session state
        map_file_path = st.session_state["shared"]["MAP_FILE_PATH"]

        if not os.path.exists(map_file_path):
            log(f"MAP_JSON file not found: {map_file_path}")
            st.session_state["taxonomy"] = {}
            return {}

        with open(map_file_path, 'r') as f:
            map_data = json.load(f)

        # Loop through all runs to find classification models and their taxonomy
        for project_id, project_data in map_data.get("projects", {}).items():
            for location_id, location_data in project_data.get("locations", {}).items():
                for run_id, run_data in location_data.get("runs", {}).items():

                    run_folder = run_data.get("folder")
                    if not run_folder:
                        continue

                    run_json_path = os.path.join(run_folder, "addaxai-run.json")

                    if not os.path.exists(run_json_path):
                        continue

                    try:
                        # Load run JSON
                        with open(run_json_path, 'r') as f:
                            run_json = json.load(f)

                        # Extract metadata
                        metadata = run_json.get("addaxai_metadata", {})
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

                    except Exception as e:
                        log(f"Error processing {run_json_path}: {str(e)}")
                        continue

        # Now load taxonomy for each unique classification model
        for model_id in classification_models_seen:
            log(f"Loading taxonomy for model: {model_id}")

            taxonomy_data = load_model_taxonomy(
                model_id,
                classification_category_descriptions=model_taxonomy_descriptions.get(model_id),
                classification_categories=model_category_labels.get(model_id)
            )

            if not taxonomy_data:
                log(f"  ✗ Unable to load taxonomy for model: {model_id}")
                continue

            for entry in taxonomy_data.get("taxon_mapping", []):
                model_class = (entry.get("model_class") or "").strip()
                if not model_class:
                    continue

                if model_class not in taxonomy:
                    taxonomy[model_class] = {
                        "class": (entry.get("class") or "").strip().lower(),
                        "order": (entry.get("order") or "").strip().lower(),
                        "family": (entry.get("family") or "").strip().lower(),
                        "genus": (entry.get("genus") or "").strip().lower(),
                        "species": (entry.get("species") or "").strip().lower()
                    }

            log(f"  ✓ Loaded taxonomy metadata for {model_id}")

        # Store in session state
        st.session_state["taxonomy"] = taxonomy
        log(f"Global taxonomy loaded: {len(taxonomy)} unique species")

        return taxonomy

    except Exception as e:
        log(f"Fatal error in load_global_taxonomy: {str(e)}")
        st.session_state["taxonomy"] = {}
        return {}
