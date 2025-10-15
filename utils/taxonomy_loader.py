"""
Global Taxonomy Loader

Loads all taxonomy data once at app startup and stores in session state.
This provides a single source of truth for species taxonomy across all tools.
"""

import os
import json
import pandas as pd
import streamlit as st
from utils.config import log, ADDAXAI_ROOT


def load_global_taxonomy():
    """
    Load all taxonomy data once and store in st.session_state["taxonomy"].

    This function:
    1. Reads all classification models from completed runs
    2. For SpeciesNet: uses classification_category_descriptions from JSON
    3. For other models: loads from taxon-mapping.csv
    4. Builds a flat dictionary of all unique species with their taxonomy

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
    speciesnet_taxonomy_descriptions = {}

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

                            # For SpeciesNet models, capture taxonomy descriptions from JSON
                            if "SPECIESNET" in classification_model_id.upper():
                                category_descriptions = run_json.get("classification_category_descriptions", {})
                                if category_descriptions and classification_model_id not in speciesnet_taxonomy_descriptions:
                                    speciesnet_taxonomy_descriptions[classification_model_id] = category_descriptions

                    except Exception as e:
                        log(f"Error processing {run_json_path}: {str(e)}")
                        continue

        # Now load taxonomy for each unique classification model
        for model_id in classification_models_seen:
            log(f"Loading taxonomy for model: {model_id}")

            # Check if this is a SpeciesNet model
            if "SPECIESNET" in model_id.upper() and model_id in speciesnet_taxonomy_descriptions:
                # Use SpeciesNet taxonomy from JSON
                category_descriptions = speciesnet_taxonomy_descriptions[model_id]

                for category_id, taxonomy_string in category_descriptions.items():
                    # Skip empty or whitespace-only strings
                    if not taxonomy_string or not taxonomy_string.strip():
                        continue

                    # Split taxonomy string: "mammalia;carnivora;felidae;panthera;leo"
                    parts = [p.strip() for p in taxonomy_string.split(';')]

                    # Skip entries that don't have at least class and species
                    if len(parts) < 5 or not parts[0] or not parts[4]:
                        continue

                    model_class = parts[4]  # Species name is the model class

                    # Add to taxonomy dict if not already present
                    if model_class not in taxonomy:
                        taxonomy[model_class] = {
                            "class": parts[0].lower(),
                            "order": parts[1].lower() if parts[1] else "",
                            "family": parts[2].lower() if parts[2] else "",
                            "genus": parts[3].lower() if parts[3] else "",
                            "species": parts[4].lower()
                        }

                log(f"  ✓ Loaded {len([k for k in taxonomy if k])} species from SpeciesNet")

            else:
                # Load from taxon-mapping.csv
                mapping_path = os.path.join(ADDAXAI_ROOT, "models", "cls", model_id, "taxon-mapping.csv")

                if not os.path.exists(mapping_path):
                    log(f"  ✗ No taxon-mapping.csv found for model: {model_id}")
                    continue

                try:
                    taxon_df = pd.read_csv(mapping_path)

                    for _, row in taxon_df.iterrows():
                        model_class = row.get("model_class", "").strip()

                        if not model_class or model_class in taxonomy:
                            continue

                        # Extract taxonomy fields
                        # CSV format has fields like "level_class", "level_order", etc.
                        # with values like "class Mammalia", "order Carnivora"
                        level_class = row.get("level_class", "")
                        level_order = row.get("level_order", "")
                        level_family = row.get("level_family", "")
                        level_genus = row.get("level_genus", "")
                        level_species = row.get("level_species", "")

                        # Extract just the taxonomic name (remove prefix like "class ", "order ")
                        # Only extract if the value has the proper prefix for that level
                        def extract_name(level_value, expected_prefix):
                            if not level_value or pd.isna(level_value):
                                return ""
                            level_value = str(level_value).strip()
                            if not level_value:
                                return ""
                            # Check if it has the expected prefix
                            if not level_value.lower().startswith(expected_prefix.lower() + " "):
                                return ""
                            parts = level_value.split(" ", 1)
                            return parts[1].lower() if len(parts) > 1 else ""

                        # Extract all taxonomic levels with validation
                        extracted_class = extract_name(level_class, "class")
                        extracted_order = extract_name(level_order, "order")
                        extracted_family = extract_name(level_family, "family")
                        extracted_genus = extract_name(level_genus, "genus")
                        extracted_species = extract_name(level_species, "species")

                        # If species name equals genus name, it means there's no real species differentiation
                        # Set species to empty string in this case
                        if extracted_species == extracted_genus:
                            extracted_species = ""

                        taxonomy[model_class] = {
                            "class": extracted_class,
                            "order": extracted_order,
                            "family": extracted_family,
                            "genus": extracted_genus,
                            "species": extracted_species
                        }

                    log(f"  ✓ Loaded taxonomy from CSV for {model_id}")

                except Exception as e:
                    log(f"  ✗ Error loading taxonomy for {model_id}: {str(e)}")
                    continue

        # Store in session state
        st.session_state["taxonomy"] = taxonomy
        log(f"Global taxonomy loaded: {len(taxonomy)} unique species")

        return taxonomy

    except Exception as e:
        log(f"Fatal error in load_global_taxonomy: {str(e)}")
        st.session_state["taxonomy"] = {}
        return {}
