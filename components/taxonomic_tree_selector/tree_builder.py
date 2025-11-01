"""
Taxonomic tree builder utilities.

Provides a single entry point for constructing the hierarchical species
tree used across the application.
"""

import streamlit as st

from utils.analysis_utils import build_taxon_tree


def build_tree_from_species(species_list):
    """
    Build hierarchical tree structure for the provided species list.

    Args:
        species_list (list[str]): Model class identifiers to include in the tree.

    Returns:
        list[dict]: Tree structure compatible with st-checkbox-tree.
    """

    taxonomy = st.session_state.get("taxonomy", {})
    if not taxonomy:
        return []

    taxon_mapping = []
    for model_class in species_list:
        taxon_data = taxonomy.get(model_class)
        if not taxon_data:
            continue

        taxon_mapping.append({
            "model_class": model_class,
            "class": (taxon_data.get("class") or "").strip(),
            "order": (taxon_data.get("order") or "").strip(),
            "family": (taxon_data.get("family") or "").strip(),
            "genus": (taxon_data.get("genus") or "").strip(),
            "species": (taxon_data.get("species") or "").strip(),
        })

    return build_taxon_tree(taxon_mapping)


def get_all_species_from_tree(tree_list):
    """
    Extract all leaf node values (species names) from a tree.

    Args:
        tree_list: Tree structure

    Returns:
        list: All species names (leaf values)
    """
    species = []

    for node in tree_list:
        if "children" in node and node["children"]:
            # Recurse into children
            species.extend(get_all_species_from_tree(node["children"]))
        else:
            # Leaf node - add the value (species name)
            species.append(node["value"])

    return species
