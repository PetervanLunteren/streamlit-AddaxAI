"""
Tree Builder for Taxonomic Hierarchy

Builds hierarchical tree structures from flat species lists using global taxonomy.
"""

import streamlit as st


def build_tree_from_species(species_list):
    """
    Build hierarchical tree structure for given species.

    Looks up taxonomy from st.session_state["taxonomy"] and builds
    a tree compatible with st-checkbox-tree widget.

    Args:
        species_list: List of species names (model_class values)
                     e.g. ["lion", "cat", "bird"]

    Returns:
        list: Tree structure with format:
              [
                  {
                      "label": "class mammalia",
                      "value": "class mammalia",
                      "children": [...]
                  },
                  ...
              ]
    """
    # Get global taxonomy
    taxonomy = st.session_state.get("taxonomy", {})

    if not taxonomy:
        return []

    # Build tree using dictionary for efficient lookup
    root = {}

    for model_class in species_list:
        if model_class not in taxonomy:
            # Species not in taxonomy, skip it
            continue

        taxon_data = taxonomy[model_class]

        # Extract taxonomic levels (stored without "class ", "order " prefix in taxonomy dict)
        class_name = taxon_data.get("class", "").strip()
        order_name = taxon_data.get("order", "").strip()
        family_name = taxon_data.get("family", "").strip()
        genus_name = taxon_data.get("genus", "").strip()
        species_name = taxon_data.get("species", "").strip()

        # Check if we have proper class level - if not, place at root with unknown taxonomy
        if not class_name:
            # No class information - place at root level
            taxonomic_value = species_name if species_name else model_class
            label = f"{taxonomic_value} (<b>{model_class}</b>, <i>unknown taxonomy</i>)"
            value = model_class
            if value not in root:
                root[value] = {
                    "label": label,
                    "value": value,
                    "children": {}
                }
            continue

        # Build path components for unique node values
        current_level = root
        last_taxon_name = None
        path_components = []
        last_valid_level_name = None  # Track the last level we actually processed

        # Process each taxonomic level with proper prefixes
        levels = [
            ("class", class_name),
            ("order", order_name),
            ("family", family_name),
            ("genus", genus_name),
            ("species", species_name)
        ]

        # Check if all non-empty levels have the same name (e.g., Aves, Aves, Aves, Aves, Aves)
        # In this case, we should stop at the first level and mark as "unspecified"
        non_empty_names = [name for _, name in levels if name]
        all_same = len(set(non_empty_names)) == 1 if non_empty_names else False

        if all_same:
            # All taxonomic levels are the same (e.g., class Aves, species Aves)
            # Create single leaf node showing: "class aves (<b>Bird</b>, <i>unspecified</i>)"
            label = f"class {class_name} (<b>{model_class}</b>, <i>unspecified</i>)"
            value = model_class

            if value not in current_level:
                current_level[value] = {
                    "label": label,
                    "value": value,
                    "children": {}
                }
            continue

        for i, (level_name, taxon_name) in enumerate(levels):
            if not taxon_name:
                continue

            # Find the last non-empty level
            is_last_level = all(not levels[j][1] for j in range(i + 1, len(levels)))

            # Build label with taxonomic prefix
            label_with_prefix = f"{level_name} {taxon_name}"

            if not is_last_level:
                # Intermediate level (class, order, family, genus)
                # Skip duplicate consecutive levels
                if label_with_prefix == last_taxon_name:
                    continue

                # Build unique value using full path
                path_components.append(label_with_prefix)
                value = "|".join(path_components)

                if value not in current_level:
                    current_level[value] = {
                        "label": label_with_prefix,
                        "value": value,
                        "children": {}
                    }

                current_level = current_level[value]["children"]
                last_taxon_name = label_with_prefix
                last_valid_level_name = level_name

            else:
                # Last level - create leaf node
                if level_name == "species":
                    # Proper species level
                    label = f"{label_with_prefix} (<b>{model_class}</b>)"
                else:
                    # Last level but not a species (e.g., only genus available)
                    label = f"{label_with_prefix} (<b>{model_class}</b>, <i>unspecified</i>)"

                value = model_class

                if value not in current_level:
                    current_level[value] = {
                        "label": label,
                        "value": value,
                        "children": {}
                    }

    # Convert nested dict to list format
    tree_list = _dict_to_list(root)

    # Sort tree (optional: leaves first for better UX)
    sorted_tree = _sort_tree(tree_list)

    return sorted_tree


def _dict_to_list(tree_dict):
    """
    Convert nested dictionary to list format for st-checkbox-tree.

    Args:
        tree_dict: Nested dictionary of tree nodes

    Returns:
        list: List of tree nodes
    """
    result = []

    for _, node_data in tree_dict.items():
        node = {
            "label": node_data["label"],
            "value": node_data["value"]
        }

        # Recursively process children
        if node_data["children"]:
            children_list = _dict_to_list(node_data["children"])
            if children_list:
                node["children"] = children_list

        result.append(node)

    return result


def _sort_tree(tree_list):
    """
    Sort tree nodes for better UX.

    Sorts each level alphabetically by label, with leaf nodes appearing first.

    Args:
        tree_list: List of tree nodes

    Returns:
        list: Sorted tree
    """
    if not tree_list:
        return tree_list

    # Separate leaves and parents
    leaves = [node for node in tree_list if "children" not in node]
    parents = [node for node in tree_list if "children" in node]

    # Sort each group alphabetically by label
    leaves_sorted = sorted(leaves, key=lambda x: x["label"].lower())
    parents_sorted = sorted(parents, key=lambda x: x["label"].lower())

    # Recursively sort children of parent nodes
    for parent in parents_sorted:
        parent["children"] = _sort_tree(parent["children"])

    # Return leaves first, then parents
    return leaves_sorted + parents_sorted


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
