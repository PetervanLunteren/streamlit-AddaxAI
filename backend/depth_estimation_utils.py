import streamlit as st
import os
import pandas as pd

# def deployment_selector(): #### SEE BELOW FOR NEW VERSION
#     # Load data
#     csv_path = os.path.join( # DEBUG
#         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#         "namibia-example-images",
#         "results_detections.csv"
#     )
#     df = pd.read_csv(csv_path)

#     # Make sure 'depth_estimated' column exists and is bool
#     if "depth_estimated" not in df.columns:
#         df["depth_estimated"] = False
#     df["depth_estimated"] = df["depth_estimated"].astype(bool)

#     grouped = df.groupby(["location", "deployment"])["depth_estimated"].any().reset_index()

#     selected_deployment = None

#     with st.container(border=True, height=300):
#         # For each location add an expander
#         for location, group_df in grouped.groupby("location"):
#             with st.expander(f":material/pin_drop: {location}", expanded=False):
#                 for _, row in group_df.iterrows():
#                     col1, col2 = st.columns([1, 1], vertical_alignment="center")
#                     with col1:
#                         if st.button(f":material/sd_card: {row.deployment}", key=f"{location}_{row.deployment}", use_container_width=True):
#                             selected_deployment = row.deployment
#                     with col2:
#                         st.markdown(f"{'depth estimated already' if row.depth_estimated else 'depth not yet estimated'}")

#     return selected_deployment



# def debug_deployment_selector(data_dict):
#     selected = {"project": None, "location": None, "deployment": None}

#     with st.container(border=True, height=400):
#         for project, locations in data_dict.items():
#             with st.expander(f":material/folder: {project}", expanded=False):
#                 for location, deployments in locations.items():
#                     with st.expander(f":material/pin_drop: {location}", expanded=False):
#                         for deployment, metadata in deployments.items():
#                             col1, col2, col3 = st.columns([4, 1, 1], vertical_alignment="center")
#                             with col1:
#                                 st.markdown(f":material/sd_card: {deployment}")
#                             with col2:
#                                 help_text = (
#                                     "**Depth Estimated:** {depth}\n\n"
#                                     "**Image Count:** {img_count}\n\n"
#                                     "**Last Updated:** {last_updated}"
#                                 ).format(
#                                     depth="✅ Yes" if metadata.get("depth_estimated") else "❌ No",
#                                     img_count=metadata.get("image_count", "n/a"),
#                                     last_updated=metadata.get("last_updated", "n/a"),
#                                 )
#                                 if st.button(f":material/info: Info", key=f"info_{project}_{location}_{deployment}", use_container_width=True, 
#                                              help=help_text, type ="tertiary"):
#                                     print("dummy print to avoid error")
#                             with col3:
#                                 if st.button(f":material/edit: Edit", key=f"edit_{project}_{location}_{deployment}", use_container_width=True):
#                                     selected["project"] = project
#                                     selected["location"] = location
#                                     selected["deployment"] = deployment
#     return selected


import streamlit_nested_layout

def debug_deployment_selector(data_dict):
    # Track selected edit in session state
    if "selected_edit" not in st.session_state:
        st.session_state.selected_edit = {"project": None, "location": None, "deployment": None}

    with st.container(border=True, height=400):
        for project, locations in data_dict.items():
            with st.expander(f":material/folder: {project}", expanded=False):
                for location, deployments in locations.items():
                    with st.expander(f":material/pin_drop: {location}", expanded=False):
                        for deployment, metadata in deployments.items():
                            col1, col2, col3 = st.columns([4, 1, 1], vertical_alignment="center")

                            with col1:
                                st.markdown(f":material/sd_card: {deployment}")

                            with col2:
                                help_text = (
                                    "**Depth Estimated:** {depth}\n\n"
                                    "**Image Count:** {img_count}\n\n"
                                    "**Last Updated:** {last_updated}"
                                ).format(
                                    depth="✅ Yes" if metadata.get("depth_estimated") else "❌ No",
                                    img_count=metadata.get("image_count", "n/a"),
                                    last_updated=metadata.get("last_updated", "n/a"),
                                )
                                st.button(f":material/info: Info", key=f"info_{project}_{location}_{deployment}",
                                          use_container_width=True, help=help_text, type="tertiary")

                            with col3:
                                is_selected = (
                                    st.session_state.selected_edit["project"] == project and
                                    st.session_state.selected_edit["location"] == location and
                                    st.session_state.selected_edit["deployment"] == deployment
                                )

                                # Use a unique key
                                button_key = f"edit_{project}_{location}_{deployment}"
                                if st.button(f":material/edit: Edit", key=button_key,
                                             use_container_width=True,
                                             type="primary" if is_selected else "secondary"):
                                    # Immediately update and rerun
                                    st.session_state.selected_edit = {
                                        "project": project,
                                        "location": location,
                                        "deployment": deployment
                                    }
                                    st.rerun()  # <-- fix the delayed update

    return st.session_state.selected_edit


example_data = {
    "Namibia Project": {
        "Etosha": {
            "CamTrap_001": {
                "depth_estimated": True,
                "image_count": 1245,
                "last_updated": "2025-05-12"
            },
            "CamTrap_002": {
                "depth_estimated": False,
                "image_count": 542,
                "last_updated": "2025-04-30"
            }
        },
        "Waterberg": {
            "CamTrap_003": {
                "depth_estimated": False,
                "image_count": 873,
                "last_updated": "2025-06-01"
            },
            "CamTrap_004": {
                "depth_estimated": True,
                "image_count": 1920,
                "last_updated": "2025-05-20"
            }
        }
    },
    "Australia Project": {
        "Kakadu": {
            "CamTrap_101": {
                "depth_estimated": True,
                "image_count": 311,
                "last_updated": "2025-06-10"
            },
            "CamTrap_102": {
                "depth_estimated": False,
                "image_count": 98,
                "last_updated": "2025-06-05"
            }
        },
        "Daintree": {
            "CamTrap_201": {
                "depth_estimated": True,
                "image_count": 1550,
                "last_updated": "2025-05-25"
            }
        }
    },
    "Peru Project": {
        "Madre de Dios": {
            "CamTrap_301": {
                "depth_estimated": False,
                "image_count": 755,
                "last_updated": "2025-04-18"
            },
            "CamTrap_302": {
                "depth_estimated": True,
                "image_count": 1340,
                "last_updated": "2025-05-30"
            }
        }
    }
}

selection = debug_deployment_selector(example_data)

st.write("Selected:", selection)
