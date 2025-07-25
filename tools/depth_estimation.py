import streamlit as st
import os
import pandas as pd
from utils.common import print_widget_label
from streamlit_tree_select import tree_select
from utils.data_verification import get_location_deployment_nested_dict
import streamlit_antd_components as sac
from utils.depth_estimation_utils import debug_deployment_selector

st.markdown("*This is where the user can choose to do depth estimation with depth anything (https://github.com/DepthAnything/Depth-Anything-V2)*")


st.markdown("*It would probabaly make sense if the user would select which deployments he wants to do the depth estimation for, as for each deployment the user probabaly need to add some calibraiton points.*")
st.markdown("*(see some example images and results in /namibia-example-images/)*")



# debug this needs to be real data somehow


# Deployment multiselect
st.divider()
print_widget_label("For which deployment do you want to estimate depth?", help_text = "Because of the calibration needed, it is required to do the depth estimation for each deployment separately.")
# selected_deployment = deployment_selector()





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


selected_deployment = debug_deployment_selector(example_data)

st.write("Selected Deployment:", selected_deployment)




if selected_deployment:
    st.success(f"Selected deployment: {selected_deployment}")
else:
    st.info("Please select a deployment.")


st.markdown("\n\n*Then do the calibration of at least one or two points?*")

st.markdown("\n\n*And then the depth will be estimated and added to the resuilts file CSV*")
