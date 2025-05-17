import streamlit as st
import streamlit as st
import os
import pandas as pd
from backend.utils import print_widget_label
from streamlit_tree_select import tree_select
from backend.data_verification import get_location_deployment_nested_dict
import streamlit_antd_components as sac
from backend.depth_estimation_utils import deployment_selector

st.markdown("*This is where the user can choose to do depth estimation with depth anything (https://github.com/DepthAnything/Depth-Anything-V2)*")


st.markdown("*It would probabaly make sense if the user would select which deployments he wants to do the depth estimation for, as for each deployment the user probabaly need to add some calibraiton points.*")
st.markdown("*(see some example images and results in /namibia-example-images/)*")



# debug this needs to be real data somehow


# Deployment multiselect
st.divider()
print_widget_label("For which deployment do you want to estimate depth?", help_text = "Because of the calibration needed, it is required to do the depth estimation for each deployment separately.")
selected_deployment = deployment_selector()


if selected_deployment:
    st.success(f"Selected deployment: {selected_deployment}")
else:
    st.info("Please select a deployment.")


st.markdown("\n\n*Then do the calibration of at least one or two points?*")

st.markdown("\n\n*And then the depth will be estimated and added to the resuilts file CSV*")
