import streamlit as st
import os
import pandas as pd
from backend.utils import print_widget_label
from streamlit_tree_select import tree_select
from backend.data_verification import get_location_deployment_nested_dict

# st.set_page_config(page_title="Tool 1", page_icon="🛠️")

st.title("Tool 1")
st.write("This is Tool 1. You can add more functionality here.")

# debug this needs to be real data somehow
csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "namibia-example-images", "results_detections.csv")
df = pd.read_csv(csv_path)

# Deployment multiselect
st.divider()
print_widget_label("Do you want to filter any locations or deployments?", help_text = "help text")
return_select = tree_select(get_location_deployment_nested_dict(df))
selected_deployments = [item for item in return_select['checked'] if item.startswith("deploy_")]

# Date range picker for DateTimeOriginal
st.divider()
print_widget_label("Choose a date range", help_text = "help text")
df["DateTimeOriginal"] = pd.to_datetime(df["DateTimeOriginal"], errors="coerce")
min_date = df["DateTimeOriginal"].min()
max_date = df["DateTimeOriginal"].max()
selected_dates = st.date_input("Date Range", value=(min_date.date(), max_date.date()), label_visibility="collapsed")


