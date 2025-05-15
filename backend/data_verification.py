

import streamlit as st
from st_aggrid import AgGrid
import pandas as pd
from st_aggrid import GridOptionsBuilder

df = pd.read_csv('/Users/peter/Desktop/streamlit_app/example-csvs/results_detections.csv')

# Configure AG Grid options
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination(paginationPageSize=20)  # Enable pagination

# Enable multi-select filter (but not editor)
gb.configure_column(
    "label",
    filter="agSetColumnFilter",  # Multi-select filtering
)

# Enable multi-select filter (but not editor)
gb.configure_column(
    "human_verified",
    filter="agSetColumnFilter",  # Multi-select filtering
)

# Remove extra UI interactions
gb.configure_grid_options(
    # domLayout="autoHeight",  # Prevent unnecessary scrolling
    suppressColumnVirtualisation=True,  # Improves performance for small datasets
    suppressRowVirtualisation=True,  
)

grid_options = gb.build()

# Render AgGrid
grid_response = AgGrid(df, gridOptions=grid_options, update_mode="grid_changed")


# Get filtered data
filtered_df = pd.DataFrame(grid_response["data"])  # DataFrame after filtering

# Show number of rows after filtering
st.write(f"Number of rows after filtering: {filtered_df.shape[0]}")


