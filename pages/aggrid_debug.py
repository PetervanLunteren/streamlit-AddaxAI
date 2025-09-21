"""
Debug page for AgGrid image rendering
"""

import warnings
import sys

if 'warnings' not in sys.modules:
    sys.modules['warnings'] = warnings

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode
import pandas as pd

st.set_page_config(layout="wide")
st.title("AgGrid Debug")

# Check if detection results are available
if "results_detections" not in st.session_state:
    st.error("No detection results found.")
    st.stop()

# Load detection results dataframe
df = st.session_state["results_detections"]

# Show first few rows of data to understand structure
st.write("First row of data:")
st.write(df.iloc[0] if not df.empty else "No data")

# Create test data with simple base64 image
test_df = pd.DataFrame({
    'test': ['Test 1', 'Test 2'],
    'image': [
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==',
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='
    ]
})

gb = GridOptionsBuilder.from_dataframe(test_df)

# Test with simple renderer
gb.configure_column(
    "image",
    headerName="Image Test",
    cellRenderer=JsCode("""
        function(params) {
            console.log('Renderer called with value:', params.value);
            if (params.value) {
                return '<img src="' + params.value + '" style="width:50px;height:50px;background:red;" />';
            }
            return 'NO VALUE';
        }
    """),
    width=120
)

grid_options = gb.build()

st.write("Test Grid (should show red squares):")
grid_response = AgGrid(
    test_df,
    gridOptions=grid_options,
    height=200,
    allow_unsafe_jscode=True,
    theme='streamlit'
)

st.write("Grid response selected rows:", grid_response['selected_rows'])