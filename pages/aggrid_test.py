"""
Test page to debug AgGrid image rendering
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
st.title("AgGrid Image Test")

# Create test data with a simple base64 image (1x1 red pixel)
test_data = pd.DataFrame({
    'test_column': ['Row 1', 'Row 2', 'Row 3'],
    'image': [
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==',
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==',
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='
    ]
})

gb = GridOptionsBuilder.from_dataframe(test_data)

# Test 1: Simple HTML test
gb.configure_column(
    "test_column",
    headerName="HTML Test",
    cellRenderer=JsCode("function(params) { return '<b style=\"color:red;\">' + params.value + '</b>'; }")
)

# Test 2: Image renderer
gb.configure_column(
    "image",
    headerName="Image Test",
    cellRenderer=JsCode("""
        function(params) {
            return '<img src="' + params.value + '" style="width:50px;height:50px;border:2px solid red;" />';
        }
    """)
)

grid_options = gb.build()

st.write("If you see:")
st.write("- **Bold red text** in first column → JS is working")
st.write("- **Red squares** in second column → Images are working")
st.write("- Plain text or base64 strings → JS is not executing")

grid_response = AgGrid(
    test_data,
    gridOptions=grid_options,
    height=300,
    allow_unsafe_jscode=True,
    theme='streamlit'
)