#!/bin/bash

# conda create -n streamlit python=3.7
# conda activate streamlit
# pip install streamlit
# pip install streamlit-extras
# pip install st-pages
# pip install streamlit-aggrid
# pip install folium
# pip install streamlit-folium
# pip install exifread
# pip install ffmpeg
# pip install gpsphoto
# pip install piexif

# Activate conda environment
source /Users/peter/miniforge3/etc/profile.d/conda.sh
conda activate streamlit

# find folder of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/frontend"

# Start Streamlit in the background
pkill -f streamlit
streamlit run main.py >> streamlit_log.txt 2>&1 &
