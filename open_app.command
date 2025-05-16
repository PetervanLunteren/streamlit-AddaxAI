#!/bin/bash

# conda create -n env-streamlit-addaxai python=3.11.11 -y
# conda activate env-streamlit-addaxai
# pip install streamlit
# pip install streamlit-extras --use-pep517
# pip install st-pages
# pip install streamlit-aggrid
# pip install folium
# pip install streamlit-folium
# pip install exifread
# pip install ffmpeg --use-pep517
# pip install gpsphoto --use-pep517
# pip install piexif

# Activate conda environment
source /Users/peter/miniforge3/etc/profile.d/conda.sh
conda activate env-streamlit-addaxai

# find folder of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/frontend"

# Start Streamlit in the background
pkill -f streamlit
streamlit run main.py >> streamlit_log.txt 2>&1 &
