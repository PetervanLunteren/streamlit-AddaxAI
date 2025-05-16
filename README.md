# Streamlit version of AddaxAI
*(work in progress...)*

## How to build
1. Make sure you have a conda distribution, e.g., https://github.com/conda-forge/miniforge
2. Create a conda environment (add packages to this list if you need extra for your code)
```
conda create -n env-streamlit-addaxai python=3.11.11 -y
conda activate env-streamlit-addaxai
pip install streamlit
pip install streamlit-extras
pip install st-pages
pip install streamlit-aggrid
pip install folium
pip install streamlit-folium
pip install exifread
pip install ffmpeg
pip install gpsphoto
pip install piexif
```
3. Install the normal AddaxAI: https://addaxdatascience.com/addaxai/#install
4. Change directory and clone repo
```
cd /Applications/AddaxAI_files/AddaxAI             (macOS)
cd ~/AddaxAI_files/AddaxAI                         (Linux)

git clone https://github.com/PetervanLunteren/streamlit-AddaxAI.git
```

7. The repo should now be cloned into `/AddaxAI_files/AddaxAI/streamlit-AddaxAI`
8. Open the app with
```
conda activate env-streamlit-addaxai
cd streamlit-AddaxAI/frontend
streamlit run main.py >> streamlit_log.txt 2>&1 &
```
Now you can push changes back to this repo without breaking the actual working app. The code is inside the `/frontend/` and `/backend/` folders.

## General ideas
This streamlit app will eventually replace the [existing tkinter app](https://github.com/PetervanLunteren/addaxai). The code has become to complex and is in need of a fresh start. This streamlit version will start from scratch and will have multiple pages/tools. It will become more of a management platform where people can add metadata around their deployments and locations, so that depth estimation, maps, charts, camtrapDP exports, etc will become available. 

<img width="268" alt="Screenshot 2025-05-16 at 20 07 52" src="https://github.com/user-attachments/assets/b42f8c4e-f35b-48ca-b050-ea6203f122c2" />


The tools are in the sidebar, but this is just a mockup. If you have other thoughts, let me know! I've added some notes to each tool. 





