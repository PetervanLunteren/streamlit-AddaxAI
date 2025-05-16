## Streamlit version of AddaxAI

*(work in progress...)*

To build:
1. Make sure you have a conda distribution, e.g., https://github.com/conda-forge/miniforge
2. Create a conda environment
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
*(add packages top the list if you need them for your code)*

3. Install the normal AddaxAI: https://addaxdatascience.com/addaxai/#install
4. Locate your `AddaxAI_files` folder
      <details>
      <summary>Location on Windows</summary>
      <br>
        
      ```r
      ─── C:
          └── 📁 Users
              └── 📁 <username>
                  └── 📁 AddaxAI_files
      ```
      </details>
      
      <details>
      <summary>Location on macOS</summary>
      <br>
        
      ```r
      ─── 📁 Applications
          └── 📁 AddaxAI_files
      ```
      </details>
      
      <details>
      <summary>Location on Linux</summary>
      <br>
        
      ```r
      ─── 📁 home
          └── 📁 <username>
              └── 📁 AddaxAI_files
      ```
      </details>
5. `cd /AddaxAI_files/AddaxAI`
6. `git clone https://github.com/PetervanLunteren/streamlit-AddaxAI.git`
7. The repo should now be cloned here `/AddaxAI_files/AddaxAI/streamlit-AddaxAI`
8. Open the app with
```
conda activate env-streamlit-addaxai
cd AddaxAI_files/AddaxAI
streamlit run main.py >> streamlit_log.txt 2>&1 &
```
8. Code is inside the `/frontend/` and `/backend/` folders
