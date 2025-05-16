## Streamlit version of AddaxAI

*This work in progress.*

Follow the steps below to build the app for development purposes. It assumes you have a working installation of a [conda package manager](https://github.com/conda-forge/miniforge), and it is on your PATH. 
1. Create a conda environment with the following commands
```
conda create -n env-streamlit-addaxai python=3.11.11 -y
conda activate env-streamlit-addaxai
pip install streamlit
pip install streamlit-extras --use-pep517
pip install st-pages
pip install streamlit-aggrid
pip install folium
pip install streamlit-folium
pip install exifread
pip install ffmpeg --use-pep517
pip install gpsphoto --use-pep517
pip install piexif
```
2. Install the normal AddaxAI: https://addaxdatascience.com/addaxai/#install
3. After the installation has finished, locate your AddaxAI files
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
4. Open you terminal and `cd` into `.../AddaxAI_files/AddaxAI`
5. Clone this repo inside the AddaxAI folder: `git clone https://github.com/PetervanLunteren/streamlit-AddaxAI.git`
6. Open the app with `open_app.command` (macOS/Linux) or `open_app.bat` (Windows)
7. Code is inside the `/frontend/` and `/backend/` folders
