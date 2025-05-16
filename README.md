## Streamlit version of AddaxAI

*This work in progress.*

Follow the steps below to build the app for development purposes. It assumes you have a working installation of a [conda package manager]([url](https://github.com/conda-forge/miniforge)). 
1. Create a conda environment with the following command
```
conda create -n streamlit python=3.7 -y && conda activate streamlit && pip install streamlit streamlit-extras st-pages streamlit-aggrid folium streamlit-folium exifread ffmpeg gpsphoto piexif
```


1. Install the normal AddaxAI: https://addaxdatascience.com/addaxai/#install
2. After the installation has finished, locate your AddaxAI files
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
3. Open you terminal and `cd` into `.../AddaxAI_files/AddaxAI`
4. Clone this repo inside the AddaxAI folder: `git clone https://github.com/PetervanLunteren/streamlit-AddaxAI.git`
