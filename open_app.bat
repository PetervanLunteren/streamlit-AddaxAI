@echo off

@REM # conda create -n env-streamlit-addaxai python=3.11.11 -y
@REM # conda activate env-streamlit-addaxai
@REM # pip install streamlit
@REM # pip install streamlit-extras --use-pep517
@REM # pip install st-pages
@REM # pip install streamlit-aggrid
@REM # pip install folium
@REM # pip install streamlit-folium
@REM # pip install exifread
@REM # pip install ffmpeg --use-pep517
@REM # pip install gpsphoto --use-pep517
@REM # pip install piexif


REM attempt to locate conda.bat
FOR /F "delims=" %%I IN ('where conda.bat 2^>nul') DO (
    SET "CONDA_BAT=%%I"
    GOTO :found
)

REM conda not located
ECHO conda.bat not found in PATH. Please ensure Miniforge or Anaconda is installed and added to PATH.
EXIT /B 1

REM conda located
:found
echo Found conda.bat at "%CONDA_BAT%"
CALL "%CONDA_BAT%" activate env-streamlit-addaxai

REM change to script directory
SET SCRIPT_DIR=%~dp0
CD /D "%SCRIPT_DIR%frontend"
echo cd to "%SCRIPT_DIR%frontend"

REM kill any previous streamlit processes
taskkill /F /IM python.exe /FI "WINDOWTITLE eq streamlit*" >nul 2>&1

REM start streamlit in background and log output
start "" /B cmd /C "streamlit run main.py >> streamlit_log.txt 2>&1"

pause