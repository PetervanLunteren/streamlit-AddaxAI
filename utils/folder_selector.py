"""
AddaxAI Folder Selection Utility

Standalone tkinter-based folder selector that can be called from Streamlit subprocess.
Provides cross-platform directory selection with optional initial directory.
"""

import tkinter as tk
from tkinter import filedialog
import sys
import os

def select_folder(initial_dir=None):
    """
    Open folder selection dialog using tkinter.
    
    Args:
        initial_dir (str, optional): Initial directory to open in dialog
        
    Returns:
        str: Selected folder path or empty string if cancelled
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    
    # Use initial_dir if provided and exists, otherwise use system default
    if initial_dir and os.path.exists(initial_dir):
        folder_path = filedialog.askdirectory(master=root, initialdir=initial_dir)
    else:
        folder_path = filedialog.askdirectory(master=root)
    
    root.destroy()
    return folder_path

if __name__ == "__main__":
    # Accept initial directory as command line argument
    initial_dir = sys.argv[1] if len(sys.argv) > 1 else None
    folder_path = select_folder(initial_dir)
    if folder_path:
        print(folder_path)  # Return the folder path to stdout
    else:
        print("No folder selected", file=sys.stderr)
