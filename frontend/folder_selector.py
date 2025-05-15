import tkinter as tk
from tkinter import filedialog
import sys

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path

if __name__ == "__main__":
    folder_path = select_folder()
    if folder_path:
        print(folder_path)  # Return the folder path to stdout
    else:
        print("No folder selected", file=sys.stderr)
