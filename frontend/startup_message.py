import tkinter as tk
from tkinter import ttk

def show_popup():
    root = tk.Tk()
    root.title("Starting Streamlit App")
    root.geometry("350x120")
    root.resizable(False, False)

    label = tk.Label(root, text="Launching Streamlit...\nPlease wait.", font=("Arial", 12))
    label.pack(pady=10)

    progress = ttk.Progressbar(root, length=300, mode='determinate')
    progress.pack(pady=10)

    # Update progress bar over 5 seconds
    for i in range(101):
        progress['value'] = i
        root.update_idletasks()
        root.after(50)  # 50ms * 100 = 5000ms (5 seconds)

    root.destroy()  # Close window after progress completes

if __name__ == "__main__":
    show_popup()
