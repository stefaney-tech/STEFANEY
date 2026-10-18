import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from datetime import datetime


class AnonymizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Anonymizer for Images")
        self.root.geometry("1200x800")

        # ---state that the whole app shares---
        self.original_image = None       # the loaded BGR image (numpy array)
        self.display_image = None        # the currently displayed image (copy)
        self.faces = []                  # list of (x, y, w, h) boxes
        self.consented = []              # list of True/False, one per face
        self.effect = 1                  # 1 = blur, 2 = pixelate, 3 = emoji
        self.show_labels = True
        self.image_path = None
        self.tk_image = None             # keep a reference or Tkinter forgets it

        # ---settings (same names as main.py)---
        self.scale_factor = 1.1
        self.min_neighbors = 8
        self.min_size = 40
        self.max_size = 80
        self.blur_padding = 0.15
        self.pixel_blocks = 10

        self.build_layout()

    def build_layout(self):
        # Left panel: controls. Right panel: image display.
        left = ttk.Frame(self.root, padding=10)
        left.pack(side="left", fill="y")

        right = ttk.Frame(self.root, padding=10)
        right.pack(side="right", expand=True, fill="both")

        ttk.Label(left, text="Controls", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))

        # Placeholder buttons (they don't do anything yet)
        ttk.Button(left, text="Open Image...").pack(fill="x", pady=4)
        ttk.Button(left, text="Anonymize").pack(fill="x", pady=4)
        ttk.Button(left, text="Save Result").pack(fill="x", pady=4)

        # Image display area
        self.image_label = ttk.Label(right, text="No image loaded", anchor="center")
        self.image_label.pack(expand=True, fill="both")


if __name__ == "__main__":
    root = tk.Tk()
    app = AnonymizerApp(root)
    root.mainloop()