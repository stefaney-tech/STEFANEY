# ANONYMIZER - Face Anonymization with a Tkinter GUI
# How it works: Haar Cascade finds the faces, the user marks who agreed to be shown (consent), and every other face is blurred, pixelated,or covered with a smiley.
#Topics used:c. Haar Cascade Classifier (face detection), a. Image Operations (crop, blur, draw, show, save), d. Image Analysis and Transformation (grayscale, resize, pixelation)
# GUI: Tkinter (buttons, sliders, dropdown). Pillow is used to show OpenCV images inside the Tkinter window.

# ---IMPORTS---
import cv2                      # OpenCV: detection and image processing
import os                       # for folders and file names
from datetime import datetime   # for the timestamp in saved file names

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk  # needed to display OpenCV images in Tkinter

# ---ADJUSTABLE SETTINGS---
# Starting values for the detector and the effects. The sliders in the GUI change these while the program is running.
DEFAULT_SETTINGS = {
    "scale_factor": 1.1,     # how much the image is shrunk at each detection step
    "min_neighbors": 8,      # higher = fewer boxes but fewer false detections
    "min_size": (40, 40),    # ignore boxes smaller than this
    "max_size": (80, 80),    # ignore boxes bigger than this
    "overlap_limit": 0.7,    # if 70% of a box is inside a bigger box, it is a duplicate
    "blur_padding": 0.15,    # cover a bit more than the box (hair, ears)
    "pixel_blocks": 10,      # number of blocks across the face when pixelating
}

MAX_DISPLAY_HEIGHT = 900     # photos taller than this are shrunk to fit the screen
UI_FONT = cv2.FONT_HERSHEY_SIMPLEX   # font for the labels drawn on the image

# Names shown in the dropdown, and short versions used in saved file names
EFFECT_NAMES = ["Gaussian Blur", "Pixelation", "Emoji"]
EFFECT_SHORTS = ["blur", "pixel", "emoji"]

# ---HELPER FUNCTIONS---
def box_area(box):
    # box is (x, y, w, h), so the area is width * height
    return box[2] * box[3]

def remove_overlapping_boxes(boxes, limit):
    # Haar sometimes puts two boxes on the same face. Sort the boxes biggest first, then drop any box that is mostly inside a box we already kept.
    kept = []
    for (x, y, w, h) in sorted(boxes, key=box_area, reverse=True):
        is_duplicate = False
        for (kx, ky, kw, kh) in kept:
            # width and height of the part where the two boxes overlap
            ow = min(x + w, kx + kw) - max(x, kx)
            oh = min(y + h, ky + kh) - max(y, ky)
            # if the overlap covers more than "limit" of this box, it is a duplicate
            if ow > 0 and oh > 0 and (ow * oh) / (w * h) > limit:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append((x, y, w, h))
    return kept

def draw_smiley(img, cx, cy, r):
    # OpenCV can't print real emoji, so the smiley is drawn with shapes. (cx, cy) is the center and r is the radius.yellow filled face, then a darker outline
    cv2.circle(img, (cx, cy), r, (0, 215, 255), -1, cv2.LINE_AA) #cv2.LINE_AA is anti-aliasing, makes the circle smooth
    cv2.circle(img, (cx, cy), r, (0, 140, 200), max(2, r // 15), cv2.LINE_AA)
    # two black eyes, a little above the center
    for eye_x in (cx - r // 3, cx + r // 3):
        cv2.circle(img, (eye_x, cy - r // 5), max(2, r // 9), (0, 0, 0), -1, cv2.LINE_AA)
    # smile = the lower arc of an ellipse (from 20 to 160 degrees)
    cv2.ellipse(img, (cx, cy + r // 5), (r // 2, r // 3), 0, 20, 160,
                (0, 0, 0), max(2, r // 12), cv2.LINE_AA)

def make_file_path(image_path, folder, label):
    # Makes a file name like output/photo_blur_20250101_120000.jpg, (photo name + effect + date and time).
    photo_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = folder + "/" + photo_name + "_" + label + "_" + timestamp

    # if the name already exists, add _2, _3, ... so nothing is overwritten
    path = base + ".jpg"
    number = 2
    while os.path.exists(path):
        path = base + "_" + str(number) + ".jpg"
        number += 1
    return path

# ---THE GUI APPLICATION---
# One class holds the whole program. Values that would normally be global variables (faces, consented, effect, ...) are stored on self.

class AnonymizerApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Anonymizer for images")
        self.root.geometry("1200x820")

        # settings and image data
        self.settings = dict(DEFAULT_SETTINGS)   # copy so the defaults stay unchanged
        self.image_path = None
        self.original_image = None      # the loaded photo (BGR)
        self.faces = []                 # list of (x, y, w, h), one per face
        self.consented = []             # one True/False per face (True = stays visible)
        self.effect = 1                 # 1 blur, 2 pixelate, 3 emoji
        self.show_labels = True
        self.preview = False            # False = box view, True = anonymized view
        self.delete_mode = False
        self.removed_faces = []         # deleted boxes, so Undo can bring them back

        # keep a reference to the image, otherwise Tkinter deletes it
        self.tk_image = None
        # the shown image is smaller than the real one, so we save the scale to convert a click on the screen to a point on the real image
        self.display_scale = 1.0
        self.display_offset = (0, 0)

        self.build_ui()

    # ---UI LAYOUT---
    # Left column: buttons, sliders, dropdown. Right column: the image.
    def build_ui(self):
        left = ttk.Frame(self.root, padding=10)
        left.pack(side="left", fill="y")

        right = ttk.Frame(self.root, padding=10)
        right.pack(side="right", expand=True, fill="both")

        # ---FILE---
        ttk.Label(left, text="File", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 4))
        ttk.Button(left, text="Open Image...", command=self.on_open).pack(fill="x", pady=2)
        ttk.Button(left, text="Reload", command=self.on_reload).pack(fill="x", pady=2)

        # ---DETECTION---
        ttk.Label(left, text="Detection", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(14, 4))
        ttk.Button(left, text="Re-detect Faces", command=self.redetect).pack(fill="x", pady=2)

        # variables connected to the sliders (they hold the current slider values)
        self.sv_scale = tk.DoubleVar(value=self.settings["scale_factor"])
        self.sv_neighbors = tk.IntVar(value=self.settings["min_neighbors"])
        self.sv_min_size = tk.IntVar(value=self.settings["min_size"][0])
        self.sv_max_size = tk.IntVar(value=self.settings["max_size"][0])
        self.sv_padding = tk.DoubleVar(value=self.settings["blur_padding"])
        self.sv_blocks = tk.IntVar(value=self.settings["pixel_blocks"])

        # one slider per setting (label, variable, min, max, step, function to call)
        self._slider(left, "Scale factor", self.sv_scale, 1.01, 1.5, 0.01, self.apply_settings)
        self._slider(left, "Min neighbors", self.sv_neighbors, 1, 15, 1, self.apply_settings)
        self._slider(left, "Min size (px)", self.sv_min_size, 10, 200, 1, self.apply_settings)
        self._slider(left, "Max size (px)", self.sv_max_size, 40, 400, 1, self.apply_settings)
        self._slider(left, "Blur padding", self.sv_padding, 0.0, 0.5, 0.01, self.apply_settings)
        self._slider(left, "Pixel blocks", self.sv_blocks, 2, 40, 1, self.apply_settings)

        # --- EFFECT ---
        # dropdown to choose blur, pixelation or emoji
        ttk.Label(left, text="Effect", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(14, 4))
        self.effect_var = tk.StringVar(value=EFFECT_NAMES[0])
        effect_box = ttk.Combobox(left, textvariable=self.effect_var,
                                  values=EFFECT_NAMES, state="readonly")
        effect_box.pack(fill="x")
        effect_box.bind("<<ComboboxSelected>>", self.on_effect_change)

        # ---MODE---
        ttk.Label(left, text="Mode", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(14, 4))
        ttk.Button(left, text="Show Boxes", command=self.show_box_view).pack(fill="x", pady=2)
        ttk.Button(left, text="Preview Result", command=self.show_preview).pack(fill="x", pady=2)

        self.delete_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Delete mode (click to remove box)",
                        variable=self.delete_var,
                        command=self.on_delete_mode_change).pack(anchor="w", pady=(8, 0))
        
        self.labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Show labels",
                        variable=self.labels_var,
                        command=self.on_labels_change).pack(anchor="w", pady=(4, 0))

        ttk.Button(left, text="Undo Delete", command=self.on_undo).pack(fill="x", pady=2)

        # --- OUTPUT ---
        ttk.Label(left, text="Output", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(14, 4))
        ttk.Button(left, text="Save Anonymized Image", command=self.on_save).pack(fill="x", pady=2)

        # --- STATUS BAR ---
        # a line at the bottom that tells the user what just happened
        self.status = tk.StringVar(value="Ready. Open an image to begin.")
        status_bar = ttk.Label(self.root, textvariable=self.status,
                               relief="sunken", anchor="w", padding=(8, 4))
        status_bar.pack(side="bottom", fill="x")

        # --- IMAGE AREA ---
        self.image_label = ttk.Label(right, text="No image loaded", anchor="center")
        self.image_label.pack(expand=True, fill="both")
        # connect mouse clicks on the image to our functions: Button-1 = left click, Button-3 = right click
        self.image_label.bind("<Button-1>", self.on_left_click)
        self.image_label.bind("<Button-3>", self.on_right_click)

    def _slider(self, parent, label, variable, lo, hi, step, callback):
        # Makes one row with a label and a slider, so the same code is not repeated for every setting.
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=14).pack(side="left")
        s = ttk.Scale(row, from_=lo, to=hi, variable=variable,
                      orient="horizontal", command=lambda _: callback())
        s.pack(side="left", fill="x", expand=True)

    # --- LOADING & DETECTION ---
    def on_open(self):
        # opens a file dialog so the user can pick a photo
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if not path:
            return   # user cancelled
        self.load_from_path(path)

    def on_reload(self):
        # loads the same photo again (resets all boxes and consent)
        if self.image_path:
            self.load_from_path(self.image_path)

    def load_from_path(self, path):
        img = cv2.imread(path)
        # imread returns None if the file could not be read
        if img is None:
            messagebox.showerror("Error", "Could not load that file.")
            return

        # shrink the photo if it is taller than MAX_DISPLAY_HEIGHT (keeps the same shape)
        h, w = img.shape[:2]
        if h > MAX_DISPLAY_HEIGHT:
            scale = MAX_DISPLAY_HEIGHT / h
            img = cv2.resize(img, (int(w * scale), MAX_DISPLAY_HEIGHT))

        # reset everything for the new photo
        self.image_path = path
        self.original_image = img
        self.faces = []
        self.consented = []
        self.removed_faces = []

        self.redetect()   # runs Haar and shows the result
        self.status.set(f"Loaded: {os.path.basename(path)}")

    def redetect(self):
        if self.original_image is None:
            return

        # load the Haar face classifier that comes with OpenCV
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        # Haar works on grayscale images
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # find the faces using the current slider settings
        found = cascade.detectMultiScale(
            gray,
            scaleFactor=self.settings["scale_factor"],
            minNeighbors=self.settings["min_neighbors"],
            minSize=self.settings["min_size"],
            maxSize=self.settings["max_size"])

        # change to plain Python numbers so the list can be edited later
        boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in found]
        # remove duplicate/nested boxes
        boxes = remove_overlapping_boxes(boxes, self.settings["overlap_limit"])

        self.faces = boxes
        self.consented = [False] * len(boxes)   # everyone starts hidden (privacy first)
        self.removed_faces = []

        # printed in the console to help choose good min/max size values
        print("Image size:", self.original_image.shape[1], "x", self.original_image.shape[0])
        print("Haar boxes:", len(boxes))
        print("Box widths:", sorted([f[2] for f in boxes]))

        self.status.set(f"Detected {len(boxes)} face(s). Click to mark consent.")
        self.show_box_view()

    def apply_settings(self):
        # Runs every time a slider moves: copies the slider values into self.settings.
        self.settings["scale_factor"] = float(self.sv_scale.get())
        self.settings["min_neighbors"] = int(self.sv_neighbors.get())
        self.settings["min_size"] = (int(self.sv_min_size.get()), int(self.sv_min_size.get()))
        self.settings["max_size"] = (int(self.sv_max_size.get()), int(self.sv_max_size.get()))
        self.settings["blur_padding"] = float(self.sv_padding.get())
        self.settings["pixel_blocks"] = int(self.sv_blocks.get())

        # if the preview is showing, redraw it so the change appears right away
        if self.preview and self.original_image is not None:
            self.show_preview()

    # ---VIEW SWITCHING---
    # Two views: the box view (to mark consent) and the preview (final result).
    def show_box_view(self):
        if self.original_image is None:
            return
        self.preview = False
        display = self.original_image.copy()   # draw on a copy, not the original

        # green box = consented (stays visible), red box = will be anonymized
        for i, (x, y, w, h) in enumerate(self.faces):
            color = (0, 255, 0) if self.consented[i] else (0, 0, 255)
            label = "CONSENTED" if self.consented[i] else "ANONYMIZED"
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            if self.show_labels:
                cv2.putText(display, label, (x, max(15, y - 8)),
                            UI_FONT, 0.4, color, 2, cv2.LINE_AA)

        self.update_display(display)

    def show_preview(self):
        # shows what the saved picture will look like
        if self.original_image is None:
            return
        self.preview = True
        result = self.anonymize_image(self.original_image)

        # optional labels on top of the preview
        if self.show_labels:
            for i, (x, y, w, h) in enumerate(self.faces):
                color = (0, 255, 0) if self.consented[i] else (0, 0, 255)
                label = "CONSENTED" if self.consented[i] else "ANONYMIZED"
                cv2.putText(result, label, (x, max(15, y - 8)),
                            UI_FONT, 0.5, color, 2, cv2.LINE_AA)

        self.update_display(result)

    def on_effect_change(self, event=None):
        # the dropdown gives a name, so find its position in the list and add 1 to get the effect number (1, 2 or 3)
        name = self.effect_var.get()
        self.effect = EFFECT_NAMES.index(name) + 1
        print("Effect:", self.effect)
        if self.preview:
            self.show_preview()

    # ---ANONYMIZE---
    # Goes through every face. If the person did not consent, the face area is blurred, pixelated, or covered with a smiley. Returns a NEW image, so the original photo stays unchanged.
    def anonymize_image(self, original):
        result = original.copy()
        img_h, img_w = result.shape[:2]

        for i, (x, y, w, h) in enumerate(self.faces):
            if self.consented[i]:
                continue   # this person agreed, so leave the face alone

            # make the area a little bigger than the box (to cover hair and ears)
            pad_x = int(w * self.settings["blur_padding"])
            pad_y = int(h * self.settings["blur_padding"])

            # keep the area inside the photo so the slicing does not break
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_w, x + w + pad_x)
            y2 = min(img_h, y + h + pad_y)

            if x2 <= x1 or y2 <= y1:
                continue   # the box is outside the photo, skip it

            # crop the face area (rows are y, columns are x)
            region = result[y1:y2, x1:x2]
            crop_h, crop_w = region.shape[:2]

            if self.effect == 1:
                # Effect 1: Gaussian blur. The kernel size must be an odd number.
                k = (crop_w // 2) * 2 + 1
                result[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)

            elif self.effect == 2:
                # Effect 2: pixelation. Shrink the face to a tiny image,
                # then enlarge it back. INTER_NEAREST makes the square blocks.
                tiny = cv2.resize(region,
                                  (self.settings["pixel_blocks"],
                                   self.settings["pixel_blocks"]),
                                  interpolation=cv2.INTER_LINEAR)
                result[y1:y2, x1:x2] = cv2.resize(tiny, (crop_w, crop_h),
                                                  interpolation=cv2.INTER_NEAREST)
            else:
                # Effect 3: emoji. Draw a smiley that covers the face.
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                r = int(max(crop_w, crop_h) * 0.6)
                draw_smiley(result, cx, cy, r)

        return result

    # ---IMAGE DISPLAY---
    # Tkinter can't show an OpenCV image directly, so the steps are:BGR -> RGB (OpenCV uses BGR) -> PIL image -> ImageTk image. Also save the scale so clicks can be mapped back to the real image.
    def update_display(self, cv_image):
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # fit the picture into the available space and keep its shape
        max_w = max(400, self.image_label.winfo_width() - 20)
        max_h = max(300, self.image_label.winfo_height() - 20)
        pil.thumbnail((max_w, max_h), Image.LANCZOS)

        # scale = real width / displayed width (used in the click functions)
        img_h, img_w = cv_image.shape[:2]
        self.display_scale = img_w / pil.width
        self.display_offset = (0, 0)   # the label centers the image, handled in _label_to_image_xy

        self.tk_image = ImageTk.PhotoImage(pil)
        self.image_label.configure(image=self.tk_image, text="")

    # ---CLICK HANDLING---
    # Tkinter gives the click position on the label. We convert it to a position on the real image, then look for a face box at that point.
    def _label_to_image_xy(self, lx, ly):
        if self.tk_image is None:
            return None, None

        label_w = self.image_label.winfo_width()
        label_h = self.image_label.winfo_height()
        img_w = self.tk_image.width()
        img_h = self.tk_image.height()

        # the picture is centered in the label, so there is empty space around it
        offset_x = (label_w - img_w) // 2
        offset_y = (label_h - img_h) // 2

        px = lx - offset_x
        py = ly - offset_y

        # the click was outside the picture
        if px < 0 or py < 0 or px >= img_w or py >= img_h:
            return None, None

        # scale back up to the real image size
        ix = int(px * self.display_scale)
        iy = int(py * self.display_scale)
        return ix, iy

    def find_face_at(self, mx, my):
        # Finds the box under the click. If boxes overlap, pick the smallest one, because it is usually the tight box on the face.Returns -1 if the click is not inside any box.
        best = -1
        best_area = 0
        for i, (x, y, w, h) in enumerate(self.faces):
            if x <= mx <= x + w and y <= my <= y + h:
                area = w * h
                if best == -1 or area < best_area:
                    best = i
                    best_area = area
        return best

    def on_left_click(self, event):
        # clicks only work in the box view, not in the preview
        if self.original_image is None or self.preview:
            return

        ix, iy = self._label_to_image_xy(event.x, event.y)
        if ix is None:
            return

        i = self.find_face_at(ix, iy)
        if i == -1:
            return   # the click was not on a box

        if self.delete_var.get():
            # delete mode: remove the box but save it so Undo can restore it
            self.removed_faces.append((self.faces[i], self.consented[i]))
            self.faces.pop(i)
            self.consented.pop(i)
            self.status.set(f"Box deleted. {len(self.faces)} left.")
        else:
            # normal mode: switch between consented and not consented
            self.consented[i] = not self.consented[i]
            self.status.set(f"Face {i} consented: {self.consented[i]}")

        self.show_box_view()

    def on_right_click(self, event):
        # right click adds a box for a face that Haar missed
        if self.original_image is None or self.preview:
            return

        ix, iy = self._label_to_image_xy(event.x, event.y)
        if ix is None:
            return

        # use the average width of the existing boxes, or 80 if there are none
        if len(self.faces) > 0:
            size = int(sum(f[2] for f in self.faces) / len(self.faces))
        else:
            size = 80

        # center the new box on the click
        new_x = ix - size // 2
        new_y = iy - size // 2
        self.faces.append((new_x, new_y, size, size))
        self.consented.append(False)   # starts hidden (privacy first)
        self.status.set(f"Added manual box at ({new_x}, {new_y}).")
        self.show_box_view()

    def on_delete_mode_change(self):
        # runs when the delete mode checkbox is ticked or unticked
        self.delete_mode = self.delete_var.get()
        print("Delete mode:", self.delete_mode)
        
    def on_labels_change(self):
        # runs when the show labels checkbox is ticked or unticked
        self.show_labels = self.labels_var.get()
        # redraw so the change is visible right away
        if self.preview:
            self.show_preview()
        else:
            self.show_box_view()

    def on_undo(self):
        # brings back the most recently deleted box
        if not self.removed_faces:
            self.status.set("Nothing to undo.")
            return
        box, was_consented = self.removed_faces.pop()
        self.faces.append(box)
        self.consented.append(was_consented)
        self.status.set(f"Restored a box. {len(self.faces)} total.")
        self.show_box_view()

    # ---SAVE---
    # Saves a clean anonymized image (no boxes, no labels). The user chooses where to save, with a suggested file name that includes the photo name, effect, date and time.
    def on_save(self):
        if self.original_image is None:
            messagebox.showwarning("Wait", "Load an image first.")
            return

        clean = self.anonymize_image(self.original_image)

        os.makedirs("output", exist_ok=True)   # make the output folder if it is missing
        effect_short = EFFECT_SHORTS[self.effect - 1]
        suggested = make_file_path(self.image_path, "output", effect_short)

        chosen = filedialog.asksaveasfilename(
            title="Save anonymized image",
            initialdir="output",
            initialfile=os.path.basename(suggested),
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])

        if not chosen:
            return   # user cancelled

        if cv2.imwrite(chosen, clean):
            hidden = self.consented.count(False)
            self.status.set(f"Saved {os.path.basename(chosen)} - {hidden} anonymized.")
            print("Saved", chosen, "-", hidden, "of", len(self.faces), "faces anonymized.")
            print("REMINDER: check the saved file that no stranger's face is visible.")
        else:
            messagebox.showerror("Error", "Could not save the image.")


# --- ENTRY POINT ---
# runs the program only when this file is executed directly
if __name__ == "__main__":
    root = tk.Tk()
    app = AnonymizerApp(root)
    root.mainloop()