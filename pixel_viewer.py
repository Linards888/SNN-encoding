"""Pixel viewer for 128x128 images.

Open an image, then hover or click on it to read individual pixels.
Run:  python pixel_viewer.py [optional/path/to/image.png]
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

IMAGE_SIZE = 128   # expected width and height in pixels
ZOOM = 5           # each image pixel is drawn as a ZOOM x ZOOM block


def load_image(path):
    """Load an image as RGBA. Returns a PIL Image."""
    return Image.open(path).convert("RGBA")


def read_pixel(image, x, y):
    """Return the (R, G, B, A) value of pixel (x, y). (0, 0) is top-left."""
    return image.getpixel((x, y))


class PixelViewer(tk.Tk):
    def __init__(self, path=None):
        super().__init__()
        self.title("Pixel Viewer")
        self.resizable(False, False)

        self.image = None
        self.photo = None
        self.selected = None  # (x, y) of clicked pixel

        # --- Toolbar ---
        bar = tk.Frame(self, padx=6, pady=6)
        bar.pack(fill="x")
        tk.Button(bar, text="Open image...", command=self.open_dialog).pack(side="left")

        tk.Label(bar, text="   x:").pack(side="left")
        self.x_entry = tk.Entry(bar, width=4)
        self.x_entry.pack(side="left")
        tk.Label(bar, text="y:").pack(side="left")
        self.y_entry = tk.Entry(bar, width=4)
        self.y_entry.pack(side="left")
        tk.Button(bar, text="Read pixel", command=self.read_from_entries).pack(side="left", padx=4)
        self.x_entry.bind("<Return>", lambda e: self.read_from_entries())
        self.y_entry.bind("<Return>", lambda e: self.read_from_entries())

        self.file_label = tk.Label(bar, text="No image loaded", fg="gray")
        self.file_label.pack(side="right")

        # --- Canvas ---
        side = IMAGE_SIZE * ZOOM
        self.canvas = tk.Canvas(self, width=side, height=side, bg="#333",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=6)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<Leave>", lambda e: self.hover_var.set("Hover: -"))
        self.canvas.bind("<Button-1>", self.on_click)

        # --- Info panel ---
        info = tk.Frame(self, padx=6, pady=6)
        info.pack(fill="x")
        self.hover_var = tk.StringVar(value="Hover: -")
        self.sel_var = tk.StringVar(value="Selected: - (click a pixel, then use arrow keys)")
        tk.Label(info, textvariable=self.hover_var, font=("Consolas", 11), anchor="w").pack(fill="x")
        sel_row = tk.Frame(info)
        sel_row.pack(fill="x")
        self.swatch = tk.Label(sel_row, width=3, bg="#333", relief="solid", bd=1)
        self.swatch.pack(side="left", padx=(0, 6))
        tk.Label(sel_row, textvariable=self.sel_var, font=("Consolas", 11), anchor="w").pack(side="left", fill="x")

        for key, d in {"<Left>": (-1, 0), "<Right>": (1, 0), "<Up>": (0, -1), "<Down>": (0, 1)}.items():
            self.bind(key, lambda e, d=d: self.move_selection(*d))

        if path:
            self.open_path(path)

    # ---------- loading ----------
    def open_dialog(self):
        path = filedialog.askopenfilename(
            title="Open 128x128 image",
            filetypes=[("Images", "*.png *.bmp *.jpg *.jpeg *.gif *.tif *.tiff *.ppm *.pgm"),
                       ("All files", "*.*")])
        if path:
            self.open_path(path)

    def open_path(self, path):
        try:
            img = load_image(path)
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))
            return
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            messagebox.showerror(
                "Wrong size",
                f"Image is {img.size[0]}x{img.size[1]} pixels.\n"
                f"Only {IMAGE_SIZE}x{IMAGE_SIZE} images are supported.")
            return

        self.image = img
        big = img.resize((IMAGE_SIZE * ZOOM, IMAGE_SIZE * ZOOM), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(big)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.file_label.config(text=path.replace("\\", "/").split("/")[-1], fg="black")
        self.select(0, 0)

    # ---------- pixel reading ----------
    def canvas_to_pixel(self, event):
        x, y = event.x // ZOOM, event.y // ZOOM
        if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
            return x, y
        return None

    @staticmethod
    def describe(x, y, rgba):
        r, g, b, a = rgba
        return f"({x:3}, {y:3})  R={r:3} G={g:3} B={b:3} A={a:3}  #{r:02X}{g:02X}{b:02X}"

    def on_hover(self, event):
        if self.image is None:
            return
        p = self.canvas_to_pixel(event)
        if p:
            self.hover_var.set("Hover:    " + self.describe(*p, read_pixel(self.image, *p)))

    def on_click(self, event):
        if self.image is None:
            return
        p = self.canvas_to_pixel(event)
        if p:
            self.select(*p)

    def read_from_entries(self):
        if self.image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        try:
            x, y = int(self.x_entry.get()), int(self.y_entry.get())
        except ValueError:
            messagebox.showerror("Bad input", "x and y must be whole numbers.")
            return
        if not (0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE):
            messagebox.showerror("Out of range", f"x and y must be between 0 and {IMAGE_SIZE - 1}.")
            return
        self.select(x, y)

    def move_selection(self, dx, dy):
        if self.image is None or self.selected is None:
            return
        if self.focus_get() in (self.x_entry, self.y_entry):
            return  # let arrow keys move the text cursor in the entry boxes
        x = min(max(self.selected[0] + dx, 0), IMAGE_SIZE - 1)
        y = min(max(self.selected[1] + dy, 0), IMAGE_SIZE - 1)
        self.select(x, y)

    def select(self, x, y):
        self.selected = (x, y)
        rgba = read_pixel(self.image, x, y)
        self.sel_var.set("Selected: " + self.describe(x, y, rgba))
        self.swatch.config(bg="#{:02X}{:02X}{:02X}".format(*rgba[:3]))
        self.x_entry.delete(0, "end"); self.x_entry.insert(0, str(x))
        self.y_entry.delete(0, "end"); self.y_entry.insert(0, str(y))

        # highlight box around the selected pixel
        self.canvas.delete("sel")
        x0, y0 = x * ZOOM, y * ZOOM
        self.canvas.create_rectangle(x0 - 1, y0 - 1, x0 + ZOOM, y0 + ZOOM,
                                     outline="#FF00FF", width=2, tags="sel")


if __name__ == "__main__":
    PixelViewer(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()
