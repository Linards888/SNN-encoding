"""Pixel drawer for 128x128 images.

Draw a picture pixel by pixel, then save it as a 128x128 PNG that
pixel_viewer.py can open.
Run:  python pixel_drawer.py [optional/path/to/image.png]

Controls
  Left mouse      draw with the current colour
  Right mouse     erase (paint with the background colour)
  P / E / F / I   pencil / eraser / fill / colour picker
  1-4             brush size
  G               toggle grid
  Ctrl+Z / Ctrl+Y undo / redo
  Ctrl+S          save,  Ctrl+O  open
"""

import sys
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk

IMAGE_SIZE = 128   # width and height of the picture in pixels
ZOOM = 5           # each image pixel is drawn as a ZOOM x ZOOM block
BACKGROUND = (255, 255, 255, 255)
MAX_UNDO = 50

PALETTE = [
    "#000000", "#404040", "#808080", "#C0C0C0", "#FFFFFF",
    "#FF0000", "#FF8000", "#FFFF00", "#00C000", "#00FFFF",
    "#0000FF", "#8000FF", "#FF00FF", "#804000", "#FF8080",
]


def hex_to_rgba(color):
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), 255)


def line_points(x0, y0, x1, y1):
    """All pixels on the line from (x0, y0) to (x1, y1) (Bresenham)."""
    points = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return points
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


class PixelDrawer(tk.Tk):
    def __init__(self, path=None):
        super().__init__()
        self.title("Pixel Drawer")
        self.resizable(False, False)

        self.image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND)
        self.photo = None
        self.path = None
        self.dirty = False
        self.undo_stack, self.redo_stack = [], []

        self.tool = tk.StringVar(value="pencil")
        self.brush = tk.IntVar(value=1)
        self.show_grid = tk.BooleanVar(value=False)
        self.color = "#000000"
        self.last = None         # last pixel of the current stroke
        self.stroke_color = None

        self.build_toolbar()
        self.build_canvas()
        self.build_status()
        self.bind_keys()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        if path:
            self.open_path(path)
        self.redraw()

    # ---------- layout ----------
    def build_toolbar(self):
        bar = tk.Frame(self, padx=6, pady=6)
        bar.pack(fill="x")
        tk.Button(bar, text="New", command=self.new_image).pack(side="left")
        tk.Button(bar, text="Open...", command=self.open_dialog).pack(side="left", padx=2)
        tk.Button(bar, text="Save", command=self.save).pack(side="left")
        tk.Button(bar, text="Save as...", command=self.save_as).pack(side="left", padx=2)
        tk.Button(bar, text="Undo", command=self.undo).pack(side="left", padx=(10, 0))
        tk.Button(bar, text="Redo", command=self.redo).pack(side="left", padx=2)
        tk.Checkbutton(bar, text="Grid", variable=self.show_grid,
                       command=self.redraw).pack(side="left", padx=(10, 0))
        self.file_label = tk.Label(bar, text="untitled", fg="gray")
        self.file_label.pack(side="right")

        tools = tk.Frame(self, padx=6)
        tools.pack(fill="x")
        for label, value in [("Pencil (P)", "pencil"), ("Eraser (E)", "eraser"),
                             ("Fill (F)", "fill"), ("Picker (I)", "picker")]:
            tk.Radiobutton(tools, text=label, value=value, variable=self.tool,
                           indicatoron=False, padx=6, pady=2).pack(side="left", padx=1)
        tk.Label(tools, text="   Brush:").pack(side="left")
        for size in (1, 2, 3, 4):
            tk.Radiobutton(tools, text=str(size), value=size, variable=self.brush,
                           indicatoron=False, width=2, pady=2).pack(side="left", padx=1)

        pal = tk.Frame(self, padx=6, pady=6)
        pal.pack(fill="x")
        self.current_swatch = tk.Label(pal, width=4, height=1, bg=self.color,
                                       relief="sunken", bd=2)
        self.current_swatch.pack(side="left", padx=(0, 8))
        for c in PALETTE:
            sw = tk.Label(pal, width=2, bg=c, relief="raised", bd=1, cursor="hand2")
            sw.pack(side="left", padx=1)
            sw.bind("<Button-1>", lambda e, c=c: self.set_color(c))
        tk.Button(pal, text="More colours...", command=self.choose_color).pack(side="left", padx=8)

    def build_canvas(self):
        side = IMAGE_SIZE * ZOOM
        self.canvas = tk.Canvas(self, width=side, height=side, bg="#333",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=6)
        self.canvas.bind("<Button-1>", lambda e: self.press(e, erase=False))
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<Button-3>", lambda e: self.press(e, erase=True))
        self.canvas.bind("<B3-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-3>", self.release)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<Leave>", lambda e: self.status_var.set(""))

    def build_status(self):
        self.status_var = tk.StringVar()
        tk.Label(self, textvariable=self.status_var, font=("Consolas", 11),
                 anchor="w", padx=6, pady=6).pack(fill="x")

    def bind_keys(self):
        self.bind("p", lambda e: self.tool.set("pencil"))
        self.bind("e", lambda e: self.tool.set("eraser"))
        self.bind("f", lambda e: self.tool.set("fill"))
        self.bind("i", lambda e: self.tool.set("picker"))
        self.bind("g", lambda e: (self.show_grid.set(not self.show_grid.get()), self.redraw()))
        for n in (1, 2, 3, 4):
            self.bind(str(n), lambda e, n=n: self.brush.set(n))
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-y>", lambda e: self.redo())
        self.bind("<Control-s>", lambda e: self.save())
        self.bind("<Control-o>", lambda e: self.open_dialog())

    # ---------- colours ----------
    def set_color(self, color):
        self.color = color.upper()
        self.current_swatch.config(bg=self.color)
        if self.tool.get() in ("eraser", "picker"):
            self.tool.set("pencil")

    def choose_color(self):
        _, chosen = colorchooser.askcolor(color=self.color, title="Pick a colour")
        if chosen:
            self.set_color(chosen)

    # ---------- drawing ----------
    def event_pixel(self, event):
        x, y = event.x // ZOOM, event.y // ZOOM
        if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
            return x, y
        return None

    def push_undo(self):
        self.undo_stack.append(self.image.copy())
        del self.undo_stack[:-MAX_UNDO]
        self.redo_stack.clear()

    def press(self, event, erase):
        p = self.event_pixel(event)
        if p is None:
            return
        tool = "eraser" if erase else self.tool.get()
        if tool == "picker":
            r, g, b, _ = self.image.getpixel(p)
            self.set_color(f"#{r:02X}{g:02X}{b:02X}")
            return
        self.push_undo()
        if tool == "fill":
            ImageDraw.floodfill(self.image, p, hex_to_rgba(self.color))
            self.after_change()
            return
        self.stroke_color = BACKGROUND if tool == "eraser" else hex_to_rgba(self.color)
        self.last = p
        self.paint([p])

    def drag(self, event):
        self.on_hover(event)
        if self.last is None:
            return
        x = min(max(event.x // ZOOM, 0), IMAGE_SIZE - 1)
        y = min(max(event.y // ZOOM, 0), IMAGE_SIZE - 1)
        if (x, y) != self.last:
            self.paint(line_points(*self.last, x, y))
            self.last = (x, y)

    def release(self, event):
        self.last = None

    def paint(self, points):
        size = self.brush.get()
        offset = (size - 1) // 2
        px = self.image.load()
        for cx, cy in points:
            for y in range(cy - offset, cy - offset + size):
                for x in range(cx - offset, cx - offset + size):
                    if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                        px[x, y] = self.stroke_color
        self.after_change()

    def after_change(self):
        self.dirty = True
        self.update_title()
        self.redraw()

    def redraw(self):
        big = self.image.resize((IMAGE_SIZE * ZOOM, IMAGE_SIZE * ZOOM), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(big)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        if self.show_grid.get():
            side = IMAGE_SIZE * ZOOM
            for i in range(1, IMAGE_SIZE):
                color = "#999" if i % 8 == 0 else "#DDD"
                self.canvas.create_line(i * ZOOM, 0, i * ZOOM, side, fill=color)
                self.canvas.create_line(0, i * ZOOM, side, i * ZOOM, fill=color)

    def on_hover(self, event):
        p = self.event_pixel(event)
        if p:
            r, g, b, a = self.image.getpixel(p)
            self.status_var.set(f"({p[0]:3}, {p[1]:3})  R={r:3} G={g:3} B={b:3}  #{r:02X}{g:02X}{b:02X}")

    # ---------- undo / redo ----------
    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.image)
            self.image = self.undo_stack.pop()
            self.after_change()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.image)
            self.image = self.redo_stack.pop()
            self.after_change()

    # ---------- files ----------
    def update_title(self):
        name = self.path.replace("\\", "/").split("/")[-1] if self.path else "untitled"
        self.file_label.config(text=name + (" *" if self.dirty else ""),
                               fg="black" if self.path else "gray")

    def confirm_discard(self):
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("Unsaved changes", "Save your drawing first?")
        if answer is None:
            return False
        return self.save() if answer else True

    def new_image(self):
        if not self.confirm_discard():
            return
        self.image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.path, self.dirty = None, False
        self.update_title()
        self.redraw()

    def open_dialog(self):
        if not self.confirm_discard():
            return
        path = filedialog.askopenfilename(
            title="Open 128x128 image",
            filetypes=[("Images", "*.png *.bmp *.jpg *.jpeg *.gif *.tif *.tiff *.ppm *.pgm"),
                       ("All files", "*.*")])
        if path:
            self.open_path(path)

    def open_path(self, path):
        try:
            img = Image.open(path).convert("RGBA")
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
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.path, self.dirty = path, False
        self.update_title()
        self.redraw()

    def save(self):
        if not self.path:
            return self.save_as()
        try:
            self.image.save(self.path)
        except Exception as exc:
            messagebox.showerror("Could not save", str(exc))
            return False
        self.dirty = False
        self.update_title()
        return True

    def save_as(self):
        path = filedialog.asksaveasfilename(
            title="Save drawing", defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("BMP image", "*.bmp")])
        if not path:
            return False
        self.path = path
        return self.save()

    def on_close(self):
        if self.confirm_discard():
            self.destroy()


if __name__ == "__main__":
    PixelDrawer(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()
