"""Seeded shape generator for 128x128 images.

Every picture comes from a seed. The same seed always gives the exact same
picture, on any computer and any Python version, so a seed is a tiny
"recipe" you can write down, share, and re-create later. The PNGs it makes
open in pixel_drawer.py and pixel_viewer.py.

Run:
  python shape_seeds.py                          one random shape, saved and opened
  python shape_seeds.py --seed 42                one picture from seed 42
  python shape_seeds.py --seed "red banana"      text seeds work too
  python shape_seeds.py --seed 42 --count 50     50 pictures from one master seed
  python shape_seeds.py --count 20 --shape triangle --style outline
  python shape_seeds.py --seed 0x9f3c51a2e07b44d1 re-make a picture from its file name

Pictures go in the shapes/ folder as shape_001.png, shape_002.png, ...
Each run prints the seed it used and adds it to shapes/seeds.csv, so any
picture you like can be made again with --seed.

How the randomness works (borrowed from how big projects do it)
  1. Where a seed comes from
     No seed given -> 64 random bits from the operating system (secrets module),
     like NumPy's default_rng() does. Integer seed -> used as-is (64 bits),
     like a Minecraft world seed. Text seed -> hashed with SHA-256, the same
     idea as NumPy's SeedSequence and Minecraft's text seeds: any string
     becomes a well-mixed number.
  2. One seed, many pictures
     A batch has one master seed. Each picture gets its own child seed made by
     mixing the master seed with the picture's number (SplitMix64), like
     NumPy's SeedSequence.spawn() and Minecraft's per-feature salts. So
     picture #7 of seed 42 is always the same, even if you ask for more or
     fewer pictures, and every child seed can be re-run on its own.
  3. The generator itself
     Xoroshiro128++ (Blackman & Vigna), the generator modern Minecraft uses
     for world generation, with its 128-bit state filled from the seed by
     SplitMix64. It is written out here instead of using Python's `random`
     module so results never change if Python changes its internals.
"""

import argparse
import csv
import hashlib
import math
import os
import secrets
import sys

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Pillow is not installed for this Python. Run:  python -m pip install pillow")

IMAGE_SIZE = 128
BACKGROUND = (255, 255, 255, 255)
INK = (0, 0, 0, 255)
SHAPES = ["circle", "ellipse", "square", "rectangle", "triangle",
          "pentagon", "hexagon", "star"]
STYLES = ["outline", "filled"]

MASK64 = (1 << 64) - 1


# ---------- random number generation ----------
def splitmix64(x):
    """Mix a 64-bit number into a well-scrambled 64-bit number."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & MASK64


class Xoroshiro128pp:
    """Xoroshiro128++ pseudo-random generator seeded from a 64-bit number."""

    def __init__(self, seed):
        seed &= MASK64
        self.s0 = splitmix64(seed)
        self.s1 = splitmix64((seed + 0x9E3779B97F4A7C15) & MASK64)
        if self.s0 == 0 and self.s1 == 0:   # all-zero state never changes
            self.s1 = 1

    def next_u64(self):
        s0, s1 = self.s0, self.s1
        result = (rotl((s0 + s1) & MASK64, 17) + s0) & MASK64
        s1 ^= s0
        self.s0 = rotl(s0, 49) ^ s1 ^ ((s1 << 21) & MASK64)
        self.s1 = rotl(s1, 28)
        return result

    def random(self):
        """Float in [0, 1)."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def uniform(self, a, b):
        return a + (b - a) * self.random()

    def randint(self, a, b):
        """Integer in [a, b], both ends included, with no bias."""
        n = b - a + 1
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            x = self.next_u64()
            if x < limit:
                return a + x % n

    def choice(self, items):
        return items[self.randint(0, len(items) - 1)]


# ---------- seeds ----------
def parse_seed(text):
    """Turn what the user typed into a 64-bit seed."""
    t = text.strip()
    try:
        if t.lower().startswith("0x"):
            return int(t, 16) & MASK64
        return int(t) & MASK64
    except ValueError:
        digest = hashlib.sha256(t.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little")


def child_seed(master, index):
    """Seed for picture number `index` of a batch."""
    return splitmix64(master ^ splitmix64(index + 1))


def seed_name(seed):
    return f"0x{seed:016x}"


# ---------- shapes ----------
def polygon_points(cx, cy, radii, angles):
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for r, a in zip(radii, angles)]


def rotated_box(cx, cy, w, h, angle):
    c, s = math.cos(angle), math.sin(angle)
    pts = []
    for dx, dy in [(-w, -h), (w, -h), (w, h), (-w, h)]:
        pts.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    return pts


def make_shape(rng, shape):
    """Return (kind, data) where kind is 'circle' or 'polygon'."""
    r = rng.uniform(14, 54)                 # outer radius, keeps it inside the image
    margin = r + 3
    cx = rng.uniform(margin, IMAGE_SIZE - 1 - margin)
    cy = rng.uniform(margin, IMAGE_SIZE - 1 - margin)
    turn = rng.uniform(0, 2 * math.pi)

    if shape == "circle":
        return "circle", (cx - r, cy - r, cx + r, cy + r)
    if shape == "ellipse":
        ry = r * rng.uniform(0.35, 0.8)
        pts = []
        for i in range(96):
            t = 2 * math.pi * i / 96
            x, y = r * math.cos(t), ry * math.sin(t)
            pts.append((cx + x * math.cos(turn) - y * math.sin(turn),
                        cy + x * math.sin(turn) + y * math.cos(turn)))
        return "polygon", pts
    if shape == "square":
        half = r / math.sqrt(2)
        return "polygon", rotated_box(cx, cy, half, half, turn)
    if shape == "rectangle":
        ratio = rng.uniform(0.3, 0.75)
        w = r / math.sqrt(1 + ratio * ratio)
        return "polygon", rotated_box(cx, cy, w, w * ratio, turn)
    if shape == "triangle":
        # corners on a circle, each gap at least 60 degrees so no slivers
        spare = [rng.random() for _ in range(3)]
        total = sum(spare)
        gaps = [math.pi / 3 + math.pi * s / total for s in spare]
        angles = [turn, turn + gaps[0], turn + gaps[0] + gaps[1]]
        return "polygon", polygon_points(cx, cy, [r] * 3, angles)
    if shape in ("pentagon", "hexagon"):
        n = 5 if shape == "pentagon" else 6
        angles = [turn + 2 * math.pi * i / n for i in range(n)]
        return "polygon", polygon_points(cx, cy, [r] * n, angles)
    if shape == "star":
        points = rng.randint(5, 7)
        inner = r * rng.uniform(0.38, 0.55)
        angles = [turn + math.pi * i / points for i in range(2 * points)]
        radii = [r if i % 2 == 0 else inner for i in range(2 * points)]
        return "polygon", polygon_points(cx, cy, radii, angles)
    raise ValueError(f"unknown shape: {shape}")


def random_colour(rng):
    """A random colour that is dark enough to see on white."""
    while True:
        c = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        if 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] < 170:
            return c + (255,)


def render(seed, shape=None, style=None, colour=False):
    """Draw the picture for one seed. Returns (image, shape, style)."""
    rng = Xoroshiro128pp(seed)
    # Always draw the same random numbers in the same order, so a fixed
    # --shape / --style only changes that part of the picture.
    picked_shape = rng.choice(SHAPES)
    picked_style = rng.choice(STYLES)
    line_width = rng.randint(1, 3)
    ink = random_colour(rng) if colour else INK
    shape = shape or picked_shape
    style = style or picked_style

    image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(image)
    kind, data = make_shape(rng, shape)
    if kind == "polygon":
        data = [(round(x), round(y)) for x, y in data]
        if style == "filled":
            draw.polygon(data, fill=ink)
        else:
            draw.polygon(data, outline=ink, width=line_width)
    else:
        box = [round(v) for v in data]
        if style == "filled":
            draw.ellipse(box, fill=ink)
        else:
            draw.ellipse(box, outline=ink, width=line_width)
    return image, shape, style


# ---------- command line ----------
def main(argv=None):
    p = argparse.ArgumentParser(description="Make 128x128 shape pictures from seeds.")
    p.add_argument("--seed", help="number, 0x-hex number, or any text (default: random)")
    p.add_argument("--count", type=int, default=1, help="how many pictures (default 1)")
    p.add_argument("--shape", choices=SHAPES, help="always use this shape")
    p.add_argument("--style", choices=STYLES, help="always outline or always filled")
    p.add_argument("--colour", "--color", action="store_true",
                   help="random ink colour instead of black")
    p.add_argument("--out", default="shapes", help="output folder (default: shapes)")
    p.add_argument("--no-open", action="store_true",
                   help="don't open the picture after making it")
    args = p.parse_args(argv)

    if args.count < 1:
        p.error("--count must be at least 1")

    master = parse_seed(args.seed) if args.seed is not None else secrets.randbits(64)
    # A single picture uses the seed directly, so a seed from a file name
    # re-creates exactly that file.
    seeds = [master] if args.count == 1 else [child_seed(master, i) for i in range(args.count)]

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "seeds.csv")
    new_log = not os.path.exists(log_path)

    if args.count > 1:
        print(f"master seed {seed_name(master)}  ({args.count} pictures)")
    with open(log_path, "a", newline="") as log:
        writer = csv.writer(log)
        if new_log:
            writer.writerow(["seed", "shape", "style", "colour", "file"])
        number = next_number(args.out)
        for seed in seeds:
            image, shape, style = render(seed, args.shape, args.style, args.colour)
            name = f"shape_{number:03d}.png"
            number += 1
            path = os.path.join(args.out, name)
            image.save(path)
            writer.writerow([seed_name(seed), shape, style, int(args.colour), name])
            print(f"{path}  {shape} ({style})  seed {seed_name(seed)}")

    if args.count == 1 and not args.no_open:
        open_file(path)


def next_number(folder):
    """1 + the highest N among the shape_N.png files already in `folder`."""
    highest = 0
    for f in os.listdir(folder):
        if f.startswith("shape_") and f.endswith(".png") and f[6:-4].isdigit():
            highest = max(highest, int(f[6:-4]))
    return highest + 1


def open_file(path):
    """Show the picture in the computer's normal image viewer."""
    try:
        if sys.platform == "win32":
            os.startfile(os.path.abspath(path))
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
