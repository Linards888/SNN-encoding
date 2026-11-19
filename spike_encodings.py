"""Three ways to turn a picture into spikes (spike encodings).

A spiking network can't take numbers as input, only spikes: short "blips"
that happen at a certain time step. An encoding is the rule that decides
WHICH input neuron fires WHEN. Every input neuron here is one pixel, and its
"darkness" goes from 0 (white) to 1 (black).

  rate     HOW OFTEN a pixel fires carries its darkness.
           Each step, each pixel fires at random with a chance that grows with
           its darkness. Dark pixel -> many spikes, white pixel -> none.
           Simple and robust, but noisy (it's random) and it needs many
           spikes and many steps to get the value across.

  latency  WHEN a pixel fires carries its darkness ("time to first spike").
           Each pixel fires exactly ONCE. The darker it is, the EARLIER it
           fires: black fires at step 0, light grey near the last step, white
           never. No randomness, and very few spikes, because one spike per
           pixel is all there is.

  phase    WHICH STEP INSIDE A REPEATING CYCLE a pixel fires carries its darkness.
           Time is cut into cycles of PHASE_BITS steps (8). The darkness is
           written as an 8-bit binary number, e.g. 0.71 -> 182 -> 10110110,
           and the pixel fires at step k of the cycle when bit k is 1. The
           step inside the cycle (the "phase") says how much the spike is
           worth: 1/2 for the first, 1/4 for the second, 1/8, ... 1/256,
           just like the digits of a binary fraction. The cycle repeats, so
           the network keeps hearing the picture. No randomness, and one
           cycle already gives the exact value (to 1/256).

What every encoding returns
  An array shaped (steps, pictures, pixels). 0 means "no spike". Any other
  number is a spike, and the number is how big a kick it gives the output
  neurons (its "size"). Rate and latency spikes all have the same size;
  phase spikes are worth less the later they come in the cycle.

Fair comparison: the same total input per picture
  Every picture, whatever the encoding and whether it is a thin outline or a
  big filled shape, delivers the same total input: SPIKE_BUDGET kicks of
  size 1 per step on average, so SPIKE_BUDGET * steps in total. Rate coding
  does this by setting the firing chances; latency and phase do it by
  scaling the size of the picture's spikes. Without this, filled shapes
  would push the output neurons much harder than outlines, and the network
  would learn "how much ink" instead of "what shape".

Run this file to see one picture in all three encodings:
  python spike_encodings.py                       a test picture
  python spike_encodings.py shapes/shape_001.png  your own picture
It saves encodings.png.
"""

import sys

import numpy as np

ENCODINGS = ["rate", "latency", "phase"]

SPIKE_BUDGET = 60.0   # average total input per time step, the same for every encoding
LATENCY_MIN = 0.05    # pixels lighter than this never fire in latency coding
PHASE_BITS = 8        # steps per phase cycle = bits of darkness sent per cycle


def _same_total(spikes, steps):
    """Scale each picture's spikes so its total input is SPIKE_BUDGET * steps."""
    total = spikes.sum(axis=(0, 2), keepdims=True)          # one total per picture
    return spikes * (SPIKE_BUDGET * steps / np.maximum(total, 1e-9))


def rate(darkness, steps, rng):
    """Rate coding: each step, fire at random with a chance proportional to darkness."""
    darkness = np.atleast_2d(darkness)
    # chances are set so a picture fires SPIKE_BUDGET spikes per step on average
    chance = darkness * SPIKE_BUDGET / np.maximum(darkness.sum(axis=1, keepdims=True), 1e-9)
    chance = np.clip(chance, 0.0, 1.0)
    return (rng.random((steps,) + chance.shape) < chance).astype(np.float64)


def latency(darkness, steps, rng=None):
    """Latency coding: one spike per pixel, darker = earlier. White never fires."""
    darkness = np.atleast_2d(darkness)
    fire_time = np.round((1.0 - darkness) * (steps - 1)).astype(int)
    spikes = np.zeros((steps,) + darkness.shape)
    picture, pixel = np.nonzero(darkness >= LATENCY_MIN)
    spikes[fire_time[picture, pixel], picture, pixel] = 1.0
    return _same_total(spikes, steps)


def phase_weights(steps):
    """What a spike is worth at each step: 1/2, 1/4, ... 1/256, then repeat."""
    return 0.5 ** (np.arange(steps) % PHASE_BITS + 1)


def phase(darkness, steps, rng=None):
    """Phase coding: darkness as a binary fraction, one bit per step, repeated each cycle."""
    darkness = np.atleast_2d(darkness)
    levels = 2 ** PHASE_BITS - 1                                # 255
    whole = np.round(darkness * levels).astype(int)             # 0.71 -> 182
    bit_of_step = PHASE_BITS - 1 - np.arange(steps) % PHASE_BITS  # biggest bit first
    bits = (whole[None] >> bit_of_step[:, None, None]) & 1      # (steps, pictures, pixels)
    spikes = bits * phase_weights(steps)[:, None, None]
    return _same_total(spikes, steps)


def encode(darkness, steps, rng, encoding="rate"):
    """Turn darkness values into spikes with the chosen encoding."""
    return {"rate": rate, "latency": latency, "phase": phase}[encoding](darkness, steps, rng)


# ---------- a picture of the three encodings ----------
def plot_encodings(darkness, steps, rng, path, title="input"):
    import matplotlib.pyplot as plt
    size = int(round(np.sqrt(darkness.size)))
    # follow a handful of pixels with different darkness through all three encodings
    order = np.argsort(darkness)
    picks = [order[np.searchsorted(darkness[order], level)] for level in (0.98, 0.71, 0.4, 0.15, 0.0)]
    picks = [p if p < darkness.size else order[-1] for p in picks]

    fig = plt.figure(figsize=(16, 8.5))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1.5, 1.5, 1.5], height_ratios=[1.3, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(darkness.reshape(size, size), cmap="gray_r", vmin=0, vmax=1)
    for n, p in enumerate(picks):
        ax.plot(p % size, p // size, "o", mfc="none", mec=f"C{n}", ms=9, mew=2)
    ax.set_title(f"{title}\n(circles: pixels followed below)")
    ax.axis("off")

    for col, name in enumerate(ENCODINGS, start=1):
        spikes = encode(darkness, steps, rng, name)[:, 0, :]
        top = fig.add_subplot(grid[0, col])
        t, pixel = np.nonzero(spikes)
        top.scatter(t, pixel, s=1, c="k")
        top.set(title=f"{name}: {len(t)} spikes", xlabel="time step",
                ylabel="input neuron (pixel)", xlim=(-1, steps), ylim=(darkness.size, 0))

        bottom = fig.add_subplot(grid[1, col])
        for n, p in enumerate(picks):
            t = np.nonzero(spikes[:, p])[0]
            height = spikes[t, p] / spikes.max()         # bar height = spike size
            bottom.vlines(t, n - 0.4 * height, n + 0.4 * height, color=f"C{n}", lw=2)
        bottom.set_yticks(range(len(picks)), [f"darkness {darkness[p]:.2f}" for p in picks])
        bottom.set(xlabel="time step  (bar height = spike size)", xlim=(-1, steps),
                   ylim=(len(picks) - 0.5, -0.5))
        if name == "phase":
            for start in range(0, steps, PHASE_BITS):
                bottom.axvline(start - 0.5, color="0.8", lw=0.8, zorder=0)
    fig.suptitle("The same picture as spikes, three ways   "
                 "(rate: how often, latency: how early, phase: where in each 8-step cycle)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    from PIL import Image

    import shape_seeds
    import simple_snn

    rng = np.random.default_rng(0)
    if len(sys.argv) > 1:
        dark = simple_snn.image_to_darkness(Image.open(sys.argv[1]).convert("RGBA"))
        name = sys.argv[1]
    else:
        image, shape, style = shape_seeds.render(shape_seeds.parse_seed("encodings"), shape="star")
        dark, name = simple_snn.image_to_darkness(image), f"{style} {shape}"
    plot_encodings(dark, simple_snn.STEPS, rng, "encodings.png", title=name)
    print("saved encodings.png")
