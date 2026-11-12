"""A simple spiking neural network (SNN) that learns to recognise shapes.

It trains on pictures from shape_seeds.py and then guesses which shape a
picture shows. Everything is plain NumPy so every step can be read.

Run:
  python simple_snn.py                        train, test, save weights and plots
  python simple_snn.py --shapes circle square triangle star
  python simple_snn.py --predict shapes/shape_001.png   use saved weights on a picture
  python simple_snn.py --help                 all options

How it works
  1. Pictures: shape_seeds.py draws labelled 128x128 shapes from seeds, so the
     same --seed always gives the same training and test pictures.
  2. Look at the shape: the shape is cut out with a square box around it, so
     it is centred and the same size, then scaled down to SIZE x SIZE
     (default 32x32) "darkness" values, 0 = white and 1 = black. Without this
     one layer of weights cannot cope with shapes moving around (it scores
     barely above guessing).
  3. Encoding (rate coding): every pixel is an input neuron. At each of STEPS
     time steps it fires a spike at random, more often the darker it is. Every
     picture gets the same overall spike budget, so a thin outline and a big
     filled shape send about the same number of spikes.
  4. Neurons (leaky integrate-and-fire): every output neuron has a membrane
     potential v. Each step v leaks a little (v *= DECAY), then every input
     spike adds that connection's weight. When v reaches THRESHOLD the neuron
     fires a spike and v resets to 0. There is one output neuron per shape.
  5. Answer: the output neuron that fired the most spikes wins.
  6. Learning: after a batch of pictures the correct neuron should have fired
     at TARGET_RATE and the others not at all. Each weight changes by
         rate * (wanted firing - actual firing) * (how often its pixel fired)
     so pixels that were active get stronger links to the right neuron and
     weaker links to neurons that fired by mistake.
"""

import argparse
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("NumPy is not installed for this Python. Run:  python -m pip install numpy")

from PIL import Image

import shape_seeds

WEIGHTS_FILE = "snn_weights.npz"

# network settings
SIZE = 32            # pictures are shrunk to SIZE x SIZE input neurons
STEPS = 50           # time steps each picture is shown for
SPIKE_BUDGET = 60.0  # average number of input spikes per time step
DECAY = 0.9          # membrane leak per step (closer to 1 = leaks slower)
THRESHOLD = 1.0      # membrane potential that makes a neuron fire
TARGET_RATE = 0.5    # how often the correct output neuron should fire


# ---------- data ----------
def image_to_darkness(image, size=SIZE):
    """PIL image -> flat array of darkness values in [0, 1].

    The shape is first cut out with a square box around it, so it ends up
    centred and the same size whatever its place and size in the picture.
    """
    grey = image.convert("L")
    ys, xs = np.nonzero(np.asarray(grey) < 225)        # pixels that are not white
    if len(xs):
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 2
        grey = grey.crop((round(cx - half), round(cy - half),
                          round(cx + half), round(cy + half)))
    grey = grey.resize((size, size), Image.BOX)
    return 1.0 - np.asarray(grey, dtype=np.float64).ravel() / 255.0


def make_dataset(master_seed, count, shapes, size=SIZE):
    """`count` pictures with labels, cycling through `shapes` so classes are balanced."""
    pictures = np.empty((count, size * size))
    labels = np.empty(count, dtype=int)
    for i in range(count):
        label = i % len(shapes)
        seed = shape_seeds.child_seed(master_seed, i)
        image, _, _ = shape_seeds.render(seed, shape=shapes[label])
        pictures[i] = image_to_darkness(image, size)
        labels[i] = label
    return pictures, labels


# ---------- encoding ----------
def firing_rates(darkness):
    """Chance per step that each input neuron fires (same total budget per picture)."""
    darkness = np.atleast_2d(darkness)
    total = darkness.sum(axis=1, keepdims=True)
    return np.clip(darkness * SPIKE_BUDGET / np.maximum(total, 1e-9), 0.0, 1.0)


def encode(darkness, steps, rng):
    """Rate coding: returns spikes shaped (steps, pictures, inputs) of 0s and 1s."""
    rates = firing_rates(darkness)
    return (rng.random((steps,) + rates.shape) < rates).astype(np.float64)


# ---------- the network ----------
def run_network(weights, spikes, record=False):
    """Simulate leaky integrate-and-fire output neurons.

    spikes: (steps, pictures, inputs). Returns spike counts (pictures, outputs),
    plus the output spikes and membrane potentials over time when record=True.
    """
    steps, batch, _ = spikes.shape
    v = np.zeros((batch, weights.shape[0]))
    counts = np.zeros_like(v)
    out_spikes, potentials = [], []
    for t in range(steps):
        v = v * DECAY + spikes[t] @ weights.T     # leak, then add input
        fired = v >= THRESHOLD
        counts += fired
        if record:
            potentials.append(v.copy())
            out_spikes.append(fired.copy())
        v[fired] = 0.0                            # reset neurons that fired
    if record:
        return counts, np.array(out_spikes), np.array(potentials)
    return counts


def predict(weights, darkness, steps, rng):
    counts = run_network(weights, encode(darkness, steps, rng))
    return counts.argmax(axis=1), counts


def accuracy(weights, pictures, labels, steps, rng):
    guesses, _ = predict(weights, pictures, steps, rng)
    return float((guesses == labels).mean())


def train(pictures, labels, n_classes, epochs, lr, steps, rng, batch=32):
    weights = rng.normal(0.0, 0.01, (n_classes, pictures.shape[1]))
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(pictures))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            spikes = encode(pictures[idx], steps, rng)
            out_rate = run_network(weights, spikes) / steps
            in_rate = spikes.mean(axis=0)                     # (batch, inputs)
            wanted = np.zeros_like(out_rate)
            wanted[np.arange(len(idx)), labels[idx]] = TARGET_RATE
            weights += lr * (wanted - out_rate).T @ in_rate / len(idx)
        acc = accuracy(weights, pictures, labels, steps, rng)
        print(f"epoch {epoch:2d}/{epochs}  training accuracy {acc:6.1%}")
    return weights


# ---------- pictures of the network ----------
def plot_weights(weights, shapes, path):
    import matplotlib.pyplot as plt
    size = int(round(np.sqrt(weights.shape[1])))
    cols = min(4, len(shapes))
    rows = -(-len(shapes) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.8 * rows), squeeze=False)
    limit = np.abs(weights).max()
    for ax, w, name in zip(axes.flat, weights, shapes):
        ax.imshow(w.reshape(size, size), cmap="bwr_r", vmin=-limit, vmax=limit)
        ax.set_title(name)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("What each output neuron listens for\n(blue = excites it, red = calms it)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_raster(weights, darkness, label, shapes, steps, rng, path):
    import matplotlib.pyplot as plt
    spikes = encode(darkness, steps, rng)
    counts, out_spikes, potentials = run_network(weights, spikes, record=True)
    guess = int(counts[0].argmax())
    size = int(round(np.sqrt(darkness.size)))

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.6, 1.6, 1.6]})
    axes[0].imshow(darkness.reshape(size, size), cmap="gray_r", vmin=0, vmax=1)
    title = f"input: {shapes[label]}" if label is not None else "input"
    axes[0].set_title(title)
    axes[0].axis("off")

    t, neuron = np.nonzero(spikes[:, 0, :])
    axes[1].scatter(t, neuron, s=2, c="k")
    axes[1].set(title="input spikes (one row per pixel)", xlabel="time step",
                ylabel="input neuron", xlim=(-1, steps))

    t, neuron = np.nonzero(out_spikes[:, 0, :])
    axes[2].scatter(t, neuron, s=25, marker="|", c="tab:blue")
    axes[2].set_yticks(range(len(shapes)), [f"{s} ({int(c)})" for s, c in zip(shapes, counts[0])])
    axes[2].set(title=f"output spikes  ->  guess: {shapes[guess]}", xlabel="time step",
                xlim=(-1, steps), ylim=(-0.5, len(shapes) - 0.5))

    for i, name in enumerate(shapes):
        axes[3].plot(potentials[:, 0, i], lw=1.5 if i == guess else 0.8,
                     label=name, alpha=1.0 if i == guess else 0.5)
    axes[3].axhline(THRESHOLD, color="k", ls="--", lw=0.8)
    axes[3].set(title="membrane potential (dashed = threshold)", xlabel="time step")
    axes[3].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return guess, counts[0]


# ---------- command line ----------
def main(argv=None):
    p = argparse.ArgumentParser(description="Train a simple spiking neural network on shapes.")
    p.add_argument("--shapes", nargs="+", choices=shape_seeds.SHAPES,
                   default=["circle", "square", "triangle", "star"],
                   help="which shapes to learn (default: circle square triangle star)")
    p.add_argument("--train", type=int, default=2000, help="training pictures (default 2000)")
    p.add_argument("--test", type=int, default=400, help="test pictures (default 400)")
    p.add_argument("--epochs", type=int, default=15, help="passes over the training set")
    p.add_argument("--lr", type=float, default=0.5, help="learning rate (default 0.5)")
    p.add_argument("--steps", type=int, default=STEPS, help="time steps per picture")
    p.add_argument("--seed", default="snn", help="seed for pictures and spikes (default 'snn')")
    p.add_argument("--predict", metavar="IMAGE",
                   help="skip training: load saved weights and classify this picture")
    p.add_argument("--no-plots", action="store_true", help="don't save the plot images")
    args = p.parse_args(argv)

    master = shape_seeds.parse_seed(args.seed)
    rng = np.random.default_rng(master)

    if args.predict:
        if not os.path.exists(WEIGHTS_FILE):
            sys.exit(f"No {WEIGHTS_FILE} yet. Train first:  python simple_snn.py")
        saved = np.load(WEIGHTS_FILE)
        weights, shapes = saved["weights"], [str(s) for s in saved["shapes"]]
        darkness = image_to_darkness(Image.open(args.predict).convert("RGBA"),
                                     int(round(np.sqrt(weights.shape[1]))))
        guess, counts = predict(weights, darkness, args.steps, rng)
        print(f"{args.predict}: looks like a {shapes[guess[0]]}")
        for name, c in sorted(zip(shapes, counts[0]), key=lambda x: -x[1]):
            print(f"  {name:10s} {int(c):3d} spikes")
        if not args.no_plots:
            plot_raster(weights, darkness, None, shapes, args.steps, rng, "snn_predict.png")
            print("spike plot saved to snn_predict.png")
        return 0

    print(f"making {args.train} training + {args.test} test pictures "
          f"of {', '.join(args.shapes)} ...")
    # training and test pictures come from different master seeds, so they never overlap
    train_x, train_y = make_dataset(shape_seeds.child_seed(master, 0), args.train, args.shapes)
    test_x, test_y = make_dataset(shape_seeds.child_seed(master, 1), args.test, args.shapes)

    weights = train(train_x, train_y, len(args.shapes), args.epochs, args.lr, args.steps, rng)

    guesses, _ = predict(weights, test_x, args.steps, rng)
    print(f"\ntest accuracy on {args.test} new pictures: {(guesses == test_y).mean():.1%}"
          f"  (guessing at random: {1 / len(args.shapes):.1%})")
    print("\nrows = real shape, columns = guess")
    print(" " * 11 + "".join(f"{s[:8]:>9s}" for s in args.shapes))
    for i, name in enumerate(args.shapes):
        row = [(guesses[test_y == i] == j).sum() for j in range(len(args.shapes))]
        print(f"{name:>10s} " + "".join(f"{n:9d}" for n in row))

    np.savez(WEIGHTS_FILE, weights=weights, shapes=np.array(args.shapes))
    print(f"\nweights saved to {WEIGHTS_FILE}")
    if not args.no_plots:
        plot_weights(weights, args.shapes, "snn_weights.png")
        plot_raster(weights, test_x[0], test_y[0], args.shapes, args.steps, rng, "snn_raster.png")
        print("plots saved to snn_weights.png and snn_raster.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
