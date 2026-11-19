# SNN-encoding


A small spiking neural network in plain NumPy that learns to recognise shapes,
and three ways to turn a picture into spikes.

## Spike encodings

A spiking network only takes spikes as input, so each picture has to be turned
into spikes first. `spike_encodings.py` has three ways to do it:

| encoding | what carries the darkness of a pixel | spikes per pixel | random? |
|---|---|---|---|
| `rate` | how **often** it fires | many | yes |
| `latency` | how **early** it fires (darker = earlier) | exactly one, white never | no |
| `phase` | **where** in a repeating 8-step cycle it fires (darkness as a binary number) | a few per cycle | no |

```
python spike_encodings.py                    # encodings.png: one picture, three encodings
python simple_snn.py --encoding rate         # train and test with each one
python simple_snn.py --encoding latency
python simple_snn.py --encoding phase
python simple_snn.py --encoding latency --predict shapes/shape_003.png
```

Test accuracy with the default settings (4 shapes, guessing = 25%):

| rate | latency | phase |
|---|---|---|
| 81.2% | 89.8% | 84.5% |
