"""Learned flight-centred latent state: a joint window transformer over all observed airport events.

- `tokens`: event table (departures from a masked movement table, arrivals as flown) and windows.
- `model`: transformer encoder with pairwise (time, runway, stand) attention biases.
- `train`: training, out-of-fold prediction and embedding alignment.
"""
