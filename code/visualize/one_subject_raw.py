#!/usr/bin/env python3
"""
one_subject_raw.py
Plot raw EEG for one subject, one epoch, one channel.
Usage:
  python code/visualize/one_subject_raw.py --subject 2010002 --epoch 0 --channel Cz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Subject ID, e.g., 2010002")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch index to plot")
    parser.add_argument("--channel", type=str, default="Cz", help="Channel name to plot")
    parser.add_argument("--fs", type=float, default=200.0, help="Sampling rate of epochs")
    args = parser.parse_args()

    npz_path = Path(f"data/interim/sub-{args.subject}_epochs.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)
    epochs = d["epochs"]             # (n_epochs, n_channels, 4096)
    channels = list(d["channels"])   # list of names or generic Ch1...
    subject = str(d.get("subject", args.subject))

    if args.epoch >= epochs.shape[0]:
        raise IndexError(f"Requested epoch {args.epoch} but only {epochs.shape[0]} epochs available.")

    if args.channel in channels:
        ch_idx = channels.index(args.channel)
        ch_name = args.channel
    else:
        ch_idx = 0
        ch_name = channels[0]

    sig = epochs[args.epoch, ch_idx]
    fs = args.fs
    t = np.arange(len(sig)) / fs

    plt.figure(figsize=(10, 3))
    plt.plot(t, sig, lw=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.title(f"Raw EEG: sub-{subject}, epoch {args.epoch}, channel {ch_name}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
