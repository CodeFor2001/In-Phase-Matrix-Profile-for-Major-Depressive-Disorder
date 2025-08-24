#!/usr/bin/env python3
"""
one_subject_raw.py
Plot raw EEG for one subject, one epoch, one channel.

Examples:
  python code/visualize/one_subject_raw.py --subject 2010002 --epoch 0 --channel Cz
  python code/visualize/one_subject_raw.py --subject 2010002 --epoch 0 --channel Pz --save_fig 1
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="Subject ID, e.g., 2010002")
    p.add_argument("--epoch", type=int, default=0, help="Epoch index to plot")
    p.add_argument("--channel", type=str, default="Cz", help="Channel name (fallback to first if not found)")
    p.add_argument("--fs", type=float, default=200.0, help="Sampling rate of stored epochs")
    p.add_argument("--save_fig", type=int, default=0, help="If 1, save figure to reports/figures/")
    args = p.parse_args()

    npz_path = Path(f"data/interim/sub-{args.subject}_epochs.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)
    epochs = d["epochs"]
    channels = list(d["channels"])
    subj = str(d.get("subject", args.subject))

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

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, sig, lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Raw EEG: sub-{subj}, epoch {args.epoch}, channel {ch_name}")
    fig.tight_layout()
    if args.save_fig:
        out = Path(f"reports/figures/raw_sub-{subj}_ep{args.epoch}_{ch_name}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"Saved figure to {out}")
    plt.show()

    print(f"Info → subject: {subj}, epoch: {args.epoch}, channel: {ch_name}, "
          f"n_samples: {len(sig)}, fs: {fs}")

if __name__ == "__main__":
    main()
