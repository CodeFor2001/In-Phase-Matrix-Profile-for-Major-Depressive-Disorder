#!/usr/bin/env python3
"""
pmp_demo.py
Visualize pMP mechanics: distance profile minima (motifs) and discord-like windows.
Usage:
  python code/visualize/pmp_demo.py --subject 2010002 --epoch 0 --channel Cz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

def z_norm(x):
    x = np.asarray(x, float)
    mu = x.mean()
    sd = x.std()
    return (x - mu) / (sd if sd > 1e-12 else 1.0)

def distance_profile(signal, query):
    s = np.asarray(signal, float)
    q = np.asarray(query, float)
    m = len(q)
    N = len(s)
    dists = []
    qz = z_norm(q)
    for i in range(N - m + 1):
        wz = z_norm(s[i:i+m])
        d = np.sqrt(np.sum((qz - wz)**2))
        dists.append(d)
    return np.array(dists)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--channel", type=str, default="Cz")
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--slice_sec", type=float, default=5.0, help="Seconds to visualize")
    parser.add_argument("--m_sec", type=float, default=1.0, help="Subsegment length (sec)")
    args = parser.parse_args()

    npz_path = Path(f"data/interim/sub-{args.subject}_epochs.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)
    epochs = d["epochs"]
    channels = list(d["channels"])

    if args.epoch >= epochs.shape[0]:
        raise IndexError(f"Requested epoch {args.epoch} but only {epochs.shape[0]} epochs available.")

    if args.channel in channels:
        ch_idx = channels.index(args.channel)
        ch_name = args.channel
    else:
        ch_idx = 0
        ch_name = channels[0]

    sig_full = epochs[args.epoch, ch_idx].astype(float)
    fs = args.fs
    n_slice = int(args.slice_sec * fs)
    s = sig_full[:n_slice]
    m = int(args.m_sec * fs)

    if m >= len(s):
        raise ValueError("m_sec slice too large for selected window.")

    # Query window and distance profile
    q = s[0:m]
    dp = distance_profile(s, q)

    # Find in-phase minima, separated by ~alpha period (≈0.1 s -> ~20 samples at 200Hz)
    min_idx, _ = find_peaks(-dp, distance=int(0.1*fs))

    # Plot raw with query
    t = np.arange(len(s))/fs
    t_dp = np.arange(len(dp))/fs
    plt.figure(figsize=(11, 6))
    plt.subplot(2,1,1)
    plt.plot(t, s, lw=1)
    plt.axvspan(0, m/fs, color='orange', alpha=0.25, label=f"Query ({args.m_sec:.2f}s)")
    plt.title(f"Raw EEG slice (channel {ch_name}) and query window")
    plt.ylabel("Amplitude (uV)")
    plt.legend(loc="upper right")

    # Plot distance profile
    plt.subplot(2,1,2)
    plt.plot(t_dp, dp, lw=1.2, label="Distance profile")
    if len(min_idx) > 0:
        plt.plot(min_idx/fs, dp[min_idx], 'ro', ms=5, label="In-phase minima")
    plt.xlabel("Time shift (s)")
    plt.ylabel("Z-norm Euclidean dist")
    plt.title("Distance profile: lower = more similar/in-phase")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Motif vs discord-like overlay
    order = np.argsort(dp)
    best_idx = order[:5]                   # motifs
    worst_idx = order[::-1][:1]            # discord-like
    t_q = np.arange(m)/fs

    plt.figure(figsize=(10,4))
    plt.plot(t_q, z_norm(q), 'k', lw=2, label="Query (z-norm)")
    for i, idx in enumerate(best_idx):
        plt.plot(t_q, z_norm(s[idx:idx+m]), lw=1, alpha=0.7, label="Motif" if i==0 else None)
    for idx in worst_idx:
        plt.plot(t_q, z_norm(s[idx:idx+m]), 'r--', lw=1.5, label="Poor match (discord-like)")
    plt.xlabel("Time (s)")
    plt.ylabel("z-norm amplitude")
    plt.title("Motifs (best matches) vs. discord-like (worst) windows")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
