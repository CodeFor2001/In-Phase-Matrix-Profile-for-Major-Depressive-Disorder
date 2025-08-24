#!/usr/bin/env python3
"""
pmp_demo.py
Visualize pMP mechanics on real EEG: raw slice with query, distance profile minima, and motif vs discord overlay.

Examples:
  python code/visualize/pmp_demo.py --subject 2010002 --epoch 0 --channel Cz
  python code/visualize/pmp_demo.py --subject 2010002 --epoch 0 --channel Cz --slice_sec 6 --m_sec 1.0 --save_fig 1
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def z_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd if sd > 1e-12 else 1.0)

def distance_profile(signal: np.ndarray, query: np.ndarray) -> np.ndarray:
    s = np.asarray(signal, float)
    q = np.asarray(query, float)
    m, N = len(q), len(s)
    qz = z_norm(q)
    dists = np.empty(N - m + 1)
    for i in range(N - m + 1):
        wz = z_norm(s[i:i + m])
        dists[i] = np.sqrt(np.sum((qz - wz) ** 2))
    return dists

def load_signal(subject: str, epoch: int, channel: str | None):
    npz_path = Path(f"data/interim/sub-{subject}_epochs.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    epochs = d["epochs"]; channels = list(d["channels"])
    if epoch >= epochs.shape[0]:
        raise IndexError(f"Requested epoch {epoch} but only {epochs.shape[0]} available.")
    if channel and (channel in channels):
        ch_idx, ch_name = channels.index(channel), channel
    else:
        ch_idx, ch_name = 0, channels[0]
    sig = epochs[epoch, ch_idx].astype(float)
    return sig, ch_name

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--channel", type=str, default="Cz")
    p.add_argument("--fs", type=float, default=200.0)
    p.add_argument("--slice_sec", type=float, default=5.0)
    p.add_argument("--m_sec", type=float, default=1.0)
    p.add_argument("--save_fig", type=int, default=0)
    args = p.parse_args()

    sig_full, ch_name = load_signal(args.subject, args.epoch, args.channel)
    fs = float(args.fs)
    n_slice = int(args.slice_sec * fs)
    s = sig_full[:n_slice]
    m = int(args.m_sec * fs)
    if m >= len(s):
        raise ValueError("m_sec too large for selected slice.")

    # Query and DP
    q = s[0:m]
    dp = distance_profile(s, q)
    # Local minima spaced by ~alpha period ~0.1s
    min_idx, _ = find_peaks(-dp, distance=int(0.1 * fs))

    # Figure 1: raw with query + distance profile with minima
    t = np.arange(len(s)) / fs
    t_dp = np.arange(len(dp)) / fs
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    ax1.plot(t, s, lw=1)
    ax1.axvspan(0, m / fs, color='orange', alpha=0.25, label=f"Query ({args.m_sec:.2f}s)")
    ax1.set_title(f"Raw EEG slice (sub-{args.subject}, ep {args.epoch}, ch {ch_name})")
    ax1.set_ylabel("Amplitude (µV)")
    ax1.legend(loc="upper right")

    ax2.plot(t_dp, dp, lw=1.2, label="Distance profile")
    if len(min_idx) > 0:
        ax2.plot(min_idx / fs, dp[min_idx], 'ro', ms=5, label="In-phase minima")
    ax2.set_title("Distance profile: lower = more similar / in-phase")
    ax2.set_xlabel("Time shift (s)")
    ax2.set_ylabel("Z-norm Euclidean dist")
    ax2.legend()
    fig1.tight_layout()
    if args.save_fig:
        out = Path(f"reports/figures/pmp_dp_sub-{args.subject}_ep{args.epoch}_{ch_name}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out, dpi=200)
        print(f"Saved figure to {out}")
    plt.show()

    # Figure 2: motif vs discord overlay (z-norm)
    order = np.argsort(dp)
    best_idx = order[:5]         # 5 best matches
    worst_idx = order[::-1][:1]  # 1 worst match

    tq = np.arange(m) / fs
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tq, z_norm(q), 'k', lw=2, label="Query (z-norm)")
    for i, idx in enumerate(best_idx):
        ax.plot(tq, z_norm(s[idx:idx + m]), lw=1, alpha=0.8, label="Motif" if i == 0 else None)
    for idx in worst_idx:
        ax.plot(tq, z_norm(s[idx:idx + m]), 'r--', lw=1.5, label="Poor match (discord-like)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z-norm amplitude")
    ax.set_title("Motifs (best matches) vs. discord-like (worst) windows")
    ax.legend()
    fig2.tight_layout()
    if args.save_fig:
        out = Path(f"reports/figures/pmp_motif_discord_sub-{args.subject}_ep{args.epoch}_{ch_name}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(out, dpi=200)
        print(f"Saved figure to {out}")
    plt.show()

    # Figure 3: mark top-3 motif locations on the raw timeline
    top3 = list(best_idx[:3])
    fig3, ax3 = plt.subplots(figsize=(11, 3))
    ax3.plot(t, s, lw=1)
    ax3.axvspan(0, m / fs, color='orange', alpha=0.25, label="Query")
    for j, i0 in enumerate(top3, start=1):
        x0, x1 = i0 / fs, (i0 + m) / fs
        ax3.axvspan(x0, x1, color='tab:green', alpha=0.18)
        ax3.annotate(f"Motif {j}", xy=(x0, ax3.get_ylim()[1]*0.85),
                     xytext=(x0, ax3.get_ylim()[1]*0.95),
                     arrowprops=dict(arrowstyle="->", color='tab:green'),
                     color='tab:green', fontsize=9)
    ax3.set_title(f"Top-3 motif locations on raw (sub-{args.subject}, ep {args.epoch}, ch {ch_name})")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude (µV)")
    ax3.legend(loc="upper right")
    fig3.tight_layout()
    if args.save_fig:
        out = Path(f"reports/figures/pmp_motif_locations_sub-{args.subject}_ep{args.epoch}_{ch_name}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig3.savefig(out, dpi=200)
        print(f"Saved figure to {out}")
    plt.show()

if __name__ == "__main__":
    main()
