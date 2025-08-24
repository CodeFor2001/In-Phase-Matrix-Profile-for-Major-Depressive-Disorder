#!/usr/bin/env python3
"""
hfd_real.py
Compute and visualize Higuchi Fractal Dimension (HFD) on real EEG for a chosen subject/epoch/channel.

Usage examples:
  - Basic (2-second slice at Cz):
      python code/visualize/hfd_real.py --subject 2010002 --epoch 0 --channel Cz
  - Different channel and window:
      python code/visualize/hfd_real.py --subject 2010002 --epoch 0 --channel Pz --win_sec 3 --kmax 8
  - Show multiple windows within the same epoch:
      python code/visualize/hfd_real.py --subject 2010002 --epoch 0 --channel Cz --multi 1
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def higuchi_fd(x: np.ndarray, kmax: int = 8) -> float:
    """
    Minimal Higuchi fractal dimension implementation.
    Returns the absolute slope magnitude from the log-log fit for clarity in demos.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < (kmax + 2):
        return np.nan

    L_vals = []
    k_vals = np.arange(1, kmax + 1, dtype=int)
    for k in k_vals:
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k, dtype=int)
            if idx.size < 2:
                continue
            # Average curve length at scale k, starting offset m
            Lm = (np.sum(np.abs(np.diff(x[idx]))) * (N - 1)) / (k * (idx.size - 1))
            Lk.append(Lm)
        L_vals.append(np.mean(Lk) if Lk else np.nan)

    L_vals = np.array(L_vals, dtype=float)
    mask = np.isfinite(L_vals) & (L_vals > 0)
    if mask.sum() < 2:
        return np.nan

    # Linear fit in log-log space
    coeffs = np.polyfit(np.log(k_vals[mask]), np.log(L_vals[mask]), 1)
    return abs(coeffs[0])

def load_subject_epoch(subject_id: str, epoch_idx: int, channel_name: str | None, fs: float):
    """
    Load one subject's epoch and pick a channel by name (fallback to first).
    Returns (signal, chosen_channel_name, channels_list).
    """
    npz_path = Path(f"data/interim/sub-{subject_id}_epochs.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)
    epochs = d["epochs"]              # (n_epochs, n_channels, samples)
    channels = list(d["channels"])    # names or 'Ch#' fallback

    if epoch_idx >= epochs.shape[0]:
        raise IndexError(f"Requested epoch {epoch_idx} but only {epochs.shape[0]} epochs available.")

    if channel_name and (channel_name in channels):
        ch_idx = channels.index(channel_name)
        chosen = channel_name
    else:
        ch_idx = 0
        chosen = channels[0]

    sig = epochs[epoch_idx, ch_idx].astype(float)
    return sig, chosen, channels

def plot_single_window(sig: np.ndarray, fs: float, win_sec: float, hfd_val: float,
                       subject_id: str, epoch_idx: int, ch_name: str):
    n = int(win_sec * fs)
    t = np.arange(n) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, sig[:n], lw=1)
    plt.title(f"Real EEG slice: sub-{subject_id}, ep {epoch_idx}, ch {ch_name} (HFD ≈ {hfd_val:.4f})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.tight_layout()
    plt.show()

def plot_multi_windows(sig: np.ndarray, fs: float, win_sec: float, subject_id: str,
                       epoch_idx: int, ch_name: str, h_vals: list[float], starts: list[int]):
    n = int(win_sec * fs)
    t = np.arange(len(sig)) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, sig, lw=1)
    for s0 in starts:
        plt.axvspan(s0 / fs, (s0 + n) / fs, color='orange', alpha=0.15)
    plt.title(f"Real EEG (sub-{subject_id}, ep {epoch_idx}, ch {ch_name}) with HFD windows")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.tight_layout()
    plt.show()
    print(f"HFD values (kmax windows): {[round(v, 4) for v in h_vals]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Subject ID, e.g., 2010002")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch index")
    parser.add_argument("--channel", type=str, default="Cz", help="Channel name (fallback to first if not found)")
    parser.add_argument("--fs", type=float, default=200.0, help="Sampling rate of stored epochs")
    parser.add_argument("--win_sec", type=float, default=2.0, help="Window length in seconds for HFD")
    parser.add_argument("--kmax", type=int, default=8, help="Higuchi parameter kmax")
    parser.add_argument("--multi", type=int, default=0, help="If 1, compute HFD on multiple windows within the epoch")
    args = parser.parse_args()

    # Load the requested signal
    sig, ch_name, _ = load_subject_epoch(args.subject, args.epoch, args.channel, args.fs)

    # Single window mode
    if not args.multi:
        n = int(args.win_sec * args.fs)
        if n < 16:
            raise ValueError("win_sec too small; increase above ~0.1 s for stability.")
        s = sig[:n]
        h = higuchi_fd(s, kmax=args.kmax)
        plot_single_window(sig, args.fs, args.win_sec, h, args.subject, args.epoch, ch_name)
        print(f"HFD (sub-{args.subject}, epoch {args.epoch}, channel {ch_name}, "
              f"{args.win_sec}s, kmax={args.kmax}): {h:.6f}")
        return

    # Multi-window mode
    n = int(args.win_sec * args.fs)
    if n < 16:
        raise ValueError("win_sec too small; increase above ~0.1 s for stability.")

    # Non-overlapping windows; up to 5 windows for a compact demo
    starts = list(range(0, max(len(sig) - n + 1, 0), n))[:5]
    h_vals = [higuchi_fd(sig[s0:s0 + n], kmax=args.kmax) for s0 in starts]
    plot_multi_windows(sig, args.fs, args.win_sec, args.subject, args.epoch, ch_name, h_vals, starts)
    for i, (s0, h) in enumerate(zip(starts, h_vals), start=1):
        print(f"Win {i}: start {s0/args.fs:.2f}s, HFD={h:.6f}")

if __name__ == "__main__":
    main()
