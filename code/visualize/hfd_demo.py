#!/usr/bin/env python3
"""
hfd_demo.py
Visualize HFD on simple signals and optionally compute on a subject signal.
Usage:
  python code/visualize/hfd_demo.py
  python code/visualize/hfd_demo.py --subject 2010002 --epoch 0 --channel Cz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def higuchi_fd(x, kmax=8):
    x = np.asarray(x, float)
    N = len(x)
    L = []
    k_vals = range(1, kmax+1)
    for k in k_vals:
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            Lm = (np.sum(np.abs(np.diff(x[idx]))) * (N-1)) / (k * (len(idx)-1))
            Lk.append(Lm)
        L.append(np.mean(Lk) if Lk else np.nan)
    L = np.array(L)
    k = np.array(list(k_vals), float)
    mask = np.isfinite(L) & (L > 0)
    if mask.sum() < 2:
        return np.nan
    coeffs = np.polyfit(np.log(k[mask]), np.log(L[mask]), 1)
    # Higuchi FD is typically 1 - slope or similar variant; for demo we return abs slope.
    return abs(coeffs[0])

def demo_signals():
    fs = 200.0
    t = np.arange(0, 2.0, 1/fs)  # 2 seconds
    sine = np.sin(2*np.pi*10*t)
    noisy = sine + 0.2*np.random.randn(len(t))
    white = np.random.randn(len(t))
    return t, sine, noisy, white, fs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", help="Subject ID for optional real EEG HFD")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--channel", type=str, default="Cz")
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--kmax", type=int, default=8)
    args = parser.parse_args()

    # Demo synthetic signals
    t, sine, noisy, white, _ = demo_signals()
    h_sine = higuchi_fd(sine, kmax=args.kmax)
    h_noisy = higuchi_fd(noisy, kmax=args.kmax)
    h_white = higuchi_fd(white, kmax=args.kmax)

    plt.figure(figsize=(10,6))
    plt.subplot(3,1,1); plt.plot(t, sine); plt.title(f"Clean 10 Hz sine (HFD~{h_sine:.3f})")
    plt.subplot(3,1,2); plt.plot(t, noisy); plt.title(f"Sine + noise (HFD~{h_noisy:.3f})")
    plt.subplot(3,1,3); plt.plot(t, white); plt.title(f"White noise (HFD~{h_white:.3f})")
    plt.tight_layout(); plt.show()

    # Optional: compute HFD on a real subject/channel/epoch slice (2s)
    if args.subject:
        npz_path = Path(f"data/interim/sub-{args.subject}_epochs.npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"Not found: {npz_path}")
        d = np.load(npz_path, allow_pickle=True)
        epochs = d["epochs"]
        channels = list(d["channels"])
        if args.epoch >= epochs.shape[0]:
            raise IndexError(f"Requested epoch {args.epoch} but only {epochs.shape[0]} available.")
        if args.channel in channels:
            ch_idx = channels.index(args.channel)
            ch_name = args.channel
        else:
            ch_idx = 0
            ch_name = channels[0]
        sig = epochs[args.epoch, ch_idx]
        fs = args.fs
        n = int(2*fs)
        s = sig[:n]
        h_real = higuchi_fd(s, kmax=args.kmax)
        print(f"Real EEG HFD (sub-{args.subject}, epoch {args.epoch}, channel {ch_name}): {h_real:.4f}")

if __name__ == "__main__":
    main()
