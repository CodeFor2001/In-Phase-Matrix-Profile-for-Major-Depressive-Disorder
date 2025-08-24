#!/usr/bin/env python3
"""
hfd_real.py
Compute and visualize Higuchi Fractal Dimension (HFD) on real EEG for a chosen subject/epoch/channel.
Overlays the HFD value on the plot with an annotation arrow.

Examples:
  # Single 2s window from Cz
  python code/visualize/hfd_real.py --subject 2010002 --epoch 0 --channel Cz

  # Longer window and save figure
  python code/visualize/hfd_real.py --subject 2010002 --epoch 0 --channel Pz --win_sec 3 --kmax 8 --save_fig 1

  # Multiple consecutive windows (up to 5), each annotated
  python code/visualize/hfd_real.py --subject 2010002 --epoch 0 --channel Cz --multi 1 --win_sec 2 --save_fig 1
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def higuchi_fd(x: np.ndarray, kmax: int = 8) -> float:
    """Minimal Higuchi FD; returns |slope| from log-log fit for demo clarity."""
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
            Lm = (np.sum(np.abs(np.diff(x[idx]))) * (N - 1)) / (k * (idx.size - 1))
            Lk.append(Lm)
        L_vals.append(np.mean(Lk) if Lk else np.nan)

    L_vals = np.array(L_vals, dtype=float)
    mask = np.isfinite(L_vals) & (L_vals > 0)
    if mask.sum() < 2:
        return np.nan

    coeffs = np.polyfit(np.log(k_vals[mask]), np.log(L_vals[mask]), 1)
    return float(abs(coeffs[0]))

def load_subject_epoch(subject_id: str, epoch_idx: int, channel_name: str | None):
    npz_path = Path(f"data/interim/sub-{subject_id}_epochs.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    epochs = d["epochs"]              # (n_epochs, n_channels, samples)
    channels = list(d["channels"])
    if epoch_idx >= epochs.shape[0]:
        raise IndexError(f"Requested epoch {epoch_idx} but only {epochs.shape[0]} available.")
    if channel_name and (channel_name in channels):
        ch_idx, chosen = channels.index(channel_name), channel_name
    else:
        ch_idx, chosen = 0, channels[0]
    sig = epochs[epoch_idx, ch_idx].astype(float)
    return sig, chosen, channels

def annotate_hfd(ax, x_pos: float, y_pos: float, text: str, xytext_offset=(0.5, 0.9),
                 color="tab:blue"):
    """
    Place an annotated label with arrow at (x_pos, y_pos).
    xytext_offset is in axis fraction units (0-1 relative to axes).
    """
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    xt = xlim[0] + xytext_offset[0] * (xlim[1] - xlim[0])
    yt = ylim[0] + xytext_offset[1] * (ylim[1] - ylim[0])
    ax.annotate(
        text,
        xy=(x_pos, y_pos),
        xytext=(xt, yt),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.8, alpha=0.9),
        color=color,
        fontsize=10,
        ha="center",
        va="center",
    )

def plot_single_window(sig: np.ndarray, fs: float, win_sec: float, hfd_val: float,
                       subject_id: str, epoch_idx: int, ch_name: str, save_fig: bool):
    n = int(win_sec * fs)
    t = np.arange(n) / fs
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, sig[:n], lw=1, color="black")
    ax.set_title(f"Real EEG slice: sub-{subject_id}, ep {epoch_idx}, ch {ch_name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    # Draw a subtle window box and annotate HFD pointing to the mid-point of the window
    ax.axvspan(t[0], t[-1], color="orange", alpha=0.12)
    y_mid = 0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1])
    annotate_hfd(ax, x_pos=t[len(t)//2], y_pos=y_mid, text=f"HFD = {hfd_val:.4f}",
                 xytext_offset=(0.82, 0.82), color="tab:blue")
    fig.tight_layout()
    if save_fig:
        out = Path(f"reports/figures/hfd_real_sub-{subject_id}_ep{epoch_idx}_{ch_name}_{win_sec:.2f}s.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220)
        print(f"Saved figure to {out}")
    plt.show()

def plot_multi_windows(sig: np.ndarray, fs: float, win_sec: float, subject_id: str,
                       epoch_idx: int, ch_name: str, h_vals: list[float], starts: list[int], save_fig: bool):
    n = int(win_sec * fs)
    t = np.arange(len(sig)) / fs
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(t, sig, lw=1, color="black")
    ax.set_title(f"Real EEG (sub-{subject_id}, ep {epoch_idx}, ch {ch_name}) with HFD windows")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")

    # Mark windows and annotate each with its HFD, arrow pointing to each window center
    for s0, h in zip(starts, h_vals):
        x0, x1 = s0 / fs, (s0 + n) / fs
        ax.axvspan(x0, x1, color='orange', alpha=0.12)
        x_center = (x0 + x1) / 2.0
        y_mid = 0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1])
        annotate_hfd(ax, x_pos=x_center, y_pos=y_mid, text=f"HFD={h:.3f}",
                     xytext_offset=(min(0.95, x_center / (t[-1] if t[-1] else 1)), 0.88),
                     color="tab:green")

    fig.tight_layout()
    if save_fig:
        out = Path(f"reports/figures/hfd_real_multi_sub-{subject_id}_ep{epoch_idx}_{ch_name}_{win_sec:.2f}s.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220)
        print(f"Saved figure to {out}")
    plt.show()
    print(f"HFD values: {[round(v, 4) for v in h_vals]}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="Subject ID, e.g., 2010002")
    p.add_argument("--epoch", type=int, default=0, help="Epoch index")
    p.add_argument("--channel", type=str, default="Cz", help="Channel name (fallback to first if not found)")
    p.add_argument("--fs", type=float, default=200.0, help="Sampling rate of stored epochs")
    p.add_argument("--win_sec", type=float, default=2.0, help="Window length (s) for HFD")
    p.add_argument("--kmax", type=int, default=8, help="Higuchi parameter kmax")
    p.add_argument("--multi", type=int, default=0, help="If 1, compute HFD on multiple non-overlapping windows")
    p.add_argument("--save_fig", type=int, default=0, help="If 1, save figures to reports/figures/")
    args = p.parse_args()

    sig, ch_name, _ = load_subject_epoch(args.subject, args.epoch, args.channel)

    if not args.multi:
        n = int(args.win_sec * args.fs)
        if n < 16:
            raise ValueError("win_sec too small; use >= 0.1 s.")
        s = sig[:n]
        h = higuchi_fd(s, kmax=args.kmax)
        plot_single_window(sig, args.fs, args.win_sec, h, args.subject, args.epoch, ch_name, bool(args.save_fig))
        print(f"HFD (sub-{args.subject}, epoch {args.epoch}, channel {ch_name}, "
              f"{args.win_sec}s, kmax={args.kmax}): {h:.6f}")
        return

    n = int(args.win_sec * args.fs)
    if n < 16:
        raise ValueError("win_sec too small; use >= 0.1 s.")
    starts = list(range(0, max(len(sig) - n + 1, 0), n))[:5]
    h_vals = [higuchi_fd(sig[s0:s0 + n], kmax=args.kmax) for s0 in starts]
    plot_multi_windows(sig, args.fs, args.win_sec, args.subject, args.epoch, ch_name, h_vals, starts, bool(args.save_fig))
    for i, (s0, h) in enumerate(zip(starts, h_vals), start=1):
        print(f"Window {i}: start {s0/args.fs:.2f}s → HFD={h:.6f}")

if __name__ == "__main__":
    main()
