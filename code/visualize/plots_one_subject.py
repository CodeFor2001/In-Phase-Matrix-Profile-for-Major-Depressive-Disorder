#!/usr/bin/env python3
"""
plots_one_subject.py
Create QA plots from a per-epoch features CSV for one subject.

Main output:
- pMP vs HFD (per-channel medians) scatter with a linear trend line and correlation stats.

Usage:
  python code/qa/plots_one_subject.py --csv data/qa/sub-2010002_per_epoch_features.csv
  python code/qa/plots_one_subject.py --csv data/qa/sub-2010002_per_epoch_features.csv --subject 2010002 --save 1
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_per_epoch(csv_path: Path):
    df = pd.read_csv(csv_path)
    pmp_cols = [c for c in df.columns if c.endswith("_pMP")]
    hfd_cols = [c for c in df.columns if c.endswith("_HFD")]
    return df, pmp_cols, hfd_cols

def channel_medians(df: pd.DataFrame, pmp_cols, hfd_cols):
    # Align by channel stem (remove suffix)
    pmp_med = {c[:-4]: df[c].median() for c in pmp_cols}   # strip "_pMP"
    hfd_med = {c[:-4]: df[c].median() for c in hfd_cols}   # strip "_HFD"
    chans = sorted(set(pmp_med) & set(hfd_med))
    x = np.array([pmp_med[ch] for ch in chans], dtype=float)
    y = np.array([hfd_med[ch] for ch in chans], dtype=float)
    return chans, x, y

def compute_correlations(x: np.ndarray, y: np.ndarray):
    # Pearson and Spearman (robust to monotonic, non-linear)
    # Avoid external heavy deps; use scipy if available, else fallback
    try:
        from scipy.stats import pearsonr, spearmanr
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
    except Exception:
        # Fallback: compute Pearson r manually, p-value unavailable
        pr = float(np.corrcoef(x, y)[0,1])
        pp = np.nan
        # Spearman: rank then Pearson
        xr = pd.Series(x).rank().values
        yr = pd.Series(y).rank().values
        sr = float(np.corrcoef(xr, yr)[0,1])
        sp = np.nan
    return pr, pp, sr, sp

def plot_scatter_with_fit(x, y, subj_label: str, outpath: Path = None, show: bool = True):
    # Fit line
    m, b = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min(), x.max(), 200)
    yfit = m * xfit + b

    # Correlations
    pr, pp, sr, sp = compute_correlations(x, y)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=22, alpha=0.85)
    ax.plot(xfit, yfit, color="tab:red", lw=1.6, label=f"fit: y={m:.2f}x+{b:.2f}")
    ax.set_xlabel("pMP (median across epochs)")
    ax.set_ylabel("HFD (median across epochs)")
    title = f"Channel medians: pMP vs HFD (sub {subj_label})\n"
    title += f"Pearson r={pr:.2f}"
    if not np.isnan(pp):
        title += f" (p={pp:.2g})"
    title += f", Spearman ρ={sr:.2f}"
    if not np.isnan(sp):
        title += f" (p={sp:.2g})"
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=220)
        print(f"Saved figure → {outpath}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to per-epoch features CSV")
    ap.add_argument("--subject", default=None, help="Subject ID for titles (optional)")
    ap.add_argument("--save", type=int, default=1, help="If 1, save figure to reports/figures/")
    ap.add_argument("--no_show", type=int, default=0, help="If 1, do not display the figure window")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df, pmp_cols, hfd_cols = load_per_epoch(csv_path)
    chans, x, y = channel_medians(df, pmp_cols, hfd_cols)

    subj_label = args.subject or csv_path.stem.replace("sub-","").replace("_per_epoch_features","")
    out = Path(f"reports/figures/qa_pmp_vs_hfd_scatter_sub-{subj_label}.png") if args.save else None

    plot_scatter_with_fit(x, y, subj_label=subj_label, outpath=out, show=(args.no_show == 0))

    # Console summary
    print(f"Channels: {len(chans)}")
    print(f"pMP median range: {x.min():.3f} – {x.max():.3f} (median {np.median(x):.3f})")
    print(f"HFD median range: {y.min():.3f} – {y.max():.3f} (median {np.median(y):.3f})")

if __name__ == "__main__":
    main()
