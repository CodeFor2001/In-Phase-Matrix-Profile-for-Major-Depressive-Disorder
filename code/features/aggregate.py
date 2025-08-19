#!/usr/bin/env python3
"""
aggregate.py - aggregate per-epoch pMP and HFD features to per-subject medians.

For each subject:
- Load the saved .npz file with shape: (n_epochs, n_channels, epoch_len_samples)
- Compute pMP and HFD for each epoch & channel
- Median across epochs -> one pMP and one HFD value per channel
- Save results as CSV in data/processed/

Output format:
subject,ch_01_pMP,ch_02_pMP,...,ch_XX_pMP,ch_01_HFD,...,ch_XX_HFD
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# make project root importable (utils/, features/)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.paths import load_settings, get_path
from utils.repro import log_run
from features.pmp import pmp_epoch
from features.hfd import hfd_epoch


def aggregate_subject(npz_path, fs, m, local_min_dist, kmax):
    """
    Compute per-channel median pMP and HFD across all epochs for one subject file.
    npz_path: Path to .npz file containing 'epochs' array.
    Returns subject_id (str), channel_labels (list), features dict.
    """
    data = np.load(npz_path, allow_pickle=True)
    epochs = data["epochs"]  # shape: (n_epochs, n_channels, n_samples)
    channels = list(data.get("channels", [f"Ch{i+1}" for i in range(epochs.shape[1])]))
    subject_id = str(data.get("subject", Path(npz_path).stem))

    n_epochs, n_ch, _ = epochs.shape
    pmp_vals = np.zeros((n_epochs, n_ch))
    hfd_vals = np.zeros((n_epochs, n_ch))

    for e_idx in range(n_epochs):
        for ch_idx in range(n_ch):
            sig = epochs[e_idx, ch_idx]
            pmp_vals[e_idx, ch_idx] = pmp_epoch(sig, fs=fs, m=m, local_min_dist=local_min_dist)
            hfd_vals[e_idx, ch_idx] = hfd_epoch(sig, kmax=kmax)

    median_pmp = np.nanmedian(pmp_vals, axis=0)
    median_hfd = np.nanmedian(hfd_vals, axis=0)

    feat_dict = {f"{ch}_pMP": median_pmp[i] for i, ch in enumerate(channels)}
    feat_dict.update({f"{ch}_HFD": median_hfd[i] for i, ch in enumerate(channels)})

    return subject_id, channels, feat_dict


def main():
    log_run("aggregate_features")
    cfg = load_settings()
    fs = cfg["sampling"]["target_fs"]
    m = cfg["pmp"]["m"]
    local_min_dist = cfg["pmp"]["local_min_dist"]
    kmax = cfg["hfd"]["kmax"]

    interim_dir = get_path("interim")
    processed_dir = get_path("processed", mkdir=True)

    npz_files = sorted(interim_dir.glob("sub-*_epochs.npz"))
    if not npz_files:
        print(f"❌ No epoch files found in {interim_dir}")
        return

    rows = []
    for npz_file in npz_files:
        print(f"[INFO] Aggregating features for {npz_file.name} ...")
        sid, channels, feat_dict = aggregate_subject(
            npz_file, fs=fs, m=m, local_min_dist=local_min_dist, kmax=kmax
        )
        row = {"subject": sid}
        row.update(feat_dict)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = processed_dir / "features_pmp_hfd.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved aggregated features for {len(df)} subjects to {out_csv}")


if __name__ == "__main__":
    main()
