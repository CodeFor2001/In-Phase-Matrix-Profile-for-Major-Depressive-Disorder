#!/usr/bin/env python3

"""
aggregate.py - aggregate per-subject median pMP and HFD features (multi-process, incremental save)

Outputs:
 - CSV with one row per subject, columns for each channel's pMP and HFD
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.paths import load_settings, get_path
from utils.repro import log_run
from features import pmp, hfd

def aggregate_subject(npz_path, fs, m, local_min_dist, kmax):
    data = np.load(npz_path, allow_pickle=True)
    epochs = data["epochs"]
    channels = data.get("channels", [f"Ch{i+1}" for i in range(epochs.shape[1])])
    subject = str(data.get("subject", npz_path.stem.split('_')[0]))
    
    n_epochs, n_ch = epochs.shape[:2]
    pmp_vals = np.full((n_epochs, n_ch), np.nan)
    hfd_vals = np.full((n_epochs, n_ch), np.nan)

    # Per-epoch/channel feature computation
    for e in range(n_epochs):
        for ch in range(n_ch):
            sig = epochs[e, ch]
            pmp_vals[e, ch] = pmp.pmp_epoch(sig, fs=fs, m=m, local_min_dist=local_min_dist)
            hfd_vals[e, ch] = hfd.hfd_epoch(sig, kmax=kmax)
    
    row = {"subject": subject}
    for i, ch in enumerate(channels):
        row[f"{ch}_pMP"] = np.nanmedian(pmp_vals[:, i])
        row[f"{ch}_HFD"] = np.nanmedian(hfd_vals[:, i])
    return row

def main():
    log_run("aggregate")
    cfg = load_settings()
    fs = cfg["sampling"]["target_fs"]
    m = cfg["pmp"]["m"]
    local_min_dist = cfg["pmp"]["local_min_dist"]
    kmax = cfg["hfd"]["kmax"]

    interim = get_path("interim")
    processed = get_path("processed", mkdir=True)

    npz_files = sorted(interim.glob("sub-*_epochs.npz"))
    if not npz_files:
        print("❌ No files found.")
        return

    out_csv = processed / "features_pmp_hfd.csv"
    if out_csv.exists():
        df_existing = pd.read_csv(out_csv)
        processed_subjects = set(df_existing["subject"])
    else:
        df_existing = pd.DataFrame()
        processed_subjects = set()

    rows = [row for _, row in df_existing.iterrows()]  # existing data

    with ProcessPoolExecutor() as executor:
        futures = {}
        for f in npz_files:
            try:
                data = np.load(f, allow_pickle=True)
                subj_id = str(data.get("subject", f.stem.split('_')[0]))
            except Exception as e:
                print(f"Skipping {f.name} due to load error: {e}")
                continue
            if subj_id in processed_subjects:
                print(f"Skipping {subj_id} (already processed)")
                continue
            futures[executor.submit(aggregate_subject, f, fs, m, local_min_dist, kmax)] = subj_id

        for fut in as_completed(futures):
            subj_id = futures[fut]
            try:
                result = fut.result()
                rows.append(result)
                # Save progress after each subject
                df_out = pd.DataFrame(rows)
                df_out.to_csv(out_csv, index=False)
                processed_subjects.add(subj_id)
                print(f"Processed {subj_id}. Saved progress.")
            except Exception as e:
                print(f"Failed processing {subj_id}: {e}")

    print(f"Aggregation complete. Total subjects processed: {len(rows)}")

if __name__ == "__main__":
    main()
