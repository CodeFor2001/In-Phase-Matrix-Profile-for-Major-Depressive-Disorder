#!/usr/bin/env python3
"""
within_subject_features.py
Compute per-epoch pMP and HFD for one subject using existing code/features/ implementations.

Usage:
  python code/qa/within_subject_features.py --subject 2010002 --fs 200 --m_sec 1.0 --kmax 8
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def setup_imports():
    """Add project paths to allow importing from code.features"""
    # Get the project root directory (two levels up from this file)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # from code/qa/file.py to project root
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Also add the code directory specifically
    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

# Set up imports before importing our modules
setup_imports()

# Now import the feature functions
try:
    from features.pmp import pmp_epoch
    from features.hfd import hfd_epoch
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure pmp.py and hfd.py exist in code/features/")
    print("Current working directory:", Path.cwd())
    print("Python path:", sys.path[:3])  # Show first 3 entries
    sys.exit(1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="Subject ID, e.g., 2010002")
    p.add_argument("--fs", type=float, default=200.0, help="Sampling rate of epochs")
    p.add_argument("--m", type=int, default=200, help="pMP subsegment length in samples (overrides m_sec)")
    p.add_argument("--m_sec", type=float, default=1.0, help="pMP subsegment length in seconds")
    p.add_argument("--local_min_dist", type=int, default=20, help="pMP local minima separation")
    p.add_argument("--kmax", type=int, default=8, help="HFD kmax parameter")
    args = p.parse_args()

    # Load subject epochs
    npz_path = Path(f"data/interim/sub-{args.subject}_epochs.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}. Run preprocessing first.")
    
    d = np.load(npz_path, allow_pickle=True)
    epochs = d["epochs"]              # (n_epochs, n_channels, n_samples)
    channels = list(d["channels"])    # channel names
    subject_id = str(d.get("subject", args.subject))
    
    # Use m_sec to compute m if m not explicitly provided as non-default
    if args.m == 200 and args.m_sec != 1.0:  # Default m but custom m_sec
        m = int(args.m_sec * args.fs)
    else:
        m = args.m
    
    print(f"Processing subject {subject_id}: {epochs.shape[0]} epochs, {epochs.shape[1]} channels")
    print(f"Using pMP parameters: fs={args.fs}, m={m}, local_min_dist={args.local_min_dist}")
    print(f"Using HFD parameters: kmax={args.kmax}")
    
    # Compute features per epoch
    rows = []
    for e in range(epochs.shape[0]):
        row = {"epoch": e}
        ep = epochs[e]  # (n_channels, n_samples)
        
        for ch_i, ch_name in enumerate(channels):
            sig = ep[ch_i].astype(float)
            
            # Use existing implementations with same parameters as aggregation
            try:
                val_pmp = pmp_epoch(sig, fs=args.fs, m=m, local_min_dist=args.local_min_dist)
            except Exception as ex:
                print(f"Warning: pMP failed for epoch {e}, channel {ch_name}: {ex}")
                val_pmp = np.nan
            
            try:
                val_hfd = hfd_epoch(sig, kmax=args.kmax)
            except Exception as ex:
                print(f"Warning: HFD failed for epoch {e}, channel {ch_name}: {ex}")
                val_hfd = np.nan
            
            row[f"{ch_name}_pMP"] = val_pmp
            row[f"{ch_name}_HFD"] = val_hfd
        
        rows.append(row)
    
    # Save results
    df = pd.DataFrame(rows)
    out_dir = Path("data/qa")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sub-{args.subject}_per_epoch_features.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved per-epoch features → {out_path}")
    
    # Quick quality report
    numeric_cols = [c for c in df.columns if c != "epoch"]
    numeric_data = df[numeric_cols]
    n_nan = numeric_data.isna().sum().sum()
    n_inf = np.isinf(numeric_data.values).sum()
    n_total = numeric_data.size
    
    print(f"Quality check: {n_nan} NaNs, {n_inf} Infs out of {n_total} total values")
    
    if n_nan > 0:
        print("Channels with NaN values:")
        nan_summary = numeric_data.isna().sum()
        nan_channels = nan_summary[nan_summary > 0]
        for ch, count in nan_channels.items():
            print(f"  {ch}: {count}/{epochs.shape[0]} epochs")
    
    # Basic sanity check: show range of values
    pmp_cols = [c for c in numeric_cols if "_pMP" in c]
    hfd_cols = [c for c in numeric_cols if "_HFD" in c]
    
    if pmp_cols:
        pmp_data = numeric_data[pmp_cols].values.flatten()
        pmp_finite = pmp_data[np.isfinite(pmp_data)]
        if len(pmp_finite) > 0:
            print(f"pMP range: {pmp_finite.min():.3f} to {pmp_finite.max():.3f} (median: {np.median(pmp_finite):.3f})")
    
    if hfd_cols:
        hfd_data = numeric_data[hfd_cols].values.flatten()
        hfd_finite = hfd_data[np.isfinite(hfd_data)]
        if len(hfd_finite) > 0:
            print(f"HFD range: {hfd_finite.min():.3f} to {hfd_finite.max():.3f} (median: {np.median(hfd_finite):.3f})")

if __name__ == "__main__":
    main()
