#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Project paths (relative to this file: code/analysis/)
PROJ_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJ_ROOT / "data" / "processed" / "features_pmp_hfd.csv"
LABELS_CSV = PROJ_ROOT / "data" / "raw" / "labels.csv"
OUT_DIR = PROJ_ROOT / "data" / "processed"
OUT_PER_CHANNEL = OUT_DIR / "classify_results.csv"
OUT_SUMMARY = OUT_DIR / "classify_summary.csv"

FEATURES = ["pMP", "HFD"]  # evaluate each feature separately

def load_data():
    # Load aggregated features (one row per subject with ChX_pMP/ChX_HFD columns)
    df_feat = pd.read_csv(FEATURES_CSV)
    # Load labels with columns: subject, group
    df_lab = pd.read_csv(LABELS_CSV)

    # Normalize keys
    df_feat["subject"] = df_feat["subject"].astype(str).str.strip()
    df_lab["subject"] = df_lab["subject"].astype(str).str.strip()
    df_lab["group"] = df_lab["group"].astype(str).str.strip()

    # Align subjects
    keep = set(df_feat["subject"]).intersection(set(df_lab["subject"]))
    if len(keep) == 0:
        raise ValueError("No overlapping subjects between features and labels.")

    df_feat = df_feat[df_feat["subject"].isin(keep)].copy()
    df_lab = df_lab[df_lab["subject"].isin(keep)].copy()

    # Merge labels
    df = df_feat.merge(df_lab[["subject", "group"]], on="subject", how="inner")
    # Ensure deterministic subject order
    df = df.sort_values("subject").reset_index(drop=True)
    return df

def discover_channels(df, features):
    # Channels are encoded as columns like Ch1_pMP, Ch1_HFD ...
    chans = set()
    for col in df.columns:
        if not col.startswith("Ch"):
            continue
        # column format: Ch{idx}_{feat}
        try:
            ch_prefix, feat = col.split("_", 1)
        except ValueError:
            continue
        if feat in features:
            # strip "Ch" and parse index
            if ch_prefix[2:].isdigit():
                chans.add(int(ch_prefix[2:]))
    return sorted(chans)

def prepare_single_channel_matrix(df, ch_idx, feat_name):
    col = f"Ch{ch_idx}_{feat_name}"
    if col not in df.columns:
        # Return None to indicate this channel doesn't have this feature
        return None, None, None
    # X is (n_subjects x 1)
    X = df[[col]].values.astype(float)
    # y is binary encoded group
    groups = df["group"].values
    # Map groups to 0/1 with stable order: control -> 0 if present; else alphabetical
    uniq = sorted(pd.unique(groups).tolist())
    if "control" in uniq and "MDD" in uniq:
        order = ["control", "MDD"]
    else:
        order = uniq
    mapping = {g: i for i, g in enumerate(order)}
    y = np.array([mapping[g] for g in groups], dtype=int)
    return X, y, mapping

def loo_linear_svm_accuracy(X, y):
    # Scale then linear SVM; fit inside each fold for proper CV.
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        model = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                              SVC(kernel="linear", C=1.0))
        model.fit(X[train_idx], y[train_idx])
        y_hat = model.predict(X[test_idx])
        y_true.append(y[test_idx][0])
        y_pred.append(y_hat[0])
    return accuracy_score(y_true, y_pred)

def main():
    df = load_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    channels = discover_channels(df, FEATURES)
    if not channels:
        raise ValueError("No channel feature columns found (Ch{idx}_pMP/HFD).")

    per_channel_rows = []
    summary_rows = []

    for feat in FEATURES:
        best_ch = None
        best_acc = -np.inf
        for ch in channels:
            X, y, mapping = prepare_single_channel_matrix(df, ch, feat)
            if X is None:
                # Skip channels missing this feature column
                continue
            # If all values are NaN in this channel, skip
            if not np.isfinite(X).any():
                continue
            # Drop rows with NaN for this channel
            mask = np.isfinite(X[:, 0])
            X_ch = X[mask]
            y_ch = y[mask]
            # Need at least two subjects and both classes to do LOO meaningfully
            if X_ch.shape[0] < 2 or len(np.unique(y_ch)) < 2:
                acc = np.nan
            else:
                acc = loo_linear_svm_accuracy(X_ch, y_ch)

            per_channel_rows.append({
                "feature": feat,
                "channel": ch,
                "n_subjects_used": int(X_ch.shape[0]),
                "n_class0": int(np.sum(y_ch == 0)),
                "n_class1": int(np.sum(y_ch == 1)),
                "accuracy_loo": None if np.isnan(acc) else float(acc),
            })

            if np.isfinite(acc) and acc > best_acc:
                best_acc = acc
                best_ch = ch

        summary_rows.append({
            "feature": feat,
            "best_channel": best_ch,
            "best_accuracy_loo": None if not np.isfinite(best_acc) else float(best_acc),
        })

    # Save outputs
    pd.DataFrame(per_channel_rows).sort_values(["feature", "channel"]).to_csv(OUT_PER_CHANNEL, index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

    # Console report
    print("Classification complete.")
    print(f"Saved per-channel results: {OUT_PER_CHANNEL}")
    print(f"Saved summary: {OUT_SUMMARY}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
