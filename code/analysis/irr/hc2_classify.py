#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from aeon.classification.hybrid import HIVECOTEV2  # HC2 classifier

# Paths to your data files (adjust if needed)
PROJ_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJ_ROOT / "data" / "processed" / "features_pmp_hfd.csv"
LABELS_CSV = PROJ_ROOT / "data" / "raw" / "labels.csv"
OUT_DIR = PROJ_ROOT / "data" / "processed"
OUT_CSV = OUT_DIR / "hc2_results.csv"

RESTRICT_TO_CHANNELS = None  # or e.g. [1,2,...,30]

RANDOM_STATE = 42
N_SPLITS = 5
TIME_LIMIT_MINUTES = 10  # limit HC2 runtime per fold
N_JOBS = -1  # use all available CPU cores

def load_data():
    df_feat = pd.read_csv(FEATURES_CSV)
    df_lab = pd.read_csv(LABELS_CSV)

    df_feat["subject"] = df_feat["subject"].astype(str).str.strip()
    df_lab["subject"] = df_lab["subject"].astype(str).str.strip()
    df_lab["group"] = df_lab["group"].astype(str).str.strip()

    keep = sorted(set(df_feat["subject"]).intersection(set(df_lab["subject"])))
    if not keep:
        raise ValueError("No overlapping subjects between features and labels.")

    df = df_feat[df_feat["subject"].isin(keep)].merge(
        df_lab[["subject", "group"]], on="subject", how="inner"
    )
    df = df.sort_values("subject").reset_index(drop=True)
    return df

def select_columns(df, feature_name, restrict_channels=None):
    cols = []
    if restrict_channels is None:
        for col in df.columns:
            if col.startswith("Ch") and col.endswith(f"_{feature_name}"):
                cols.append(col)
    else:
        for ch in restrict_channels:
            col = f"Ch{ch}_{feature_name}"
            if col in df.columns:
                cols.append(col)
    return cols

def build_design_matrices(df, restrict_channels):
    pmp_cols = select_columns(df, "pMP", restrict_channels)
    hfd_cols = select_columns(df, "HFD", restrict_channels)

    X_pmp = df[pmp_cols].to_numpy(dtype=np.float64) if pmp_cols else None
    X_hfd = df[hfd_cols].to_numpy(dtype=np.float64) if hfd_cols else None

    if X_pmp is not None and X_hfd is not None:
        X_both = np.hstack([X_pmp, X_hfd])
    elif X_pmp is not None:
        X_both = X_pmp
    elif X_hfd is not None:
        X_both = X_hfd
    else:
        raise ValueError("No pMP/HFD columns found.")

    y_labels = df["group"].values
    uniq = sorted(pd.unique(y_labels).tolist())
    if "control" in uniq and "MDD" in uniq:
        order = ["control", "MDD"]
    else:
        order = uniq
    label_map = {g: i for i, g in enumerate(order)}
    y = np.array([label_map[g] for g in y_labels], dtype=int)

    mats = []
    if X_pmp is not None:
        mats.append(("pMP", X_pmp))
    if X_hfd is not None:
        mats.append(("HFD", X_hfd))
    mats.append(("pMP+HFD", X_both))

    return mats, y

def clean_matrix(X, y):
    col_mask = np.isfinite(X).any(axis=0)
    X1 = X[:, col_mask].astype(np.float64, copy=False)
    var = np.nanvar(X1, axis=0)
    col_mask2 = np.asarray(var, dtype=np.float64).reshape(-1) > 0.0
    X2 = X1[:, col_mask2]
    row_mask = np.isfinite(X2).all(axis=1)
    X3 = X2[row_mask]
    y3 = y[row_mask]
    return X3, y3

def to_aeon_panel(X2d):
    # Repeat static features along time axis for HC2 convolution kernels
    n_cases, n_channels = X2d.shape
    min_time_points = 10
    X_expanded = np.repeat(X2d[:, :, np.newaxis], min_time_points, axis=2).astype(np.float64, copy=False)
    return X_expanded

def evaluate_cv_hc2(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = HIVECOTEV2(
            time_limit_in_minutes=TIME_LIMIT_MINUTES,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=1,
        )
        clf.fit(X_tr, y_tr)
        y_hat = clf.predict(X_te)

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_hat.tolist())

    acc = accuracy_score(y_true_all, y_pred_all)
    bacc = balanced_accuracy_score(y_true_all, y_pred_all)
    auc = None
    # HC2 might not support predict_proba by default reliably, so skip AUROC here
    return acc, bacc, auc

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    mats, y = build_design_matrices(df, RESTRICT_TO_CHANNELS)
    rows = []

    for name, X in mats:
        print(f"[DEBUG] {name} raw 2D shape: {X.shape}")
        X_clean, y_clean = clean_matrix(X, y)
        n_rows, n_cols = X_clean.shape
        print(f"[DEBUG] {name} cleaned 2D shape: {X_clean.shape}")

        if n_rows < 5 or n_cols < 1 or len(np.unique(y_clean)) < 2:
            print(f"Skipping {name}: insufficient data after cleaning (n={n_rows}, d={n_cols}).")
            continue

        X_panel = to_aeon_panel(X_clean)
        print(f"[DEBUG] {name} panel 3D shape for aeon: {X_panel.shape}")

        acc, bacc, auc = evaluate_cv_hc2(X_panel, y_clean)
        rows.append({
            "feature_set": name,
            "n_subjects": int(n_rows),
            "n_channels_as_features": int(n_cols),
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "roc_auc": None if auc is None else float(auc),
        })
        print(f"{name} [HC2]: acc={acc:.3f}, bacc={bacc:.3f}, auc={auc if auc is not None else 'NA'}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
