#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# Project paths (adapt if needed)
PROJ_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJ_ROOT / "data" / "processed" / "features_pmp_hfd.csv"
LABELS_CSV = PROJ_ROOT / "data" / "raw" / "labels.csv"
OUT_DIR = PROJ_ROOT / "data" / "processed"
OUT_CSV = OUT_DIR / "svm_results.csv"

# Optionally restrict to a specific channel index set (e.g., the 30 from a paper)
RESTRICT_TO_CHANNELS = None  # e.g., [1,2,...,30]

RANDOM_STATE = 42
N_SPLITS = 5

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

def evaluate_cv(X, y, random_state=RANDOM_STATE, n_splits=N_SPLITS):
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=1.0, probability=True, random_state=random_state)
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_true_all, y_pred_all, y_proba_all = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_hat.tolist())

        if len(np.unique(y)) == 2:
            probs = model.predict_proba(X_te)[:, 1]
            y_proba_all.extend(probs.tolist())

    acc = accuracy_score(y_true_all, y_pred_all)
    bacc = balanced_accuracy_score(y_true_all, y_pred_all)
    auc = None
    if len(np.unique(y)) == 2 and len(y_proba_all) == len(y_true_all):
        try:
            proba_vec = np.asarray(y_proba_all, dtype=np.float64)
            # Ensure 1D vector, no tuple comparisons
            if proba_vec.ndim == 1 and proba_vec.shape[0] == len(y_true_all):
                auc = roc_auc_score(y_true_all, proba_vec)
        except Exception:
            auc = None
    return acc, bacc, auc

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    mats, y = build_design_matrices(df, RESTRICT_TO_CHANNELS)
    rows = []
    for name, X in mats:
        print(f"[DEBUG] {name} raw shape: {X.shape}, dtype: {X.dtype}")

        # Drop columns with all-NaN
        col_mask = np.isfinite(X).any(axis=0)
        X1 = X[:, col_mask].astype(np.float64, copy=False)

        # Drop zero-variance columns
        var = np.nanvar(X1, axis=0)
        var = np.asarray(var, dtype=np.float64).reshape(-1)
        col_mask2 = var > 0.0
        X2 = X1[:, col_mask2]

        # Drop rows with any NaN remaining
        row_mask = np.isfinite(X2).all(axis=1)
        X3 = X2[row_mask]
        y3 = y[row_mask]

        print(f"[DEBUG] {name} cleaned shape: {X3.shape}, y shape: {y3.shape}")

        # Safe shape indexing
        n_rows = int(X3.shape[0]) if X3.ndim >= 1 else 0
        n_cols = int(X3.shape[1]) if X3.ndim == 2 else 0

        if n_rows < 5 or n_cols < 1 or len(np.unique(y3)) < 2:
            print(f"Skipping {name}: insufficient data after cleaning (n={n_rows}, d={n_cols}).")
            continue

        acc, bacc, auc = evaluate_cv(X3, y3)
        rows.append({
            "feature_set": name,
            "n_subjects": n_rows,
            "n_features": n_cols,
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "roc_auc": None if auc is None else float(auc),
        })
        print(f"{name}: acc={acc:.3f}, bacc={bacc:.3f}, auc={auc if auc is not None else 'NA'}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
