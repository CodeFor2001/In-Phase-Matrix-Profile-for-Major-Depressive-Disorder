#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

# Project paths (relative to this script)
PROJ_ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["pMP", "HFD", "pmp+hfd"]  # Added combined feature
FEATURES_CSV = PROJ_ROOT / "data" / "processed" / "features_pmp_hfd.csv"
LABELS_CSV = PROJ_ROOT / "data" / "raw" / "labels.csv"
OUT_DIR = PROJ_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTFILE_DETAIL = OUT_DIR / "classify_combined_results.csv"
OUTFILE_SUMMARY = OUT_DIR / "classify_combined_summary.csv"

def load_data():
    df_feat = pd.read_csv(FEATURES_CSV)
    df_lab = pd.read_csv(LABELS_CSV)
    df_feat["subject"] = df_feat["subject"].astype(str).str.strip()
    df_lab["subject"] = df_lab["subject"].astype(str).str.strip()
    df_lab["group"] = df_lab["group"].astype(str).str.strip()
    common = set(df_feat["subject"]).intersection(set(df_lab["subject"]))
    if not common:
        raise ValueError("No overlapping subjects between feature and label files.")
    df_feat = df_feat[df_feat["subject"].isin(common)].copy()
    df_lab = df_lab[df_lab["subject"].isin(common)].copy()
    df = pd.merge(df_feat, df_lab[["subject", "group"]], on="subject", how="inner")
    df = df.sort_values(by="subject").reset_index(drop=True)
    return df

def discover_channels(df, features):
    chans = set()
    for col in df.columns:
        if not col.startswith("Ch"):
            continue
        try:
            ch_prefix, feat = col.split("_", 1)
        except ValueError:
            continue
        if feat in features:
            try:
                chans.add(int(ch_prefix[2:]))
            except:
                continue
    return sorted(chans)

def prepare_feature_matrix(df, ch_idx, feature_name):
    if feature_name == "pmp+hfd":
        col1 = f"Ch{ch_idx}_pMP"
        col2 = f"Ch{ch_idx}_HFD"
        cols = []
        if col1 in df.columns:
            cols.append(col1)
        if col2 in df.columns:
            cols.append(col2)
        if not cols:
            # No combined features for this channel
            return None, None
        X = df[cols].values.astype(np.float64)
    else:
        col = f"Ch{ch_idx}_{feature_name}"
        if col not in df.columns:
            return None, None
        X = df[[col]].values.astype(np.float64)

    y_labels = df["group"].values
    classes = sorted(np.unique(y_labels))
    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[v] for v in y_labels])

    return X, y

def loo_svm_accuracy(X, y):
    if X is None or len(X) == 0 or len(np.unique(y)) < 2:
        return np.nan
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            SVC(kernel="linear", C=1.0)
        )
        model.fit(X[train_idx], y[train_idx])
        yhat = model.predict(X[test_idx])
        y_true.append(y[test_idx][0])
        y_pred.append(yhat[0])
    return accuracy_score(y_true, y_pred)

def main():
    df = load_data()
    channels = discover_channels(df, FEATURES)
    detail_rows = []
    summary_rows = []
    for feat in FEATURES:
        best_acc = -np.inf
        best_ch = None
        for ch in channels:
            X, y = prepare_feature_matrix(df, ch, feat)
            if X is None:
                continue
            if not np.isfinite(X).any():
                continue
            mask = np.isfinite(X).all(axis=1)
            X_clean = X[mask]
            y_clean = y[mask]
            if len(y_clean) < 2 or len(np.unique(y_clean)) < 2:
                acc = np.nan
            else:
                acc = loo_svm_accuracy(X_clean, y_clean)
            detail_rows.append({
                "feature": feat,
                "channel": ch,
                "n_subjects": int(len(y_clean)),
                "n_class0": int(np.sum(y_clean == 0)),
                "n_class1": int(np.sum(y_clean == 1)),
                "accuracy_loo": None if np.isnan(acc) else float(acc)
            })
            if np.isfinite(acc) and acc > best_acc:
                best_acc = acc
                best_ch = ch
        summary_rows.append({
            "feature": feat,
            "best_channel": best_ch,
            "best_accuracy_loo": None if np.isnan(best_acc) else float(best_acc)
        })
    pd.DataFrame(detail_rows).to_csv(OUTFILE_DETAIL, index=False)
    pd.DataFrame(summary_rows).to_csv(OUTFILE_SUMMARY, index=False)
    print(f"Saved detailed results to {OUTFILE_DETAIL}")
    print(f"Saved summary results to {OUTFILE_SUMMARY}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
