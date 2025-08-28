#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import re
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# Paths configuration (adjust if needed)
PROJ_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJ_ROOT / "data" / "processed" / "features_pmp_hfd.csv"
LABELS_CSV = PROJ_ROOT / "data" / "raw" / "labels.csv"
OUT_DIR = PROJ_ROOT / "data" / "processed"
OUT_PER_CHANNEL = OUT_DIR / "classification_kfold_channelwise.csv"
OUT_SUMMARY = OUT_DIR / "classification_kfold_summary.csv"

FEATURES = ["pMP", "HFD"]
N_FOLDS = 5
RANDOM_STATE = 42

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
    chan_set = set()
    pat = re.compile(r"^Ch(\d+)_(\w+)$", re.I)
    for col in df.columns:
        m = pat.match(col.strip())
        if m and m.group(2) in features:
            chan_set.add(int(m.group(1)))
    return sorted(chan_set)

def prepare_channel_feature_matrix(df, ch_idx, feat_name):
    col = f"Ch{ch_idx}_{feat_name}"
    if col not in df.columns:
        return None, None

    X = df[[col]].values.astype(np.float64)
    y_labels = df["group"].values

    # Map group labels to binary 0/1
    classes = sorted(np.unique(y_labels))
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    y = np.array([mapping[label] for label in y_labels], dtype=int)
    return X, y

def prepare_combined_features(df, ch_idx):
    cols_pmp = f"Ch{ch_idx}_pMP"
    cols_hfd = f"Ch{ch_idx}_HFD"

    if cols_pmp not in df.columns and cols_hfd not in df.columns:
        return None, None

    features = []
    if cols_pmp in df.columns:
        features.append(df[[cols_pmp]].values.astype(np.float64))
    if cols_hfd in df.columns:
        features.append(df[[cols_hfd]].values.astype(np.float64))
    X = np.hstack(features)

    y_labels = df["group"].values
    classes = sorted(np.unique(y_labels))
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    y = np.array([mapping[label] for label in y_labels], dtype=int)
    return X, y

def run_kfold_classification(X, y, n_folds=N_FOLDS, random_state=RANDOM_STATE):
    if X is None or y is None or len(np.unique(y)) < 2 or len(y) < 5:
        return [], np.nan, np.nan, np.nan  # insufficient data

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    y_true_all, y_pred_all, y_proba_all = [], [], []

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=random_state))
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else np.nan

        fold_results.append({"fold": fold_idx, "accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc})

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_proba_all.extend(y_proba)

    # Overall aggregated
    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    overall_auc = roc_auc_score(y_true_all, y_proba_all) if len(np.unique(y_true_all)) == 2 else np.nan

    return fold_results, overall_acc, overall_bal_acc, overall_auc

def main():
    df = load_data()
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    channels = discover_channels(df, FEATURES)

    all_fold_rows = []
    summary_rows = []

    for feat in FEATURES + ["combined"]:
        best_acc = -np.inf
        best_ch = None

        for ch in channels:
            if feat == "combined":
                X, y = prepare_combined_features(df, ch)
            else:
                X, y = prepare_channel_feature_matrix(df, ch, feat)

            if X is None or len(X) == 0 or len(np.unique(y)) < 2:
                continue

            valid_mask = np.isfinite(X).all(axis=1)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]

            if len(y_valid) < 5:
                continue

            fold_results, acc, bal_acc, auc = run_kfold_classification(X_valid, y_valid)

            for fr in fold_results:
                all_fold_rows.append({
                    "feature": feat,
                    "channel": ch,
                    "fold": fr["fold"],
                    "accuracy": fr["accuracy"],
                    "balanced_accuracy": fr["balanced_accuracy"],
                    "auc": fr["auc"],
                    "samples": len(y_valid)
                })

            if acc > best_acc:
                best_acc = acc
                best_ch = ch

        summary_rows.append({
            "feature": feat,
            "best_channel": best_ch,
            "best_accuracy": best_acc
        })

    pd.DataFrame(all_fold_rows).to_csv(OUT_DIR / "classification_kfold_folds_channelwise.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "classification_kfold_summary.csv", index=False)

    print(f"Saved per-fold, per-channel results to 'classification_kfold_folds_channelwise.csv'")
    print(f"Saved summary results to 'classification_kfold_summary.csv'")

if __name__ == "__main__":
    main()
