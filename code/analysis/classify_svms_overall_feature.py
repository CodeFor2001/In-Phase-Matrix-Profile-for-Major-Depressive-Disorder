#!/usr/bin/env python3

"""
Classification with SVM (LOO and KFold) on EEG feature sets (pMP, HFD, pmp+hfd)
Saves combined per-subject feature tables and classification metrics.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ---- Project Path Setup ----
PROJ_ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["pMP", "HFD", "pmp+hfd"]  # Features to use, plus combined

FEATURES_CSV = PROJ_ROOT / "data" / "processed" / "features_pmp_hfd.csv"
LABELS_CSV = PROJ_ROOT / "data" / "raw" / "labels.csv"

OUT_DIR = PROJ_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTFILE_FEATURES = OUT_DIR / "classify_combined_all_channels_SVM.csv"
OUTFILE_METRICS = OUT_DIR / "classify_combined_SVM_metrics.csv"


# ---- Load and Prepare Data ----

def load_data():
    """Loads and aligns feature and label data on 'subject'."""
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


def combine_features_all_channels(df, features):
    """
    Combines features from all channels for each subject, for all feature sets.
    Returns a tidy dataframe with columns for each channel-feature combination + metadata.
    """
    subjects = df["subject"].unique()
    combined_rows = []
    for subject in subjects:
        sub_df = df[df["subject"] == subject]
        row = {"subject": subject}
        for feat in features:
            if feat == "pmp+hfd":
                pmp_cols = [col for col in sub_df.columns if col.endswith("_pMP")]
                hfd_cols = [col for col in sub_df.columns if col.endswith("_HFD")]
                all_cols = pmp_cols + hfd_cols
                vals = sub_df[all_cols].values.flatten()
                row[feat] = list(vals)
            else:
                feat_cols = [col for col in sub_df.columns if col.endswith(f"_{feat}")]
                vals = sub_df[feat_cols].values.flatten()
                row[feat] = list(vals)
        row["group"] = sub_df["group"].iloc[0]
        combined_rows.append(row)

    df_combined = pd.DataFrame(combined_rows)
    expanded_cols = []
    for feat in features:
        max_len = max(df_combined[feat].apply(len))
        feat_df = pd.DataFrame(
            df_combined[feat].tolist(),
            columns=[f"{feat}_ch{i+1}" for i in range(max_len)],
            index=df_combined.index,
        )
        expanded_cols.append(feat_df)

    df_final = pd.concat([df_combined[["subject", "group"]]] + expanded_cols, axis=1)
    return df_final


# ---- SVM Model Functions ----

def loo_svm_metrics(X, y):
    """Performs Leave-One-Out SVM and returns accuracy, F1, and AUC."""
    if X is None or len(X) == 0 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan

    loo = LeaveOneOut()
    y_true, y_pred, y_score = [], [], []

    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(kernel="linear", C=1.0, probability=True)
    )

    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        yhat = model.predict(X[test_idx])
        yprob = model.predict_proba(X[test_idx])[:, 1]
        y_true.append(y[test_idx][0])
        y_pred.append(yhat[0])
        y_score.append(yprob[0])

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan  # If AUC cannot be computed due to label issues
    return acc, f1, auc


def kfold_svm_metrics(X, y, n_splits=5):
    """Performs 5-Fold CV SVM and returns accuracy, F1, and AUC."""
    if X is None or len(X) == 0 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_true, y_pred, y_score = [], [], []

    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(kernel="linear", C=1.0, probability=True)
    )

    for train_idx, test_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        yhat = model.predict(X[test_idx])
        yprob = model.predict_proba(X[test_idx])[:, 1]
        y_true += list(y[test_idx])
        y_pred += list(yhat)
        y_score += list(yprob)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan
    return acc, f1, auc


# ---- Main Workflow ----

def main():
    # Load data and map labels to integers
    df = load_data()
    df_combined = combine_features_all_channels(df, FEATURES)

    feature_cols = [col for col in df_combined.columns if col not in ("subject", "group")]
    X = df_combined[feature_cols].values.astype(np.float64)
    y_labels = df_combined["group"].values
    classes = sorted(np.unique(y_labels))
    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[v] for v in y_labels])

    # Handle NaNs
    mask = np.isfinite(X).all(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]

    # Compute metrics for both CV types
    acc_loo, f1_loo, auc_loo = loo_svm_metrics(X_clean, y_clean)
    acc_kf, f1_kf, auc_kf = kfold_svm_metrics(X_clean, y_clean, n_splits=5)

    # Save combined feature table
    df_combined.to_csv(OUTFILE_FEATURES, index=False)

    # Save metrics for both CV methods
    metrics_df = pd.DataFrame([
        {
            "cv_type": "LOO",
            "accuracy": acc_loo,
            "f1_score": f1_loo,
            "auc": auc_loo,
            "n_subjects": len(y_clean),
            "n_class0": int(np.sum(y_clean == 0)),
            "n_class1": int(np.sum(y_clean == 1))
        },
        {
            "cv_type": "KFold",
            "accuracy": acc_kf,
            "f1_score": f1_kf,
            "auc": auc_kf,
            "n_subjects": len(y_clean),
            "n_class0": int(np.sum(y_clean == 0)),
            "n_class1": int(np.sum(y_clean == 1))
        }
    ])
    metrics_df.to_csv(OUTFILE_METRICS, index=False)

    # Console output
    print(f"Saved combined features for all channels to {OUTFILE_FEATURES}")
    print(f"Saved classification metrics to {OUTFILE_METRICS}\n")
    print(f"LOO SVM accuracy: {acc_loo:.4f}, F1 score: {f1_loo:.4f}, AUC: {auc_loo if not np.isnan(auc_loo) else 'nan'}")
    print(f"KFold SVM accuracy: {acc_kf:.4f}, F1 score: {f1_kf:.4f}, AUC: {auc_kf if not np.isnan(auc_kf) else 'nan'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
