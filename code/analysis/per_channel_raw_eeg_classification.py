#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from aeon.classification.convolution_based import MultiRocketClassifier
from aeon.classification.hybrid import HIVECOTEV2

# Configuration parameters
PROJ_ROOT = Path(__file__).resolve().parents[2]
NPZ_DIR = PROJ_ROOT / "data" / "interim"
LABELS_CSV = PROJ_ROOT / "data" / "raw" / "labels.csv"
OUT_DIR = PROJ_ROOT / "data" / "processed"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Output files
OUT_DETAILED = OUT_DIR / "per_channel_classification_detailed.csv"
OUT_SUMMARY = OUT_DIR / "per_channel_classification_summary.csv"

RANDOM_STATE = 42
N_SPLITS = 5
HC2_TIME_LIMIT = 10  # minutes

def load_data():
    # Load labels
    df_labels = pd.read_csv(LABELS_CSV)
    df_labels["subject"] = df_labels["subject"].astype(str).str.strip()
    subjects_set = set(df_labels["subject"])

    all_epochs = []
    all_subjects = []

    # Load epochs from npz
    for fpath in sorted(NPZ_DIR.glob("*.npz")):
        subj = fpath.name[len("sub-"):-len("_epochs.npz")]
        if subj not in subjects_set:
            print(f"Skipping subject {subj}, not in labels")
            continue
        data = np.load(fpath)
        epochs = data["epochs"]  # shape (n_epochs, n_channels, n_timepoints)
        all_epochs.append(epochs)
        all_subjects += [subj] * epochs.shape[0]

    if not all_epochs:
        raise RuntimeError("No valid data loaded.")

    X = np.concatenate(all_epochs, axis=0)  # (samples, channels, timepoints)
    subjects = np.array(all_subjects)

    # Create label array
    label_map = {row["subject"]: row["group"] for _, row in df_labels.iterrows()}
    y_labels = np.array([label_map[s] for s in subjects])

    # Label encoding
    classes = sorted(np.unique(y_labels))
    label_encoder = {c: i for i, c in enumerate(classes)}
    y = np.array([label_encoder[v] for v in y_labels])

    return X, y, subjects

def run_svm_cv(X, y, cv_type="loo", folds=5):
    if cv_type == "loo":
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    y_true_all, y_pred_all, y_prob_all = [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)

        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
        )
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        y_prob = clf.predict_proba(X_test_flat)[:, 1]

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    acc = accuracy_score(y_true_all, y_pred_all)
    bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    auc = roc_auc_score(y_true_all, y_prob_all) if len(np.unique(y_true_all)) == 2 else np.nan
    return acc, bal_acc, auc

def run_multirocket_cv(X, y, folds=5):
    # Input shape: (samples, channels, timepoints)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = MultiRocketClassifier(n_jobs=-1, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    acc = accuracy_score(y_true_all, y_pred_all)
    bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    return acc, bal_acc, np.nan  # ROC-AUC unavailable here

def run_hc2_cv(X, y, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = HIVECOTEV2(n_jobs=-1, random_state=RANDOM_STATE, time_limit_in_minutes=HC2_TIME_LIMIT)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    acc = accuracy_score(y_true_all, y_pred_all)
    bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    return acc, bal_acc, np.nan

def main():
    print("Loading data...")
    X, y, subjects = load_data()
    print(f"Data shape: {X.shape}, Samples: {len(y)}, Subjects: {len(np.unique(subjects))}")

    channels = list(range(X.shape[1]))
    results = []

    for ch_idx in channels:
        print(f"Processing channel {ch_idx +1}...")
        X_ch = X[:, ch_idx, :]  # shape (samples, timepoints)

        # check if both classes are present
        if len(np.unique(y)) < 2:
            print(f"Skipping channel {ch_idx+1} due to less than two classes present.")
            continue

        X_ch_expanded = X_ch[:, np.newaxis, :]  # make shape (samples, 1, timepoints) for classifier

        # Run classifiers per channel
        svm_loo_acc, svm_loo_bal, svm_loo_auc = run_svm_cv(X_ch_expanded, y, cv_type="loo")
        svm_kf_acc, svm_kf_bal, svm_kf_auc = run_svm_cv(X_ch_expanded, y, cv_type="kf", folds=5)
        mr_acc, mr_bal, mr_auc = run_multirocket_cv(X_ch_expanded, y)
        hc2_acc, hc2_bal, hc2_auc = run_hc2_cv(X_ch_expanded, y)

        results.append({
            "channel": ch_idx+1,
            "svm_loo_accuracy": svm_loo_acc,
            "svm_loo_balanced_accuracy": svm_loo_bal,
            "svm_loo_auc": svm_loo_auc,
            "svm_kfold_accuracy": svm_kf_acc,
            "svm_kfold_balanced_accuracy": svm_kf_bal,
            "svm_kfold_auc": svm_kf_auc,
            "multirocket_accuracy": mr_acc,
            "multirocket_balanced_accuracy": mr_bal,
            "hc2_accuracy": hc2_acc,
            "hc2_balanced_accuracy": hc2_bal
        })

    df_results = pd.DataFrame(results)
    out_csv = OUT_DIR / "per_channel_raw_eeg_classification.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"Saved per-channel classification results to: {out_csv}")

if __name__ == "__main__":
    main()
