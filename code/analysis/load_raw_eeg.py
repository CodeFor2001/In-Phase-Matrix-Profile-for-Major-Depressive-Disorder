#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from aeon.classification.convolution_based import MultiRocketHydraClassifier

def load_raw_eeg_epochs(npz_folder: str, labels_csv: str):
    labels_df = pd.read_csv(labels_csv)
    labels_df["subject"] = labels_df["subject"].astype(str).str.strip()

    all_epochs = []
    all_labels = []
    all_subjects = []

    npz_folder = Path(npz_folder)
    for npz_file in sorted(npz_folder.glob("*.npz")):
        fname = npz_file.name
        if fname.startswith("sub-") and fname.endswith("_epochs.npz"):
            subject = fname[len("sub-"):-len("_epochs.npz")]
        else:
            data = np.load(npz_file)
            raw_subject = data["subject"]
            subject = str(raw_subject.item() if np.ndim(raw_subject) == 0 else raw_subject[0])

        label = labels_df.loc[labels_df["subject"] == subject, "group"].values
        if label.size == 0:
            print(f"Warning: No label found for subject {subject}, skipping.")
            continue

        data = np.load(npz_file)
        epochs = data["epochs"]  # (n_epochs, n_channels, n_timepoints)
        labels_repeated = np.repeat(label[0], epochs.shape[0])
        subjects_repeated = np.repeat(subject, epochs.shape[0])

        all_epochs.append(epochs)
        all_labels.append(labels_repeated)
        all_subjects.append(subjects_repeated)

    if not all_epochs:
        raise RuntimeError("No data loaded. Check label matching!")

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels)
    subjects = np.concatenate(all_subjects)
    return X, y, subjects

def svm_raw_epoch_classify(X, y, groups, n_splits=5):
    print("Running SVM with group aware CV...")
    gkf = GroupKFold(n_splits=n_splits)

    y_true, y_pred, y_proba = [], [], []
    X_flat = X.reshape(X.shape[0], -1)

    for train_idx, test_idx in gkf.split(X_flat, y, groups):
        X_tr, X_te = X_flat[train_idx], X_flat[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        y_true.extend(y_te)
        y_pred.extend(y_hat)
        y_proba.extend(y_prob)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else None
    print(f"SVM Accuracy: {acc:.3f}, Balanced Accuracy: {bacc:.3f}, AUROC: {auc if auc else 'NA'}")

def mrhydra_raw_epoch_classify(X, y, groups, n_splits=5):
    print("Running MultiRocket-Hydra with group aware CV...")
    gkf = GroupKFold(n_splits=n_splits)

    y_true, y_pred = [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = MultiRocketHydraClassifier(n_jobs=-1, random_state=42)
        clf.fit(X_tr, y_tr)
        y_hat = clf.predict(X_te)

        y_true.extend(y_te)
        y_pred.extend(y_hat)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    print(f"MultiRocket-Hydra Accuracy: {acc:.3f}, Balanced Accuracy: {bacc:.3f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python classify_raw_epochs.py <npz_folder> <labels_csv>")
        sys.exit(1)

    npz_folder = sys.argv[1]
    labels_csv = sys.argv[2]

    X, y, groups = load_raw_eeg_epochs(npz_folder, labels_csv)

    svm_raw_epoch_classify(X, y, groups)
    mrhydra_raw_epoch_classify(X, y, groups)
