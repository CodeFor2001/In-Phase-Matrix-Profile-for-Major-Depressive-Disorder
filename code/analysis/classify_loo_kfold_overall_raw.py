#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# Configurations - adjust paths if necessary
PROJ_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = PROJ_ROOT / "data" / "raw" / "labels.csv"
NPZ_FOLDER = PROJ_ROOT / "data" / "interim"
OUT_DIR = PROJ_ROOT / "data" / "processed" 
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_LOO_CSV = OUT_DIR / "svm_loo_raw_overall_results.csv"
OUT_LOO_SUMMARY = OUT_DIR / "svm_loo_raw_overall_summary.csv"
OUT_KFOLD_CSV = OUT_DIR / "svm_kfold_raw_overall_results.csv"
OUT_KFOLD_SUMMARY = OUT_DIR / "svm_kfold_raw_overall_summary.csv"

RANDOM_STATE = 42
N_SPLITS = 5  # for K-Fold cv

def load_raw_eeg_and_labels(npz_folder, labels_csv):
    labels_df = pd.read_csv(labels_csv)
    labels_df["subject"] = labels_df["subject"].astype(str).str.strip()

    all_epochs = []
    all_subjects = []

    npz_folder = Path(npz_folder)
    for npz_file in sorted(npz_folder.glob("*.npz")):
        fname = npz_file.name
        if fname.startswith("sub-") and fname.endswith("_epochs.npz"):
            subject = fname[len("sub-"):-len("_epochs.npz")]
        else:
            data = np.load(npz_file)
            subject_field = data.get("subject", None)
            if subject_field is not None:
                if np.ndim(subject_field) == 0:
                    subject = str(subject_field.item())
                else:
                    subject = str(subject_field[0])
            else:
                continue  # Skip if no subject info

        if subject not in labels_df["subject"].values:
            print(f"Subject {subject} not found in labels, skipping.")
            continue

        data = np.load(npz_file)
        epochs = data["epochs"]
        all_epochs.append(epochs)
        all_subjects.extend([subject] * epochs.shape[0])

    if not all_epochs:
        raise RuntimeError("No data loaded - check paths or labels.")

    X = np.concatenate(all_epochs, axis=0)  # (samples, channels, timepoints)
    subjects = np.array(all_subjects)

    # Merge subject to label mapping for samples
    label_map = {row["subject"]: row["group"] for _, row in labels_df.iterrows()}
    y = np.array([label_map[s] for s in subjects])

    # Encode classes to integers:
    classes = sorted(np.unique(y))
    label_encoder = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([label_encoder[v] for v in y])

    return X, y_enc, subjects

def run_loo_svm(X, y_enc):
    loo = LeaveOneOut()
    y_true, y_pred, y_prob = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_tr = X[train_idx]
        y_tr = y_enc[train_idx]
        X_te = X[test_idx]
        y_te = y_enc[test_idx]

        # Flatten spatial and temporal: (samples x features)
        X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
        X_te_flat = X_te.reshape(X_te.shape[0], -1)

        model = make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", probability=True, random_state=42)
        )
        model.fit(X_tr_flat, y_tr)
        y_hat = model.predict(X_te_flat)
        y_p = model.predict_proba(X_te_flat)[:,1]

        y_true.extend(y_te.tolist())
        y_pred.extend(y_hat.tolist())
        y_prob.extend(y_p.tolist())

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))==2 else np.nan

    return y_true, y_pred, y_prob, acc, bal_acc, auc

def run_kfold_svm(X, y_enc, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_true, y_pred, y_prob = [], [], []

    for train_idx, test_idx in skf.split(X, y_enc):
        X_tr = X[train_idx]
        y_tr = y_enc[train_idx]
        X_te = X[test_idx]
        y_te = y_enc[test_idx]

        X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
        X_te_flat = X_te.reshape(X_te.shape[0], -1)

        model = make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", probability=True, random_state=random_state)
        )
        model.fit(X_tr_flat, y_tr)
        y_hat = model.predict(X_te_flat)
        y_p = model.predict_proba(X_te_flat)[:,1]

        y_true.extend(y_te.tolist())
        y_pred.extend(y_hat.tolist())
        y_prob.extend(y_p.tolist())

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))==2 else np.nan

    return y_true, y_pred, y_prob, acc, bal_acc, auc

def save_results_detailed(y_true, y_pred, y_prob, subjects, filename):
    df = pd.DataFrame({
        "subject": subjects,
        "true_label": y_true,
        "predicted_label": y_pred,
        "positive_class_prob": y_prob
    })
    df.to_csv(filename, index=False)

def main():
    print("Loading data...")
    X, y_enc, subjects = load_raw_eeg_and_labels(NPZ_FOLDER, LABELS_CSV)

    print(f"Data shape: {X.shape}, Samples: {len(y_enc)}, Subjects: {len(np.unique(subjects))}")

    print("Running Leave-One-Out CV SVM on raw EEG...")
    y_true_loo, y_pred_loo, y_prob_loo, acc_loo, bal_acc_loo, auc_loo = run_loo_svm(X, y_enc)
    save_results_detailed(y_true_loo, y_pred_loo, y_prob_loo, subjects, OUT_LOO_CSV)

    print("Running K-Fold CV SVM on raw EEG...")
    y_true_kf, y_pred_kf, y_prob_kf, acc_kf, bal_acc_kf, auc_kf = run_kfold_svm(X, y_enc)
    save_results_detailed(y_true_kf, y_pred_kf, y_prob_kf, subjects, OUT_KFOLD_CSV)

    df_summary = pd.DataFrame([
        {"cv_type": "LOO", "accuracy": acc_loo, "balanced_accuracy": bal_acc_loo, "roc_auc": auc_loo},
        {"cv_type": "KFold", "accuracy": acc_kf, "balanced_accuracy": bal_acc_kf, "roc_auc": auc_kf}
    ])

    df_summary.to_csv(OUT_LOO_SUMMARY, index=False)

    print(f"Saved detailed CSVs: \n - {OUT_LOO_CSV} \n - {OUT_KFOLD_CSV}")
    print(f"Saved summary CSV: {OUT_LOO_SUMMARY}")

if __name__ == "__main__":
    main()
