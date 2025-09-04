#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from aeon.classification.hybrid import HIVECOTEV2

# Config
PROJ_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = PROJ_ROOT / "data" / "raw" / "labels.csv"
NPZ_FOLDER = PROJ_ROOT / "data" / "interim"
OUT_DIR = PROJ_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DETAILED = OUT_DIR / "hc2_raw_detailed.csv"
OUT_SUMMARY = OUT_DIR / "hc2_raw_summary.csv"

def load_raw_eeg_and_labels(npz_folder, labels_csv):
    labels_df = pd.read_csv(labels_csv)
    labels_df["subject"] = labels_df["subject"].astype(str).str.strip()

    all_epochs, all_subjects = [], []
    npz_folder = Path(npz_folder)

    for npz_file in sorted(npz_folder.glob("*.npz")):
        fname = npz_file.name
        if fname.startswith("sub-") and fname.endswith("_epochs.npz"):
            subject = fname[len("sub-"):-len("_epochs.npz")]
        else:
            data = np.load(npz_file)
            subject_field = data.get("subject", None)
            if subject_field is not None:
                subject = str(subject_field.item() if np.ndim(subject_field) == 0 else subject_field[0])
            else:
                continue

        if subject not in labels_df["subject"].values:
            print(f"Skipping {subject} (no label).")
            continue

        data = np.load(npz_file)
        epochs = data["epochs"]
        all_epochs.append(epochs)
        all_subjects.extend([subject] * epochs.shape[0])

    if not all_epochs:
        raise RuntimeError("No data loaded.")

    X = np.concatenate(all_epochs, axis=0)
    subjects = np.array(all_subjects)
    label_map = {row["subject"]: row["group"] for _, row in labels_df.iterrows()}
    y_labels = np.array([label_map[s] for s in subjects])

    classes = sorted(np.unique(y_labels))
    encoder = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([encoder[v] for v in y_labels])

    return X, y_enc, subjects

def run_hc2(X, y_enc, subjects, n_splits=5, time_limit=10):
    gkf = GroupKFold(n_splits=n_splits)
    y_true, y_pred, subj_all = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_enc, subjects), 1):
        print(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]
        subj_te = subjects[test_idx]

        clf = HIVECOTEV2(n_jobs=-1, random_state=42, time_limit_in_minutes=time_limit)
        clf.fit(X_tr, y_tr)
        y_hat = clf.predict(X_te)

        y_true.extend(y_te)
        y_pred.extend(y_hat)
        subj_all.extend(subj_te)

    # Save detailed
    pd.DataFrame({
        "subject": subj_all,
        "true_label": y_true,
        "predicted_label": y_pred
    }).to_csv(OUT_DETAILED, index=False)

    # Save summary
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    pd.DataFrame([{
        "classifier": "HIVECOTEV2",
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "n_samples": len(y_true)
    }]).to_csv(OUT_SUMMARY, index=False)

    print(f"Saved: {OUT_DETAILED}, {OUT_SUMMARY}")

def main():
    print("Loading data...")
    X, y_enc, subjects = load_raw_eeg_and_labels(NPZ_FOLDER, LABELS_CSV)
    run_hc2(X, y_enc, subjects)

if __name__ == "__main__":
    main()
