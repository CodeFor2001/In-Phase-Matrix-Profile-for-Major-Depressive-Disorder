#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# Paths (adjust if needed)
PROJ_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJ_ROOT / "data" / "processed" / "per_channel_raw_eeg_classification.csv"
SUMMARY_CSV = PROJ_ROOT / "data" / "processed" / "per_channel_classification_summary.csv"

# Mapping of classifier name prefixes to meaningful summary labels
CLASSIFIERS = {
    "svm_loo": "SVM Leave-One-Out",
    "svm_kfold": "SVM 5-Fold CV",
    "multirocket": "MultiRocket",
    "hc2": "HIVECOTEV2"
}

def summarize_metrics(df):
    summary_rows = []
    for prefix, display_name in CLASSIFIERS.items():
        acc_col = f"{prefix}_accuracy"
        bal_col = f"{prefix}_balanced_accuracy"
        auc_col = f"{prefix}_auc"

        # Mean and std
        acc_mean = df[acc_col].mean() if acc_col in df.columns else None
        acc_std = df[acc_col].std() if acc_col in df.columns else None

        bal_mean = df[bal_col].mean() if bal_col in df.columns else None
        bal_std = df[bal_col].std() if bal_col in df.columns else None

        auc_mean = df[auc_col].mean() if auc_col in df.columns else None
        auc_std = df[auc_col].std() if auc_col in df.columns else None

        # Best channel for accuracy
        if acc_col in df.columns:
            best_idx = df[acc_col].idxmax()
            best_channel = df.loc[best_idx, "channel"]
            best_acc = df.loc[best_idx, acc_col]
        else:
            best_channel = None
            best_acc = None

        summary_rows.append({
            "classifier": display_name,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "balanced_accuracy_mean": bal_mean,
            "balanced_accuracy_std": bal_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "best_channel": best_channel,
            "best_channel_accuracy": best_acc
        })

    return pd.DataFrame(summary_rows)

def main():
    df = pd.read_csv(INPUT_CSV)
    summary_df = summarize_metrics(df)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary saved to {SUMMARY_CSV}")

if __name__ == "__main__":
    main()
