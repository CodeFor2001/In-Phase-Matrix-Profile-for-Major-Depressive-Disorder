#!/usr/bin/env python3
"""
Experiment Charts
- Lineplot: Accuracy across folds per channel (classification_kfold_folds_channelwise.csv)
- Barplot: Mean accuracy per feature type (classify_combined_results.csv)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid", font_scale=1.2)

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data" / "processed"
OUT_DIR = PROJ_ROOT / "reports" / "figures_experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1. Lineplot – Accuracy across folds per channel
# -------------------------
df_kfold = pd.read_csv(DATA_DIR / "classification_kfold_folds_channelwise.csv")

plt.figure(figsize=(12,6))
sns.lineplot(x="fold", y="accuracy", hue="channel", data=df_kfold, legend=False, alpha=0.6)
plt.title("Accuracy Across Folds per Channel (SVM KFold)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.tight_layout()

# -------------------------
# 2. Barplot – Mean accuracy per feature type
# -------------------------
df_combined = pd.read_csv(DATA_DIR / "classify_combined_results.csv")
mean_acc = df_combined.groupby("feature")["accuracy_loo"].mean().reset_index()

plt.figure(figsize=(8,6))
sns.barplot(x="feature", y="accuracy_loo", data=mean_acc, palette="Set2")
plt.title("Mean Accuracy per Feature Type (Channelwise-SVM LOO)")
plt.xlabel("Feature")
plt.ylabel("Mean Accuracy")
plt.tight_layout()

# -------------------------
# 3. Combined Figure (side by side)
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(18,6))

# Lineplot
sns.lineplot(x="fold", y="accuracy", hue="channel", data=df_kfold, legend=False, alpha=0.6, ax=axes[0])
axes[0].set_title("Accuracy Across Folds per Channel (SVM KFold)")
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("Accuracy")

# Barplot
sns.barplot(x="feature", y="accuracy_loo", data=mean_acc, palette="Set2", ax=axes[1])
axes[1].set_title("Mean Accuracy per Feature Type (Channelwise - SVM LOO)")
axes[1].set_xlabel("Feature")
axes[1].set_ylabel("Mean Accuracy")

plt.tight_layout()
plt.savefig(OUT_DIR/"ChannelwiseFeatureCharts.png", dpi=300)

print(f"✅ Charts saved in {OUT_DIR}")
