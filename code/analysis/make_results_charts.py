#!/usr/bin/env python3
"""
Dissertation Results Charts (Cleaned Version)
- RQ1: Channelwise pMP vs Raw
- RQ2: Multichannel Features vs Raw
- RQ3: Interpretability (heatmaps & scatter)
- Final Unified Comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import wilcoxon

sns.set(style="whitegrid", font_scale=1.2)

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data" / "processed"
OUT_DIR = PROJ_ROOT / "reports" / "figures_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Helper for p-value annotation
# -------------------------
def add_pval_annotation(ax, x1, x2, y, pval, text_offset=0.01):
    """Draw significance stars between two bars"""
    line_y = y + text_offset
    ax.plot([x1, x1, x2, x2], [y, line_y, line_y, y], lw=1.5, c="k")
    if pval < 0.001:
        text = "***"
    elif pval < 0.01:
        text = "**"
    elif pval < 0.05:
        text = "*"
    else:
        text = "n.s."
    ax.text((x1+x2)/2, line_y+text_offset, text, ha="center", va="bottom")

# -------------------------
# Load Data
# -------------------------
df_feat_loo = pd.read_csv(DATA_DIR / "classify_combined_summary.csv")   # Channelwise Features (LOO)
df_feat_kf = pd.read_csv(DATA_DIR / "classification_kfold_summary.csv") # Channelwise Features (KFold)
df_feat_multi = pd.read_csv(DATA_DIR / "classify_combined_SVM_metrics.csv") # Multichannel Features

df_raw_ch = pd.read_csv(DATA_DIR / "per_channel_raw_eeg_classification_summary.csv") # Channelwise Raw
df_raw_loo = pd.read_csv(DATA_DIR / "svm_loo_raw_overall_summary.csv")   # Multichannel Raw SVM LOO/KFold
df_raw_svm = pd.read_csv(DATA_DIR / "svm_raw_summary.csv")               # Multichannel Raw SVM KFold
df_raw_mr = pd.read_csv(DATA_DIR / "multirocket_raw_summary.csv")        # Multichannel Raw MultiRocket
df_raw_hc2 = pd.read_csv(DATA_DIR / "hc2_raw_summary.csv")               # Multichannel Raw HC2

# =====================================================
# RQ1: Channelwise Features vs Raw
# =====================================================

# Channelwise Features: pMP only
cw_features = pd.DataFrame({
    "method": ["pMP (LOO)", "pMP (KFold)"],
    "accuracy": [
        df_feat_loo.loc[df_feat_loo["feature"]=="pMP","best_accuracy_loo"].values[0],
        df_feat_kf.loc[df_feat_kf["feature"]=="pMP","best_accuracy"].values[0]
    ]
})

df_feat_loo["cv_type"] = "LOO"
df_feat_kf["cv_type"] = "KFold"
df_feat = pd.concat([
    df_feat_loo.rename(columns={"best_accuracy_loo":"best_accuracy"}),
    df_feat_kf.rename(columns={"best_accuracy":"best_accuracy"})
])

plt.figure(figsize=(8,6))
sns.barplot(x="feature", y="best_accuracy", hue="cv_type", data=df_feat, palette="Set2")
plt.title("RQ1: Channelwise Features – Best Channel Accuracies")
plt.ylabel("Accuracy")
plt.savefig(OUT_DIR/"RQ1_channelwise_features.png", dpi=300)

# Channelwise Raw (mean accuracy per classifier)
cw_raw = df_raw_ch[["classifier","accuracy_mean"]].rename(
    columns={"classifier":"method","accuracy_mean":"accuracy"}
)

plt.figure(figsize=(8,6))
sns.barplot(x="method", y="accuracy", data=cw_raw, palette="muted")
plt.title("RQ1: Channelwise Raw – Mean Accuracies")
plt.ylabel("Accuracy")
plt.xticks(rotation=20, ha="right")
plt.savefig(OUT_DIR/"RQ1_channelwise_raw.png", dpi=300)

# Features vs Raw (SVM LOO) – statistical comparison
feat_acc = df_feat_loo.loc[df_feat_loo["feature"]=="pMP","best_accuracy_loo"].values
raw_acc = df_raw_ch.loc[df_raw_ch["classifier"]=="SVM LOO","accuracy_mean"].values
if len(feat_acc) and len(raw_acc):
    stat, p = wilcoxon(feat_acc, raw_acc)
    plt.figure(figsize=(6,6))
    sns.barplot(x=["pMP (LOO)","SVM LOO"], y=[feat_acc.mean(), raw_acc.mean()], palette="Set2")
    ax = plt.gca()
    ymax = max(feat_acc.max(), raw_acc.max())
    add_pval_annotation(ax, 0, 1, ymax, p)
    plt.title("RQ1: Channelwise Features vs Raw (SVM LOO)")
    plt.ylabel("Accuracy")
    plt.savefig(OUT_DIR/"RQ1_features_vs_raw_pval.png", dpi=300)

# =====================================================
# RQ2: Multichannel Features vs Raw
# =====================================================

# Multichannel Features (SVM LOO & KFold)
mc_features = pd.DataFrame({
    "method": ["SVM (Features LOO)", "SVM (Features KFold)"],
    "accuracy": [
        df_feat_multi.loc[df_feat_multi["cv_type"]=="LOO","accuracy"].values[0],
        df_feat_multi.loc[df_feat_multi["cv_type"]=="KFold","accuracy"].values[0]
    ]
})

# Multichannel Raw
mc_raw = pd.DataFrame({
    "method": ["SVM (LOO)", "SVM (KFold)", "MultiRocketHydra", "HC2"],
    "accuracy": [
        df_raw_loo.loc[df_raw_loo["cv_type"]=="LOO","accuracy"].values[0],
        df_raw_loo.loc[df_raw_loo["cv_type"]=="KFold","accuracy"].values[0],
        df_raw_mr["accuracy"].values[0],
        df_raw_hc2["accuracy"].values[0]
    ]
})

plt.figure(figsize=(10,6))
sns.barplot(x="method", y="accuracy", data=pd.concat([mc_features, mc_raw]), palette="Set1")
plt.title("RQ2: Multichannel Features vs Raw")
plt.ylabel("Accuracy")
plt.xticks(rotation=25, ha="right")
plt.savefig(OUT_DIR/"RQ2_multichannel_features_vs_raw.png", dpi=300)

# Multichannel Raw – all metrics
df_multi_all = pd.concat([
    df_raw_loo.rename(columns={"cv_type":"method"}).assign(classifier="SVM"),
    df_raw_svm.assign(method="SVM (KFold)", classifier="SVM"),
    df_raw_mr.assign(method="MultiRocketHydra", classifier="MultiRocket"),
    df_raw_hc2.assign(method="HC2", classifier="HC2"),
])
melted = df_multi_all.melt(id_vars=["method"], 
                           value_vars=["accuracy","balanced_accuracy","roc_auc"],
                           var_name="metric", value_name="score")
plt.figure(figsize=(12,6))
sns.barplot(x="method", y="score", hue="metric", data=melted, palette="muted")
plt.title("RQ2: Multichannel Raw – Accuracy, Balanced Accuracy, AUC")
plt.ylabel("Score")
plt.xticks(rotation=25, ha="right")
plt.savefig(OUT_DIR/"RQ2_multichannel_raw_metrics.png", dpi=300)

# =====================================================
# RQ3: Interpretability
# =====================================================

# Heatmap – Features (pMP LOO only)
pivot_feat = df_feat_loo.pivot(index="feature", columns="best_channel", values="best_accuracy_loo")
plt.figure(figsize=(12,6))
sns.heatmap(pivot_feat, cmap="coolwarm", center=0.5, cbar_kws={"label":"Accuracy"})
plt.title("RQ3: Heatmap – Channelwise Feature Accuracies (SVM LOO)")
plt.savefig(OUT_DIR/"RQ3_features_heatmap.png", dpi=300)

# Heatmap – Raw
pivot_raw = df_raw_ch.pivot(index="classifier", columns="best_channel", values="accuracy_mean")
plt.figure(figsize=(12,6))
sns.heatmap(pivot_raw, cmap="viridis", cbar_kws={"label":"Accuracy"})
plt.title("RQ3: Channelwise Raw Accuracies")
plt.savefig(OUT_DIR/"RQ3_raw_heatmap.png", dpi=300)

# Scatter – pMP vs SVM LOO
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df_feat_loo.loc[df_feat_loo["feature"]=="pMP","best_accuracy_loo"],
    y=df_raw_ch.loc[df_raw_ch["classifier"]=="SVM LOO","accuracy_mean"]
)
plt.plot([0,1],[0,1],'r--')
plt.xlabel("pMP (LOO) Accuracy")
plt.ylabel("SVM LOO Raw Accuracy")
plt.title("RQ3: Channelwise pMP vs Raw Accuracy")
plt.savefig(OUT_DIR/"RQ3_scatter_pmp_vs_raw.png", dpi=300)

# =====================================================
# Final Unified Comparison
# =====================================================

df_all = pd.concat([
    cw_features.assign(category="Channelwise Feature"),
    cw_raw.assign(category="Channelwise Raw"),
    mc_features.assign(category="Multichannel Feature"),
    mc_raw.assign(category="Multichannel Raw"),
], ignore_index=True)
# Standardize method labels
label_map = {
    "pMP (LOO)": "SVM LOO",
    "pMP (KFold)": "SVM KFold",
    "SVM (LOO)": "SVM LOO",
    "SVM (KFold)": "SVM KFold",
    "SVM (Features LOO)": "SVM LOO",
    "SVM (Features KFold)": "SVM KFold",
    "SVM Leave-One-Out": "SVM LOO",
    "SVM 5-Fold CV": "SVM KFold",
    "MultiRocket": "MultiRocketHydra",
    "MultiRocketHydra": "MultiRocketHydra",
    "HIVECOTEV2": "HC2",
    "HC2": "HC2"
}

df_all["method"] = df_all["method"].map(lambda x: label_map.get(x, x))

plt.figure(figsize=(12,7))
sns.barplot(x="category", y="accuracy", hue="method", data=df_all, palette="Set2")
plt.title("Final Comparison: Channelwise vs Multichannel, Features vs Raw")
plt.ylabel("Accuracy")
plt.xlabel("")
plt.xticks(rotation=15)
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", title="Method")
plt.tight_layout()
plt.savefig(OUT_DIR/"Final_comparison_clean.png", dpi=300)

print(f"✅ Figures saved to {OUT_DIR}")
