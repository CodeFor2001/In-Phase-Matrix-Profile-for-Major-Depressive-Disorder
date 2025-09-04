#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid", font_scale=1.2)

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data" / "processed"
OUT_DIR = PROJ_ROOT / "reports" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# RQ1: Feature vs Raw (Per-channel)
# -------------------------------------------------
# -------------------------------------------------
# RQ1: Feature vs Raw (Per-channel)
# -------------------------------------------------
def plot_rq1():
    df_feat_loo = pd.read_csv(DATA_DIR / "classify_combined_summary.csv")        # SVM LOO
    df_feat_kf = pd.read_csv(DATA_DIR / "classification_kfold_summary.csv")      # SVM KFold
    df_raw = pd.read_csv(DATA_DIR / "per_channel_raw_eeg_classification.csv")

    # --- Figure 5.2: Compare LOO vs KFold features
    df_feat_loo["cv_type"] = "LOO"
    df_feat_kf["cv_type"] = "KFold"
    df_feat = pd.concat([df_feat_loo, df_feat_kf])

    plt.figure(figsize=(10,6))
    sns.barplot(x="feature", y="best_accuracy_loo", hue="cv_type", data=df_feat, palette="Set2")
    plt.title("RQ1 (H1): Best Feature Accuracies (LOO vs KFold)")
    plt.ylabel("Best Channel Accuracy")
    plt.xlabel("Feature")
    plt.xticks(rotation=0)
    plt.savefig(OUT_DIR / "fig_RQ1_features_best.png", dpi=300)
    plt.close()

    # --- Figure 5.3: Raw classifiers best channel
    metrics = ["svm_loo_accuracy", "svm_kfold_accuracy", "multirocket_accuracy", "hc2_accuracy"]
    best_raw = {m: df_raw[m].max() for m in metrics}
    df_best_raw = pd.DataFrame(list(best_raw.items()), columns=["method", "best_acc"])

    plt.figure(figsize=(10,6))
    sns.barplot(x="method", y="best_acc", data=df_best_raw, palette="muted")
    plt.title("RQ1 (H1): Best Raw Classifier Accuracies")
    plt.ylabel("Best Channel Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_RQ1_raw_best.png", dpi=300)
    plt.close()

    # --- Figure 5.4: Distribution of per-channel raw vs features
    melted = df_raw.melt(id_vars="channel",
                         value_vars=["svm_loo_accuracy","svm_kfold_accuracy","multirocket_accuracy","hc2_accuracy"],
                         var_name="classifier", value_name="accuracy")

    plt.figure(figsize=(12,6))
    sns.boxplot(x="classifier", y="accuracy", data=melted, palette="Set3")
    plt.title("RQ1 (H1): Distribution of Per-channel Raw Classifier Accuracies")
    plt.ylabel("Accuracy")
    plt.xlabel("Classifier")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_RQ1_raw_distributions.png", dpi=300)
    plt.close()


# -------------------------------------------------
# RQ2: Per-channel vs Multichannel
# -------------------------------------------------
def plot_rq2():
    df_raw = pd.read_csv(DATA_DIR / "per_channel_raw_eeg_classification.csv")
    df = pd.read_csv(DATA_DIR / "svm_loo_raw_overall_summary.csv")  # contains both LOO & KFold

    # --- Figure 5.5: Mean accuracy comparison (per-channel avg vs multichannel)
    mean_per_channel = df_raw[["svm_loo_accuracy",
                               "multirocket_accuracy",
                               "hc2_accuracy"]].mean().mean()

    df_perf = pd.DataFrame({
        "method": ["Per-channel (avg)",
                   "SVM Multichannel (LOO)",
                   "SVM Multichannel (KFold)"],
        "accuracy": [mean_per_channel,
                     df.loc[df["cv_type"]=="LOO","accuracy"].values[0],
                     df.loc[df["cv_type"]=="KFold","accuracy"].values[0]]
    })

    plt.figure(figsize=(8,6))
    sns.barplot(x="method", y="accuracy", data=df_perf, palette="Set2")
    plt.title("RQ2 (H2): Per-channel vs Multichannel Accuracy")
    plt.ylabel("Accuracy")
    plt.savefig(OUT_DIR / "fig_RQ2_mean_comparison.png", dpi=300)
    plt.close()

    # --- Figure 5.6: Accuracy, Balanced Accuracy, AUC comparison
    melted = df.melt(id_vars="cv_type", var_name="metric", value_name="score")
    plt.figure(figsize=(10,6))
    sns.barplot(x="metric", y="score", hue="cv_type", data=melted, palette="muted")
    plt.title("RQ2 (H2): Multichannel Performance Metrics (SVM)")
    plt.ylabel("Score")
    plt.savefig(OUT_DIR / "fig_RQ2_metrics.png", dpi=300)
    plt.close()

# -------------------------------------------------
# RQ3: Interpretability & Localisation
# -------------------------------------------------
def plot_rq3():
    df_feat = pd.read_csv(DATA_DIR / "classify_combined_results.csv")
    df_raw = pd.read_csv(DATA_DIR / "per_channel_raw_eeg_classification.csv")

    # --- Figure 5.7: Heatmap feature accuracies
    feat_map = df_feat.pivot(index="feature", columns="channel", values="accuracy_loo")
    plt.figure(figsize=(14,4))
    sns.heatmap(feat_map, cmap="viridis", cbar_kws={"label":"Accuracy"})
    plt.title("RQ3 (H3): Channel-wise Accuracy – Features (PMP/HFD)")
    plt.savefig(OUT_DIR / "fig_RQ3_feature_heatmap.png", dpi=300)
    plt.close()

    # --- Figure 5.8: Heatmap raw accuracies
    raw_map = df_raw.set_index("channel")[["svm_loo_accuracy","multirocket_accuracy","hc2_accuracy"]].T
    plt.figure(figsize=(14,4))
    sns.heatmap(raw_map, cmap="magma", cbar_kws={"label":"Accuracy"})
    plt.title("RQ3 (H3): Channel-wise Accuracy – Raw Classifiers")
    plt.savefig(OUT_DIR / "fig_RQ3_raw_heatmap.png", dpi=300)
    plt.close()

    # --- Figure 5.9: Side-by-side comparison (features vs raw SVM)
    merged = pd.merge(
        df_feat[df_feat["feature"]=="pMP"][["channel","accuracy_loo"]],
        df_raw[["channel","svm_loo_accuracy"]],
        on="channel"
    ).rename(columns={"accuracy_loo":"Feature_PMP","svm_loo_accuracy":"Raw_SVM"})
    melted = merged.melt(id_vars="channel", var_name="method", value_name="accuracy")
    plt.figure(figsize=(14,6))
    sns.lineplot(x="channel", y="accuracy", hue="method", data=melted, marker="o")
    plt.title("RQ3 (H3): Channel-localisation – Feature vs Raw")
    plt.ylabel("Accuracy")
    plt.xlabel("Channel")
    plt.savefig(OUT_DIR / "fig_RQ3_feature_vs_raw_channels.png", dpi=300)
    plt.close()

    # -------------------------------------------------
# Extra RQ1: Channelwise Features across classifiers
# -------------------------------------------------
def plot_features_classifiers_comparison():
    df_loo = pd.read_csv(DATA_DIR / "classify_combined_summary.csv")
    df_kf = pd.read_csv(DATA_DIR / "classification_kfold_summary.csv")

    df_loo["classifier"] = "SVM LOO"
    df_kf["classifier"] = "SVM KFold"
    df_feat = pd.concat([df_loo, df_kf])

    plt.figure(figsize=(10,6))
    sns.barplot(x="feature", y="best_accuracy_loo", hue="classifier", data=df_feat, palette="Set2")
    plt.title("RQ1: Channelwise Feature Performance across Classifiers")
    plt.ylabel("Best Channel Accuracy")
    plt.savefig(OUT_DIR / "fig_RQ1_features_classifiers.png", dpi=300)
    plt.close()


# -------------------------------------------------
# Extra RQ2: Raw channelwise vs Raw multichannel
# -------------------------------------------------
# -------------------------------------------------
# RQ2: Raw channelwise vs Raw multichannel (all classifiers)
# -------------------------------------------------
def plot_raw_channel_vs_overall_full():
    df_raw = pd.read_csv(DATA_DIR / "per_channel_raw_eeg_classification.csv")
    df_overall = pd.read_csv(DATA_DIR / "per_channel_raw_eeg_classification_summary.csv")

    # Mean channelwise accuracy (per classifier)
    mean_raw = {
        "SVM (per-channel LOO)": df_raw["svm_loo_accuracy"].mean(),
        "SVM (per-channel KFold)": df_raw["svm_kfold_accuracy"].mean(),
        "MultiRocket (per-channel)": df_raw["multirocket_accuracy"].mean(),
        "HC2 (per-channel)": df_raw["hc2_accuracy"].mean(),
    }

    # Overall results
    overall = {
        row["classifier"]: row["accuracy_mean"]
        for _, row in df_overall.iterrows()
    }

    df_compare = pd.DataFrame(list(mean_raw.items()) + list(overall.items()),
                              columns=["method","accuracy"])

    plt.figure(figsize=(12,6))
    sns.barplot(x="method", y="accuracy", data=df_compare, palette="muted")
    plt.title("RQ2: Raw Channelwise vs Multichannel Accuracy (All Classifiers)")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_RQ2_raw_channel_vs_overall_full.png", dpi=300)
    plt.close()

    # Balanced Accuracy comparison
    df_bal = pd.DataFrame({
        "method": list(overall.keys()),
        "balanced_accuracy": df_overall["balanced_accuracy_mean"].values
    })
    plt.figure(figsize=(10,6))
    sns.barplot(x="method", y="balanced_accuracy", data=df_bal, palette="Set2")
    plt.title("RQ2: Multichannel Classifier Balanced Accuracy")
    plt.ylabel("Balanced Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_RQ2_raw_overall_balacc.png", dpi=300)
    plt.close()

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    plot_rq1()
    plot_rq2()
    plot_rq3()
    plot_features_classifiers_comparison()
    plot_raw_channel_vs_overall_full()
    print(f"Figures saved in {OUT_DIR}")
