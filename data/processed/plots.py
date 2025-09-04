import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

# Load CSV files - update paths if necessary
combo_results = pd.read_csv('classify_combined_results.csv')
per_channel_raw = pd.read_csv('per_channel_raw_eeg_classification.csv')
svm_loo_raw_summary = pd.read_csv('svm_loo_raw_overall_summary.csv')
kfold_summary = pd.read_csv('classification_kfold_summary.csv')
combo_summary = pd.read_csv('classify_combined_summary.csv')
per_channel_raw_summary = pd.read_csv('per_channel_raw_eeg_classification_summary.csv')
svm_all_channels = pd.read_csv('classify_combined_all_channels_SVM.csv')

# Chart 1: Per-channel accuracy comparison (LOO) feature vs raw
plt.figure(figsize=(16, 6))
sns.lineplot(data=combo_results[combo_results['feature']=='pMP'], x='channel', y='accuracy_loo', marker='o', label='PMP Feature')
sns.lineplot(data=combo_results[combo_results['feature']=='HFD'], x='channel', y='accuracy_loo', marker='o', label='HFD Feature')
sns.lineplot(data=combo_results[combo_results['feature']=='pmp+hfd'], x='channel', y='accuracy_loo', marker='o', label='PMP+HFD Feature')
sns.lineplot(data=per_channel_raw, x='channel', y='svm_loo_accuracy', label='Raw EEG SVM LOO')
sns.lineplot(data=per_channel_raw, x='channel', y='multirocket_accuracy', label='Raw EEG MultiROCKET LOO')
sns.lineplot(data=per_channel_raw, x='channel', y='hc2_accuracy', label='Raw EEG HIVE-COTE LOO')
plt.axhline(0.5, color='red', linestyle='--', label='Chance Level')
plt.xlabel('Channel Number')
plt.ylabel('Accuracy (LOO)')
plt.title('Per-Channel Classification Accuracy: Features vs Raw')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('chart1_feature_vs_raw_per_channel_LOO.png')
plt.close()

# Chart 2: Distributions of per-channel accuracies by method
plt.figure(figsize=(12, 6))
data_box = pd.DataFrame({
    'PMP Feature': combo_results[combo_results['feature']=='pMP']['accuracy_loo'],
    'HFD Feature': combo_results[combo_results['feature']=='HFD']['accuracy_loo'],
    'PMP+HFD Feature': combo_results[combo_results['feature']=='pmp+hfd']['accuracy_loo'],
    'Raw EEG SVM LOO': per_channel_raw['svm_loo_accuracy'],
    'Raw EEG MultiROCKET LOO': per_channel_raw['multirocket_accuracy'],
    'Raw EEG HIVE-COTE LOO': per_channel_raw['hc2_accuracy']
})
sns.boxplot(data=data_box, orient='h')
plt.xlabel('Accuracy (LOO)')
plt.title('Distribution of Per-Channel Classification Accuracies by Method')
plt.xlim(0.3, 0.8)
plt.savefig('chart2_accuracy_distribution_boxplot.png')
plt.close()

# Chart 3: Overall accuracy: multichannel TSML vs per-channel feature bests
plt.figure(figsize=(10, 6))
methods = ['Per-channel PMP Best', 'Per-channel HFD Best', 'Per-channel PMP+HFD Best', 'Multichannel SVM (LOO)', 'Multichannel MultiROCKET (Est.)', 'Multichannel HIVE-COTE (Est.)']
accuracy_vals = [
    combo_summary.loc[combo_summary['feature']=='pMP', 'best_accuracy_loo'].values[0],
    combo_summary.loc[combo_summary['feature']=='HFD', 'best_accuracy_loo'].values[0],
    combo_summary.loc[combo_summary['feature']=='pmp+hfd', 'best_accuracy_loo'].values[0],
    svm_loo_raw_summary['accuracy'].values[0],  # SVM LOO overall
    0.735, # Approximate MultiROCKET overall accuracy (example)
    0.7407 # Approximate HIVE-COTE overall accuracy (example)
]
sns.barplot(x=methods, y=accuracy_vals, palette='muted')
plt.ylabel('Accuracy (LOO)')
plt.ylim(0.4, 0.8)
plt.xticks(rotation=45, ha='right')
plt.title('Overall Accuracy Comparison: Multichannel TSML vs Per-channel Features')
plt.tight_layout()
plt.savefig('chart3_overall_accuracy_comparison.png')
plt.close()

# Chart 4: Channel localization: heatmaps of per-channel accuracies PMP, HFD, PMP+HFD
pivot_pmp = combo_results.query("feature=='pMP'")[['channel', 'accuracy_loo']].set_index('channel').T
pivot_hfd = combo_results.query("feature=='HFD'")[['channel', 'accuracy_loo']].set_index('channel').T
pivot_combined = combo_results.query("feature=='pmp+hfd'")[['channel', 'accuracy_loo']].set_index('channel').T
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.heatmap(pivot_pmp, cmap='coolwarm', cbar=False, annot=True, fmt='.2f')
plt.title('PMP per-channel accuracy')
plt.xlabel('Channel')
plt.ylabel('')
plt.subplot(1, 3, 2)
sns.heatmap(pivot_hfd, cmap='coolwarm', cbar=False, annot=True, fmt='.2f')
plt.title('HFD per-channel accuracy')
plt.xlabel('Channel')
plt.ylabel('')
plt.subplot(1, 3, 3)
sns.heatmap(pivot_combined, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('PMP+HFD per-channel accuracy')
plt.xlabel('Channel')
plt.ylabel('')
plt.tight_layout()
plt.savefig('chart4_channel_localization_heatmaps.png')
plt.close()

# Chart 5: Feature importance proxy - best channel accuracies as proxy
plt.figure(figsize=(8, 5))
labels = combo_summary['feature']
accuracies = combo_summary['best_accuracy_loo']
sns.barplot(x=labels, y=accuracies, palette='bright')
plt.title('Best Channel Accuracy as Proxy for Channel Importance by Feature Set')
plt.ylabel('Accuracy (LOO)')
plt.ylim(0.4, 0.7)
plt.tight_layout()
plt.savefig('chart5_feature_importance_proxy.png')
plt.close()

# Chart 6: Accuracy and interpretability trade-offs table
accuracy_data = [
    ['Per-channel Features (Best)', combo_summary['best_accuracy_loo'].mean(), 'High interpretability, good channel localization'],
    ['Per-channel Raw EEG', per_channel_raw_summary['accuracy_mean'].mean(), 'Moderate interpretability, limited localization'],
    ['Multichannel TSML', 0.74, 'Lower interpretability, high accuracy']  # Approximate from MultiROCKET and HIVE-COTE
]
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
col_labels = ['Method', 'Mean Accuracy', 'Interpretability / Localization']
table = ax.table(cellText=accuracy_data, colLabels=col_labels, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)
plt.title('Accuracy and Interpretability Trade-offs')
plt.savefig('chart6_accuracy_interpretability_table.png')
plt.close()

print("All charts created and saved as PNG files.")
