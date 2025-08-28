# #!/usr/bin/env python3

# import numpy as np
# import pandas as pd
# import re
# from pathlib import Path
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# from aeon.classification.convolution_based import MultiRocketClassifier
# from aeon.classification.hybrid import HIVECOTEV2

# PROJ_ROOT = Path(__file__).resolve().parents[2]
# FEATURE_CSV = PROJ_ROOT / "data" / "processed" / "features_pmp_hfd.csv"
# LABEL_CSV = PROJ_ROOT / "data" / "raw" / "labels.csv"
# OUT_DIR = PROJ_ROOT / "data" / "processed"
# OUT_DIR.mkdir(exist_ok=True, parents=True)

# RANDOM_STATE = 42
# N_FOLDS = 5
# HC2_TIME_LIMIT = 10  # Minutes

# FEATURES = ['pmp', 'hfd', 'pmp+hfd']
# CLASSIFIERS = ['svm', 'multirocket', 'hc2']

# def load_data():
#     df_feat = pd.read_csv(FEATURE_CSV)
#     df_lab = pd.read_csv(LABEL_CSV)
#     df_feat["subject"] = df_feat["subject"].astype(str).str.strip()
#     df_lab["subject"] = df_lab["subject"].astype(str).str.strip()
#     df_lab["group"] = df_lab["group"].astype(str).str.strip()

#     common = set(df_feat["subject"]).intersection(set(df_lab["subject"]))
#     if not common:
#         raise ValueError("No overlapping subjects in feature and label files.")

#     df_feat = df_feat[df_feat["subject"].isin(common)].copy()
#     df_lab = df_lab[df_lab["subject"].isin(common)].copy()
#     df = pd.merge(df_feat, df_lab[["subject", "group"]], on="subject", how="inner")
#     df = df.sort_values('subject').reset_index(drop=True)
#     return df

# def discover_channels(df):
#     chans = set()
#     pattern = re.compile(r'^Ch(\d+)_(.*)$')
#     for col in df.columns:
#         m = pattern.match(col)
#         if m:
#             chans.add(int(m.group(1)))
#     return sorted(chans)

# def get_features(df, ch, feat):
#     if feat == 'pmp+hfd':
#         cols = [f'Ch{ch}_pMP', f'Ch{ch}_HFD']
#         cols = [c for c in cols if c in df.columns]
#         if not cols:
#             return None, None
#         X = df[cols].values.astype(np.float64)
#     else:
#         col = f'Ch{ch}_' + feat
#         if col not in df.columns:
#             return None, None
#         X = df[[col]].values.astype(np.float64)
#     y_raw = df['group'].values
#     classes = sorted(np.unique(y_raw))
#     mapping = {c: i for i, c in enumerate(classes)}
#     y = np.array([mapping[v] for v in y_raw])
#     return X, y

# def run_cv(X, y, classifier='svm'):
#     if X is None or len(X) == 0 or len(np.unique(y)) < 2:
#         return [], np.nan, np.nan, np.nan

#     skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

#     y_true_all = []
#     y_pred_all = []
#     y_proba_all = []
#     fold_results = []

#     for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]

#         if classifier == 'svm':
#             model = make_pipeline(
#                 StandardScaler(),
#                 SVC(kernel='linear', probability=True, random_state=RANDOM_STATE)
#             )
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             y_proba = model.predict_proba(X_test)[:, 1]
#         elif classifier == 'multirocket':
#             model = MultiRocketClassifier(n_jobs=-1, random_state=RANDOM_STATE)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             y_proba = None
#         elif classifier == 'hc2':
#             model = HIVECOTEV2(n_jobs=-1, random_state=RANDOM_STATE, time_limit_in_minutes=HC2_TIME_LIMIT)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             y_proba = None
#         else:
#             raise ValueError(f"Unknown classifier: {classifier}")

#         acc = accuracy_score(y_test, y_pred)
#         bal_acc = balanced_accuracy_score(y_test, y_pred)
#         auc = np.nan
#         if y_proba is not None and len(np.unique(y_test)) == 2:
#             try:
#                 auc = roc_auc_score(y_test, y_proba)
#             except Exception:
#                 pass

#         fold_results.append({
#             'fold': fold_idx, 'accuracy': acc, 'balanced_accuracy': bal_acc, 'auc': auc
#         })

#         y_true_all.extend(y_test)
#         y_pred_all.extend(y_pred)
#         if y_proba is not None:
#             y_proba_all.extend(y_proba)

#     overall_acc = accuracy_score(y_true_all, y_pred_all)
#     overall_bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
#     overall_auc = np.nan
#     if y_proba_all and len(np.unique(y_true_all)) == 2:
#         try:
#             overall_auc = roc_auc_score(y_true_all, y_proba_all)
#         except Exception:
#             pass

#     return fold_results, overall_acc, overall_bal_acc, overall_auc

# def main():
#     df = load_data()
#     chans = discover_channels(df)

#     # Dict to keep results per classifier
#     detailed_results = {clf: [] for clf in CLASSIFIERS}
#     summary_results = {clf: [] for clf in CLASSIFIERS}

#     for feat in FEATURES:
#         for ch in chans:
#             X, y = get_features(df, ch, feat)
#             if X is None:
#                 continue

#             valid_mask = np.isfinite(X).all(axis=1)
#             X_v, y_v = X[valid_mask], y[valid_mask]

#             if len(y_v) < 5 or len(np.unique(y_v)) < 2:
#                 continue

#             # Reshape for MultiRocket and HC2 classifiers expected input: 3D (samples, channels, timepoints)
#             if isinstance(X_v, np.ndarray) and (X_v.ndim == 2):
#                 X_v_expanded = X_v.reshape((X_v.shape[0], 1, X_v.shape[1]))
#             else:
#                 X_v_expanded = X_v

#             for clf in CLASSIFIERS:
#                 print(f"Processing: Feature={feat}, Channel={ch}, Classifier={clf}")
#                 if clf == 'svm':
#                     fold_res, acc, bal_acc, auc = run_cv(X_v, y_v, 'svm')
#                 else:
#                     fold_res, acc, bal_acc, auc = run_cv(X_v_expanded, y_v, clf)

#                 # Append per fold detailed results
#                 for fr in fold_res:
#                     detailed_results[clf].append({
#                         'feature': feat,
#                         'channel': ch,
#                         'classifier': clf,
#                         'fold': fr['fold'],
#                         'accuracy': fr['accuracy'],
#                         'balanced_accuracy': fr['balanced_accuracy'],
#                         'auc': fr['auc'],
#                         'samples': len(y_v)
#                     })

#                 # Append per channel summary for this classifier and feature
#                 summary_results[clf].append({
#                     'feature': feat,
#                     'channel': ch,
#                     'accuracy': acc,
#                     'balanced_accuracy': bal_acc,
#                     'auc': auc,
#                     'samples': len(y_v)
#                 })

#     # Save separate detailed result CSVs per classifier
#     for clf in CLASSIFIERS:
#         df_det = pd.DataFrame(detailed_results[clf])
#         df_det.to_csv(OUT_DIR / f'{clf}_per_channel_folds.csv', index=False)
#         print(f'Saved detailed per-fold results for {clf} at {OUT_DIR / f"{clf}_per_channel_folds.csv"}')

#     # Save one combined summary with classifier name added
#     combined_summary = []
#     for clf in CLASSIFIERS:
#         for row in summary_results[clf]:
#             row_copy = row.copy()
#             row_copy['classifier'] = clf
#             combined_summary.append(row_copy)
#     df_summary = pd.DataFrame(combined_summary)
#     df_summary.to_csv(OUT_DIR / 'per_channel_classification_summary.csv', index=False)
#     print(f'Saved combined per-channel summary at {OUT_DIR / "per_channel_classification_summary.csv"}')

# if __name__ == '__main__':
#     CLASSIFIERS = ['svm', 'multirocket', 'hc2']
#     main()
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from aeon.classification.convolution_based import MultiRocketHydraClassifier

def sliding_window(data, window_size, step):
    # data: (timesteps, features)
    segments = []
    for start in range(0, len(data) - window_size + 1, step):
        seg = data[start:start+window_size]
        segments.append(seg)
    return np.array(segments)

def prepare_sliding_windows(feature_df, window_size=5, step=2):
    # feature_df assumed to have columns: ['subject', 'channel', 'time_idx', 'pMP', 'HFD', ...]
    
    X_list = []
    y_list = []
    groups = []
    
    subjects = feature_df['subject'].unique()
    
    for subj in subjects:
        subj_df = feature_df[feature_df['subject'] == subj]
        chans = subj_df['channel'].unique()
        
        for chan in chans:
            chan_df = subj_df[subj_df['channel'] == chan].sort_values('time_idx')
            
            # Extract features as array: (timepoints, features)
            feat_cols = [col for col in ['pMP', 'HFD'] if col in feature_df.columns]
            feat_mat = chan_df[feat_cols].values  # e.g. (T, 2)
            
            # Apply sliding windows
            windows = sliding_window(feat_mat, window_size, step)  # (N_windows, window_size, features)
            
            # Prepare MultiRocketHydra input: (samples, channels, timepoints)
            # Here channels = features, timepoints = window_size
            windows = np.transpose(windows, (0, 2, 1))  # reshape to (N, features=channels, timepoints)
            
            X_list.append(windows)
            y_list.extend([subj_df['group'].iloc[0]] * windows.shape[0])
            groups.extend([subj] * windows.shape[0])
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    groups = np.array(groups)
    
    # Map labels to 0/1
    labels = pd.Series(y)
    label_map = {label: idx for idx, label in enumerate(np.unique(y))}
    y_encoded = labels.map(label_map).values
    
    return X, y_encoded, groups

# Example usage in a model training context:

feature_df = pd.read_csv('data/processed/features_pmp_hfd.csv')  # subject,channel,time_idx,pMP,HFD

window_size = 5
step = 2

X, y, groups = prepare_sliding_windows(feature_df, window_size, step)

cv = GroupKFold(n_splits=5)
clf = MultiRocketHydraClassifier(random_state=42, n_jobs=-1)

acc_list = []
for train_idx, test_idx in cv.split(X, y, groups):
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])
    acc = np.mean(y_pred == y[test_idx])
    acc_list.append(acc)
    print(f'Fold accuracy: {acc:.3f}')

print(f'Average cross-validation accuracy: {np.mean(acc_list):.3f}')
