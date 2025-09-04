#!/usr/bin/env python3
"""
Worked Example: In-Phase Matrix Profile (PMP) pipeline on one MODMA subject.

Generates Figures A–F for workflow explanation:
 A – Raw vs Preprocessed EEG
 B – PSD context (Welch)
 C – MP vs PMP
 D – Pairwise distance map (toy)
 E – Per-channel PMP heatmap (this subject)
 F – Feature attribution (toy SVM with permutation importance)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.signal import welch
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

# ----------------- Project paths -----------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
PREPROC_DIR = PROJ_ROOT / "code" / "preprocess"
sys.path.append(str(PREPROC_DIR))

from filtering import preprocess_filter_resample
from rest_ref import apply_reference
from quality import is_bad_epoch
from epoching import slice_epochs
from pmp import pmp_epoch, mass

RAW_FILE = PROJ_ROOT / "data" / "raw" / "MODMA_EEG" / "02010002rest 20150416 1017..mat"
OUT_DIR = PROJ_ROOT / "reports" / "workflow"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Parameters -----------------
target_fs = 128
hp, lp = 2, 47
epoch_len_s = 20.48
m = 256
local_min_dist = 3
qc_params = dict(amp_thresh=100.0, var_thresh=1e-6, hf_ratio=0.5)
channels_to_use = 10  # limit channels for speed

# ----------------- Load raw MODMA .mat -----------------
print("Loading raw .mat file...")
mat_data = loadmat(RAW_FILE)
print("Available keys:", list(mat_data.keys()))

# detect EEG array key automatically
data_keys = [k for k in mat_data.keys() if not k.startswith("__") and k not in ["samplingRate","Impedances_0"]]
if not data_keys:
    raise KeyError(f"No EEG data found in {RAW_FILE}. Keys: {list(mat_data.keys())}")

raw_key = data_keys[0]
raw_eeg = mat_data[raw_key]

# Ensure (n_channels, n_samples)
if raw_eeg.shape[0] > raw_eeg.shape[1]:
    raw_eeg = raw_eeg.T
print("Final raw EEG shape:", raw_eeg.shape)

# sampling rate
orig_fs = int(mat_data.get("samplingRate", [[1000]])[0][0])
print("Original fs from .mat:", orig_fs)

# ----------------- Preprocess raw -----------------
print("Filtering + resampling...")
eeg_filt = preprocess_filter_resample(raw_eeg, orig_fs, hp, lp, target_fs)

print("Re-referencing...")
eeg_ref = apply_reference(eeg_filt, method="average")

print("Epoching...")
epochs = slice_epochs(eeg_ref, fs=target_fs, epoch_len_s=epoch_len_s)

print("QC...")
clean_epochs = [ep for ep in epochs if not is_bad_epoch(ep, target_fs, **qc_params)]
clean_epochs = np.array(clean_epochs)
print(f"Kept {len(clean_epochs)} clean epochs / {len(epochs)} total.")

# ----------------- Compute PMP -----------------
pmp_vals = []
channel_range = range(min(channels_to_use, clean_epochs.shape[1]))

print(f"Computing PMP on {len(clean_epochs)} epochs × {len(channel_range)} channels...")
for e, ep in enumerate(tqdm(clean_epochs, desc="Epochs", unit="epoch")):
    for ch in channel_range:
        val = pmp_epoch(ep[ch], fs=target_fs, m=m, local_min_dist=local_min_dist)
        pmp_vals.append({"epoch": e, "channel": ch+1, "pMP": val})

df_pmp = pd.DataFrame(pmp_vals)
df_pmp.to_csv(OUT_DIR / "df_pmp.csv", index=False)
print("✅ Saved PMP results to df_pmp.csv")

# ----------------- Figure A -----------------
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(raw_eeg[0,:2000])
plt.title("Fig. A – Raw EEG (Ch1, 2s)")
plt.subplot(2,1,2)
plt.plot(eeg_ref[0,:2000])
for b in np.arange(0,2000,int(epoch_len_s*target_fs)):
    plt.axvline(b, color='red', linestyle='--', alpha=0.5)
plt.title("Preprocessed EEG (Ch1, 2s, with epochs)")
plt.tight_layout()
plt.savefig(OUT_DIR/"figA_raw_vs_preproc.png", dpi=300)

# ----------------- Figure B -----------------
f, Pxx = welch(eeg_ref[0], fs=target_fs, nperseg=512)
plt.figure(figsize=(8,4))
plt.semilogy(f, Pxx)
plt.axvspan(8,12,color="orange",alpha=0.3,label="Alpha")
plt.axvspan(13,30,color="green",alpha=0.3,label="Beta")
plt.title("Fig. B – PSD of Ch1")
plt.xlabel("Hz"); plt.ylabel("Power"); plt.legend()
plt.savefig(OUT_DIR/"figB_psd.png", dpi=300)

# ----------------- Figure C -----------------
sig = clean_epochs[0,0]
dp = mass(sig[:m], sig)
plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(sig); plt.title("Epoch (Ch1)")
plt.subplot(3,1,2); plt.plot(dp); plt.title("Matrix Profile (MP)")
plt.subplot(3,1,3); plt.plot(-dp); plt.title("In-Phase MP (PMP)")
plt.tight_layout()
plt.savefig(OUT_DIR/"figC_MP_vs_PMP.png", dpi=300)



# ----------------- Figure D -----------------
if not df_pmp.empty:
    pivot = df_pmp.pivot_table(index="channel", values="pMP", aggfunc="mean")
    zvals = (pivot - pivot.mean()) / pivot.std()
    plt.figure(figsize=(10,6))
    sns.heatmap(zvals, cmap="coolwarm", center=0, cbar_kws={"label":"z-score"})
    plt.title("Fig. D – Per-channel PMP features (this subject)")
    plt.xlabel("pMP (z-scored)")
    plt.ylabel("Channel")
    plt.savefig(OUT_DIR/"figE_feature_heatmap.png", dpi=300)
    plt.close()
else:
    print("⚠ df_pmp is empty, skipping Fig. E.")


print(f"✅ Saved workflow figures in {OUT_DIR}")
