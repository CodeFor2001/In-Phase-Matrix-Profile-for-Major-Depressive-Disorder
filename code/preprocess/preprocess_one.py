#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from scipy.signal import firwin, filtfilt, resample_poly, welch
from scipy.io import loadmat
from scipy.integrate import trapezoid
import sys

# --------- Utilities ---------
def bandpass_filter(data, fs, lo=2.0, hi=47.0, numtaps=801):
    nyq = fs / 2.0
    b = firwin(numtaps, [lo/nyq, hi/nyq], pass_zero=False)
    return filtfilt(b, [1.0], data, axis=1)

def resample_to_200hz(data, fs):
    target_fs = 200.0
    from fractions import Fraction
    frac = Fraction(target_fs / fs).limit_denominator()
    up, down = frac.numerator, frac.denominator
    out = resample_poly(data, up, down, axis=1)
    return out, target_fs

def average_reference(data):
    mean = np.mean(data, axis=0, keepdims=True)
    return data - mean

def first_n_seconds(data, fs, n_sec):
    n = int(n_sec * fs)
    return data[:, :min(n, data.shape[1])]

def epoch_fixed(data, fs, epoch_len_s=20.48):
    nper = int(epoch_len_s * fs)  # 4096 when fs=200
    n_total = data.shape[1]
    n_epochs = n_total // nper
    if n_epochs == 0:
        return np.empty((0, data.shape, nper))
    ep = np.stack([data[:, i*nper:(i+1)*nper] for i in range(n_epochs)], axis=0)
    return ep

def qc_epochs(epochs, amp_thresh=150.0, var_thresh=1e-9, hf_ratio_thresh=0.6, fs=200.0):
    if epochs.size == 0:
        return np.array([], dtype=bool)

    n_epochs, n_ch, n_samp = epochs.shape
    keep = np.ones(n_epochs, dtype=bool)
    for i in range(n_epochs):
        ep = epochs[i]

        # amplitude check
        if np.any(np.abs(ep) > amp_thresh):
            keep[i] = False
            continue

        # flatline per channel
        if np.any(np.var(ep, axis=1) < var_thresh):
            keep[i] = False
            continue

        # HF ratio using Welch PSD on the average across channels
        sig = np.mean(ep, axis=0)
        f, pxx = welch(sig, fs=fs, nperseg=min(1024, n_samp))
        band = (f >= 2) & (f <= 47)
        hf = (f >= 40) & (f <= 47)
        band_power = trapezoid(pxx[band], f[band]) if np.any(band) else 0.0
        hf_power = trapezoid(pxx[hf], f[hf]) if np.any(hf) else 0.0
        ratio = (hf_power / band_power) if band_power > 0 else 0.0
        if ratio > hf_ratio_thresh:
            keep[i] = False
            continue

    return keep

# --------- Minimal .mat loader ---------
def load_modma_mat_simple(path, drop_last_row_if_129=True):
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    arr = None
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
            arr = v
            break
    if arr is None:
        raise ValueError("No 2D numeric array found")
    data = np.array(arr, dtype=np.float64)  # expect (channels x samples) or (samples x channels)
    if data.shape[0] >= data.shape[1]:
        data = data.T
    if data.shape[0] == 129 and drop_last_row_if_129:
        data = data[:128, :]
    fs = 500.0  # default, adjust if true fs known
    channels = [f"Ch{i+1}" for i in range(data.shape[0])]  # FIXED: use data.shape
    return data, fs, channels

# --------- Main single-subject run ---------
if __name__ == "__main__":
    # USAGE:
    # python code/preprocess/preprocess_one.py "data/raw/MODMA_EEG/02010002rest 20150416 1017..mat" 2010002

    in_path = Path(sys.argv[1])
    subject_id = sys.argv[2]

    # Load
    data, fs_in, channels = load_modma_mat_simple(str(in_path))

    # Filter
    data = bandpass_filter(data, fs_in, lo=2.0, hi=47.0)

    # Resample to 200Hz
    data, fs = resample_to_200hz(data, fs_in)

    # Avg re-reference
    data = average_reference(data)

    # First 6 minutes (or less if shorter)
    data = first_n_seconds(data, fs, n_sec=6*60)

    # Epoch into 20.48s
    epochs = epoch_fixed(data, fs, epoch_len_s=20.48)

    # QC and keep first 10 clean
    keep = qc_epochs(epochs, amp_thresh=150.0, var_thresh=1e-9, hf_ratio_thresh=0.6, fs=fs)
    clean_epochs = epochs[keep]
    if clean_epochs.shape[0] > 10:  
        clean_epochs = clean_epochs[:10]

    # Save
    out_dir = Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sub-{subject_id}_epochs.npz"
    np.savez(out_path, epochs=clean_epochs, channels=np.array(channels, dtype=object), subject=subject_id)
    print(f"Saved {clean_epochs.shape} epochs for subject {subject_id} → {out_path}")
