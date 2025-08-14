#!/usr/bin/env python3
"""
epoching.py - Slice continuous EEG into fixed 20.48s segments & apply QC.
Implements exactly the segmentation protocol from Uudeberg et al. (2024).
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.paths import load_settings, get_path
from utils.repro import log_run
from preprocess.quality import is_bad_epoch


def slice_epochs(eeg_data, fs, epoch_len_s):
    """
    Slice continuous data into non-overlapping epochs of length epoch_len_s.
    eeg_data: ndarray (n_channels, n_samples)
    Returns list of ndarray epochs.
    """
    samples_per_epoch = int(epoch_len_s * fs)
    n_total_samples = eeg_data.shape[1]
    n_epochs = n_total_samples // samples_per_epoch
    epochs = []
    for i in range(n_epochs):
        start = i * samples_per_epoch
        stop = start + samples_per_epoch
        epochs.append(eeg_data[:, start:stop])
    return epochs


def preprocess_and_epoch(eeg_data, fs, target_fs, epoch_len_s, n_epochs_keep, qc_params):
    """
    Given continuous EEG (already filtered & re-referenced), slice into epochs,
    run QC, and keep the first `n_epochs_keep` clean epochs.
    """
    # Slice
    epochs = slice_epochs(eeg_data, fs, epoch_len_s)

    clean_epochs = []
    for ep in epochs:
        if not is_bad_epoch(ep, fs, **qc_params):
            clean_epochs.append(ep)
        if len(clean_epochs) >= n_epochs_keep:
            break

    return np.array(clean_epochs)  # shape: (n_epochs, n_channels, epoch_len_samples)


if __name__ == "__main__":
    log_run("epoching_test")
    cfg = load_settings()

    # Config parameters
    total_minutes = cfg["epoching"]["total_minutes"]
    epoch_len_s = cfg["epoching"]["epoch_len_s"]
    n_epochs_keep = cfg["epoching"]["n_epochs_keep"]
    fs = cfg["sampling"]["target_fs"]

    # QC thresholds
    qc_params = dict(
        amp_thresh=100.0,
        var_thresh=1e-6,
        hf_ratio=0.5
    )

    # --- Synthetic test data ---
    total_samples = int(total_minutes * 60 * fs)
    n_channels = 4
    rng = np.random.default_rng(42)
    t = np.arange(total_samples) / fs
    # Simulated 10 Hz alpha rhythm + small noise
    sig = np.sin(2 * np.pi * 10 * t) + 0.01 * rng.standard_normal(total_samples)
    eeg = np.vstack([sig + 0.005 * rng.standard_normal(total_samples) for _ in range(n_channels)])

    epochs = preprocess_and_epoch(
        eeg_data=eeg,
        fs=fs,
        target_fs=fs,
        epoch_len_s=epoch_len_s,
        n_epochs_keep=n_epochs_keep,
        qc_params=qc_params
    )

    print(f"Total generated epochs: {epochs.shape[0]} (should be {n_epochs_keep} if all clean)")
    print(f"Epoch shape: {epochs.shape[1:]} (channels x samples)")
