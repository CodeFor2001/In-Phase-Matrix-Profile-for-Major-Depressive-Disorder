#!/usr/bin/env python3
"""
quality.py - EEG artifact detection utilities
Implements:
1. Amplitude clamp check
2. Flatline detection
3. High-frequency noise ratio
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.paths import load_settings
from utils.repro import log_run
from scipy.signal import welch


def amplitude_check(epoch, threshold_uv=100.0):
    """
    Reject if absolute amplitude exceeds threshold (in microvolts).
    """
    return np.any(np.abs(epoch) > threshold_uv)


def flatline_check(epoch, var_threshold=1e-6):
    """
    Reject if signal variance is near zero (possible electrode dropout).
    """
    return np.var(epoch, axis=-1).min() < var_threshold


def highfreq_noise_check(epoch, fs, band=(40, 80), ratio_threshold=0.5):
    """
    Reject if high-frequency (40–80 Hz) power ratio exceeds threshold.
    """
    ch_ok = []
    for ch_data in epoch:
        freqs, psd = welch(ch_data, fs=fs, nperseg=min(len(ch_data), 512))
        total_power = np.sum(psd)
        hf_mask = (freqs >= band[0]) & (freqs <= band[1])
        hf_power = np.sum(psd[hf_mask])
        ratio = hf_power / total_power if total_power > 0 else 0
        ch_ok.append(ratio > ratio_threshold)  # True means bad
    return any(ch_ok)


def is_bad_epoch(epoch, fs, amp_thresh=100.0, var_thresh=1e-6, hf_ratio=0.5):
    """
    Run all QC checks; return True if epoch fails any.
    epoch: ndarray (n_channels, n_samples)
    """
    if amplitude_check(epoch, amp_thresh):
        return True
    if flatline_check(epoch, var_thresh):
        return True
    if highfreq_noise_check(epoch, fs, ratio_threshold=hf_ratio):
        return True
    return False


if __name__ == "__main__":
    # Self-test with synthetic epochs
    log_run("quality_test")

    cfg = load_settings()
    fs = cfg["sampling"]["target_fs"]

    # Create three synthetic 1-second epochs (channels x samples)
    ch, s_len = 2, fs
    clean = np.random.randn(ch, s_len) * 10  # ~10 µV
    high_amp = clean.copy()
    high_amp[0, 10] = 500  # spike

    flat = np.zeros((ch, s_len))

    noisy = clean.copy()
    noisy[0] += np.sin(2 * np.pi * 50 * np.linspace(0, 1, s_len)) * 20  # 50Hz

    for name, ep in [("clean", clean), ("high_amp", high_amp), ("flat", flat), ("noisy", noisy)]:
        print(f"{name:>8}: bad? {is_bad_epoch(ep, fs)}")
