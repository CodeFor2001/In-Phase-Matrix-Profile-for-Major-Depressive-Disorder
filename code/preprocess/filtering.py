#!/usr/bin/env python3
"""
filtering.py - Band-pass filter (2–47 Hz) + resample to target_fs
Implements zero-phase FIR filtering for EEG.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


import numpy as np
from scipy.signal import firwin, filtfilt, resample_poly
from utils.paths import load_settings
from utils.repro import log_run


def design_fir_bandpass(hp, lp, fs, numtaps=None):
    """
    Design a linear-phase FIR bandpass filter.
    hp, lp: cutoff frequencies in Hz
    fs: sampling rate
    numtaps: filter length; if None, choose ~3*fs/hp for steepness
    """
    if numtaps is None:
        numtaps = int(3 * fs / hp)
        if numtaps % 2 == 0:
            numtaps += 1  # ensure odd for symmetric FIR
    taps = firwin(numtaps, [hp, lp], pass_zero=False, fs=fs)
    return taps


def apply_filter(data, taps):
    """
    Apply zero-phase FIR to EEG data.
    data: ndarray (n_channels, n_samples)
    taps: FIR coefficients
    Returns filtered array.
    """
    return filtfilt(taps, [1.0], data, axis=-1)


def resample(data, orig_fs, target_fs):
    """
    Resample data from orig_fs to target_fs using polyphase filtering.
    """
    # Fractional resample factors
    from math import gcd
    g = gcd(int(target_fs), int(orig_fs))
    up = target_fs // g
    down = orig_fs // g
    return resample_poly(data, up, down, axis=-1)


def preprocess_filter_resample(eeg_data, orig_fs, hp, lp, target_fs):
    """
    Apply band-pass filter and resample.
    eeg_data: ndarray (n_channels, n_samples)
    Returns processed array.
    """
    taps = design_fir_bandpass(hp, lp, orig_fs)
    filtered = apply_filter(eeg_data, taps)
    if orig_fs != target_fs:
        filtered = resample(filtered, orig_fs, target_fs)
    return filtered


if __name__ == "__main__":
    # Self-test with synthetic data
    log_run("filtering_test")

    cfg = load_settings()
    orig_fs = cfg["sampling"]["orig_fs"]
    target_fs = cfg["sampling"]["target_fs"]
    hp = cfg["filter"]["hp"]
    lp = cfg["filter"]["lp"]

    # Make synthetic EEG: 10 Hz + 60 Hz noise
    dur = 5  # seconds
    t = np.arange(0, dur, 1/orig_fs)
    sig = np.sin(2*np.pi*10*t) + 0.2*np.sin(2*np.pi*60*t)
    eeg = np.stack([sig, sig*0.5])  # 2 channels

    proc = preprocess_filter_resample(eeg, orig_fs, hp, lp, target_fs)
    print(f"Original shape: {eeg.shape}, FS={orig_fs}")
    print(f"Processed shape: {proc.shape}, FS={target_fs}")
