#!/usr/bin/env python3
"""
rest_ref.py - EEG re-referencing utilities
- Average reference (CAR) implemented
- Placeholder for REST (to be added later via MNE)
"""

import numpy as np
from pathlib import Path

# Make code/ discoverable if running as a script
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.paths import load_settings
from utils.repro import log_run


def average_reference(eeg_data):
    """
    Common Average Reference (CAR).
    eeg_data: ndarray (n_channels, n_samples)
    Returns re-referenced array where channel mean across channels is ~0 at each time point.
    """
    if eeg_data.ndim != 2:
        raise ValueError("eeg_data must be 2D (n_channels, n_samples)")
    mean_across_ch = eeg_data.mean(axis=0, keepdims=True)
    return eeg_data - mean_across_ch


def apply_reference(eeg_data, method="average"):
    """
    Apply a reference method.
    Currently supported: 'average'
    """
    method = (method or "average").lower()
    if method == "average":
        return average_reference(eeg_data)
    else:
        raise NotImplementedError(f"Reference method '{method}' not implemented.")


if __name__ == "__main__":
    # Self-test on synthetic data
    log_run("rest_ref_test")

    cfg = load_settings()
    n_ch, n_samp = 4, 1000
    rng = np.random.default_rng(42)

    # Synthetic multi-channel signal: a shared 10 Hz + channel-specific noise
    t = np.linspace(0, 5, n_samp, endpoint=False)
    shared = np.sin(2*np.pi*10*t)
    eeg = np.vstack([shared + 0.05*rng.standard_normal(n_samp) for _ in range(n_ch)])

    reref = apply_reference(eeg, method="average")

    # Sanity checks
    avg_before = eeg.mean(axis=0).std()
    avg_after = reref.mean(axis=0).std()
    print(f"Std of across-channel mean before: {avg_before:.6f}")
    print(f"Std of across-channel mean after:  {avg_after:.6f}")
    print(f"Shape in/out: {eeg.shape} -> {reref.shape}")
