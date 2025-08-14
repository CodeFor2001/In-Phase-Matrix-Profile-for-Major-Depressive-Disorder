#!/usr/bin/env python3
"""
pmp.py - In-phase Matrix Profile computation (Uudeberg et al. 2024)
Implements MASS-FFT distance profiles, local minima selection,
median-of-minima per query, average across queries.

References:
- Uudeberg et al., "In-phase Matrix Profile", Biomedical Signal Processing and Control, 88 (2024)
- MASS_V2 algorithm by Mueen et al.
"""

import numpy as np
from scipy.signal import find_peaks
from numpy.fft import fft, ifft
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.paths import load_settings
from utils.repro import log_run


# --------------------
# MASS algorithm helper
# --------------------
def mass(query, T):
    """
    Compute z-normalized Euclidean distance profile between query (length m)
    and all subsequences of T (length n). Returns length n-m+1.
    """
    m = len(query)
    n = len(T)
    if n < m:
        raise ValueError("Time series length n must be >= query length m.")

    # z-normalize query; handle zero std
    q = query.astype(float)
    q_mu = np.mean(q)
    q_sigma = np.std(q)
    if q_sigma == 0:
        # zero-variance query -> distances depend only on T variance; set to tiny to avoid NaN
        q_norm = (q - q_mu)
    else:
        q_norm = (q - q_mu) / q_sigma

    # cumulative stats for T
    cum_sum = np.cumsum(T)
    cum_sum2 = np.cumsum(T**2)
    sum_T = cum_sum[m-1:] - np.concatenate(([0.0], cum_sum[:-m]))
    sum_T2 = cum_sum2[m-1:] - np.concatenate(([0.0], cum_sum2[:-m]))
    mean_T = sum_T / m
    sigma_T = np.sqrt(np.maximum(sum_T2 / m - mean_T**2, 0))

    # FFT-based cross-correlation q with T
    # choose L as next power of 2 >= n + m - 1
    L = 1 << (int(np.ceil(np.log2(n + m - 1))))
    q_rev = q_norm[::-1]
    Q_fft = fft(np.pad(q_rev, (0, L - m)))
    T_fft = fft(np.pad(T, (0, L - n)))
    QT = np.real(ifft(T_fft * Q_fft))  # length L
    # valid cross-correlation segment indices m-1 ... m-1 + (n-m+1) - 1
    QT = QT[m-1:m-1 + (n - m + 1)]

    # z-normed distance profile:
    # dist^2 = 2m * (1 - (QT - m*mean_T*mean_q) / (m * sigma_T * sigma_q))
    # with our q normalized: mean_q = 0, sigma_q = 1 (if q_sigma > 0), else special-case
    if q_sigma == 0:
        # if query is constant, correlation term uses (QT - m*mean_T*mean_q)=QT (since mean_q=const)
        # but sigma_q=0 -> use direct formula fallback: distance between constant and subseq means/vars
        # Use ED(q, s) with q centered: ||s - mean(s)||^2 + const; simplest robust fallback:
        # Set correlation term to 0 so dist grows with sigma_T
        corr = np.zeros_like(QT)
    else:
        corr = (QT - m * mean_T * 0.0) / (sigma_T + 1e-12)

    # Convert correlation to distance squared per MASS:
    dist2 = 2 * (m - corr)
    # Where sigma_T==0 (flat subsequence), define distance based on difference to zero-variance:
    flat_mask = sigma_T < 1e-12
    if np.any(flat_mask):
        # If both are effectively flat after z-norm, distance is 0; else large
        dist2[flat_mask] = 2 * m  # maximal (safe) distance for undefined correlation

    dist = np.sqrt(np.maximum(dist2, 0))
    return dist



def pmp_epoch(epoch_1d, fs, m, local_min_dist=3):
    """
    Compute pMP for one 1D epoch (already filtered, re-referenced, artifact-free).
    epoch_1d: shape (n_samples,)
    fs: sampling frequency (Hz)
    m: subsegment length (samples)
    local_min_dist: minimal distance (in samples) between local minima peaks
    """
    n = len(epoch_1d)
    # Build all subsegments
    subs = np.lib.stride_tricks.sliding_window_view(epoch_1d, m)  # shape: (n-m+1, m)

    # Z-normalize each subsegment
    subs = (subs - subs.mean(axis=1, keepdims=True)) / subs.std(axis=1, keepdims=True)

    dp_medians = np.empty(n - m + 1)
    for i, q in enumerate(subs):
        dp = mass(q, epoch_1d)
        dp[i] = np.inf  # remove trivial match
        # negative peaks = in-phase matches
        peaks, _ = find_peaks(-dp, distance=local_min_dist)
        if len(peaks) == 0:
            dp_medians[i] = np.nan
        else:
            dp_medians[i] = np.median(dp[peaks])

    # pMP value for this epoch: mean of medians (ignoring NaNs)
    return np.nanmean(dp_medians)


if __name__ == "__main__":
    log_run("pmp_test")
    cfg = load_settings()

    fs = cfg["sampling"]["target_fs"]
    m = cfg["pmp"]["m"]
    local_min_dist = cfg["pmp"]["local_min_dist"]

    # Test on synthetic alpha-like signal
    dur_s = 20.48
    t = np.arange(0, dur_s, 1/fs)
    # 10 Hz sine wave + small noise
    sig = np.sin(2*np.pi*10*t) + 0.05*np.random.randn(len(t))

    pmp_val = pmp_epoch(sig, fs=fs, m=m, local_min_dist=local_min_dist)
    print(f"pMP value (synthetic 10Hz alpha-like signal): {pmp_val:.4f}")
