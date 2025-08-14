#!/usr/bin/env python3
"""
hfd.py - Higuchi's Fractal Dimension (HFD) computation
Implements kmax=8 as described in Uudeberg et al. (2024).
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.paths import load_settings
from utils.repro import log_run


def higuchi_fd(x, kmax):
    """
    Compute Higuchi Fractal Dimension for a 1D signal.

    x: 1D numpy array
    kmax: maximum k (integer)
    """
    N = len(x)
    Lmk = np.zeros((kmax, kmax))

    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            lk = 0.0
            for i in range(1, len(idx)):
                lk += abs(x[idx[i]] - x[idx[i - 1]])
            # normalize length
            norm_factor = (N - 1) / ( (len(idx) - 1) * k )
            Lm.append(lk * norm_factor)
        Lmk[k - 1] = np.mean(Lm)
    # Linear fit to log-log values
    ln_k = np.log(1.0 / np.arange(1, kmax + 1))
    ln_L = np.log(np.mean(Lmk, axis=1))
    slope, _ = np.polyfit(ln_k, ln_L, 1)
    return slope


def hfd_epoch(epoch_1d, kmax=8):
    """
    Compute HFD for a single-channel epoch.
    epoch_1d: 1D numpy array
    """
    return higuchi_fd(epoch_1d, kmax)


if __name__ == "__main__":
    log_run("hfd_test")
    cfg = load_settings()
    kmax = cfg["hfd"]["kmax"]

    # Synthetic test: random noise should give high FD close to 2
    fs = cfg["sampling"]["target_fs"]
    dur = 20.48
    t = np.linspace(0, dur, int(dur*fs), endpoint=False)
    sig = np.sin(2*np.pi*10*t) + 0.5*np.random.randn(len(t))

    val = hfd_epoch(sig, kmax=kmax)
    print(f"HFD value (10Hz + noise): {val:.4f}")
