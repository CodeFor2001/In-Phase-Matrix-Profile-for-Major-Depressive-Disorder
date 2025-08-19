#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def _first_2d_numeric(mat_dict):
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
            return k, v
    return None, None

def _extract_fs(mat_dict, default=500.0):
    for key in ("fs", "Fs", "srate", "sr", "SRate"):
        if key in mat_dict:
            val = mat_dict[key]
            try:
                if np.isscalar(val):
                    return float(val)
                if isinstance(val, np.ndarray) and val.size == 1:
                    return float(val.ravel()[0])
            except Exception:
                pass
    return float(default)

def load_modma_mat(path: str | Path, drop_last_row_if_129=True):
    path = Path(path)
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    key, arr = _first_2d_numeric(mat)
    if key is None:
        raise ValueError(f"No 2D numeric array found in {path.name}")

    data = np.array(arr, dtype=np.float64)  # expect (channels x samples) or (samples x channels)

    # Ensure shape is (channels x samples)
    if data.ndim != 2:
        raise ValueError(f"Array in {path.name} is not 2D, got shape {data.shape}")
    if data.shape[0] < data.shape[1]:
        # likely already (channels x samples): e.g., 129 x 75189
        pass
    else:
        # transpose if it looks like (samples x channels)
        data = data.T

    n_ch, n_samp = data.shape

    # Drop the 129th row if present (common non-EEG row)
    if drop_last_row_if_129 and n_ch == 129:
        data = data[:128, :]
        n_ch = 128

    fs = _extract_fs(mat, default=500.0)
    channels = [f"Ch{i+1}" for i in range(n_ch)]

    return {"data": data, "fs": fs, "channels": channels}

if __name__ == "__main__":
    import sys
    info = load_modma_mat(sys.argv[1])
    print(f"Loaded: {info['data'].shape} channels x {info['data'].shape[1]} samples at fs={info['fs']}")
