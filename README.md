data/
  ├── raw/
  ├── interim/
  ├── processed/
  └── logs/
code/
  ├── config/
  │    └── settings.yaml
  ├── ingest/
  │    ├── download_modma.py
  │    └── index_bids.py
  ├── preprocess/
  │    ├── filtering.py
  │    ├── rest_ref.py
  │    ├── epoching.py
  │    └── quality.py
  ├── features/
  │    ├── pmp.py
  │    ├── hfd.py
  │    └── aggregate.py
  ├── analysis/
  │    ├── stats.py
  │    └── classify.py
  └── utils/
       ├── paths.py
       ├── io.py
       └── repro.py
notebooks/
  └── 01_sanity_checks.ipynb
requirements.txt

Steps to implement
1) Ingest and index MODMA
code/ingest/download_modma.py

Uses openneuro-py to fetch ds004217 into data/raw/.

code/ingest/index_bids.py

Validates BIDS and emits a CSV (data/interim/modma_index.csv) listing subjects, sessions, tasks (rest), runs, file paths.

Run order:

python code/ingest/download_modma.py

python code/ingest/index_bids.py

2) Preprocessing pipeline
code/preprocess/filtering.py

Zero-phase FIR 2–47Hz; resample to 200Hz.

code/preprocess/rest_ref.py

REST if available; else average reference fallback with a clear log note.

code/preprocess/epoching.py

From first 6min of resting-state eyes-closed, slice 20.48s epochs, select first 10 artifact-free per channel.

code/preprocess/quality.py

Simple artifact checks: amplitude threshold, flatline, and high-frequency power ratio; flags epochs/channels for rejection.

Outputs:

data/interim/sub-XX_epoched.npz

Arrays: epochs[ch, epoch_idx, samples], chan_names, fs

3) Feature extraction
code/features/pmp.py

Implements pMP with MASS-like FFT distance profile; uses local minima on distance profile to keep in-phase matches.

code/features/hfd.py

Higuchi FD with kmax=8.

code/features/aggregate.py

Median across 10 epochs → per-subject per-channel features; writes:

data/processed/PMP_features.csv

data/processed/HFD_features.csv

data/processed/labels.csv

4) Analysis
code/analysis/stats.py

Mann–Whitney U per channel, modified Bonferroni per paper; prints significant channels.

code/analysis/classify.py

LOOCV SVM per channel; reports best single-channel accuracy for pMP and HFD.
