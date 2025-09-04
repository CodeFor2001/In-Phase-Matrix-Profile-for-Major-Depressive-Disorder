
# In-Phase Matrix Profile for Major Depressive Disorder

This repository implements the **In-Phase Matrix Profile (pMP)** method for EEG-based detection of Major Depressive Disorder (MDD). It provides all code, scripts, feature extraction, and classifier pipelines for thorough, reproducible machine learning experiments on resting-state EEG data.

---

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Data Organization](#data-organization)
- [Usage](#usage)
- [Matrix Profile (pMP) Method](#matrix-profile-pmp-method)
- [Benchmark Features and Classifiers](#benchmark-features-and-classifiers)
- [Results and Reproducibility](#results-and-reproducibility)

---

## Introduction

Major Depressive Disorder (MDD) is typically diagnosed via clinical interviews and subjective measures. Time-series analysis of EEG has emerged as a promising, objective approach. The **in-phase matrix profile (pMP)** algorithm implemented here improves on classic fractal methods, delivering superior discrimination between MDD and controls in resting-state EEG.

---

## Key Features

- **EEG Preprocessing:** Scripts for filtering, epoching, referencing, artifact rejection.
- **Feature Extraction:** Includes pMP and Higuchi Fractal Dimension (HFD).
- **Classification:** SVM, MultiROCKET, HIVE-COTE 2.0; channelwise and multichannel evaluations.
- **Benchmarking:** Direct comparison between feature-based and raw-signal classifiers.
- **Validation:** LOO (Leave-One-Out), KFold, GroupKFold cross-validation on subject-level data.

---

## Installation

Clone and install dependencies using pip or conda:

git clone https://github.com/CodeFor2001/In-Phase-Matrix-Profile-for-Major-Depressive-Disorder.git
cd In-Phase-Matrix-Profile-for-Major-Depressive-Disorder
pip install -r requirements.txt

text

Or use a provided `environment.yml` for conda setup.

---

## Data Organization

data/
raw/ # Raw EEG and subject labels
interim/ # Preprocessed epochs (npz)
processed/ # CSV outputs: features, results
scripts/ # All scripts for data and analysis
docs/ # Additional documentation, references

text

---

## Usage

1. **Preprocessing:** Prepare EEG using the provided or custom scripts.
2. **Feature Extraction:** Run scripts to compute pMP and HFD for each subject/channel.
3. **Classification:**
   - Channelwise and multichannel classifiers:
     - SVM (LOO, KFold)
     - MultiROCKET, HIVE-COTE (where feasible)
   - Scripts: see `classify_loo_channelwise_feature.py`, `classify_svms_overall_feature.py`, etc.
4. **Analysis:** Generated CSVs in `data/processed/` contain scores for further analysis and visualization.

See the `scripts/` folder and docstrings for argument details and workflow.

---

## Matrix Profile (pMP) Method

The **in-phase matrix profile** (pMP) algorithm identifies self-similar, phase-aligned subsequences in EEG signals. It robustly handles EEG noise and timing, outperforming entropy/fractal features in resting-state MDD diagnosis.

- Sensitive to periodic brain rhythms.
- Discovers distinctive motifs affected in depression.
- Benchmark tests show higher accuracy than fractal dimension features.

For further reading, see [Nature Scientific Reports, 2024][citation].

---

## Benchmark Features and Classifiers

- Features: pMP, HFD, statistical/spectral baselines.
- Classifiers:
  - **Channelwise:** SVM (LOO, KFold), MultiROCKET, HIVE-COTE 2.0
  - **Multichannel:** SVM (concatenated features), MultiROCKET (multivariate), HC2
- Validation:
  - Leave-One-Out (subject-level)
  - Stratified KFold (class-balanced)
  - GroupKFold (to prevent subject leakage)

---

## Results and Reproducibility

All output CSVs with per-channel/multichannel scores are stored in `data/processed/`. Analysis routines are designed for full reproducibility from feature extraction to classifier validation.
