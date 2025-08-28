#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

def load_epochs_and_labels(npz_folder: Path | str, labels_csv: Path | str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads epochs, labels, and group information from .npz files and a labels CSV.

    Args:
        npz_folder: Path to the folder containing .npz files for each subject.
                    Each .npz file should contain 'epochs' and 'subject' arrays.
        labels_csv: Path to the CSV file containing 'subject' and 'group' columns.

    Returns:
        A tuple containing:
        - X (np.ndarray): Concatenated epochs from all subjects. Shape (total_epochs, n_channels, n_timepoints).
        - y (np.ndarray): Corresponding labels for each epoch.
        - groups (np.ndarray): Corresponding subject IDs for each epoch, for cross-validation.
    """
    try:
        # Set subject as index for efficient lookup. Assume 'subject' column contains strings.
        labels_df = pd.read_csv(labels_csv, dtype={'subject': str}).set_index("subject")
    except FileNotFoundError:
        print(f"Error: Labels file not found at {labels_csv}")
        raise

    all_epochs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_groups: List[np.ndarray] = []

    npz_folder = Path(npz_folder)
    if not npz_folder.is_dir():
        print(f"Error: NPZ folder not found at {npz_folder}")
        return np.array([]), np.array([]), np.array([])

    for npz_file in sorted(npz_folder.glob("*.npz")):
        data = np.load(npz_file)
        epochs = data["epochs"]  # (n_epochs, n_channels, n_timepoints)

        # Extract subject_id, robust to how it was saved (scalar vs array)
        try:
            subject_id = str(np.asarray(data["subject"]).item())
        except (KeyError, ValueError):
            print(f"Warning: Could not extract a single subject ID from {npz_file.name}. Skipping.")
            continue

        try:
            y_label = labels_df.loc[subject_id, "group"]
            # In case of duplicate subject IDs in CSV, loc might return a Series
            if isinstance(y_label, pd.Series):
                y_label = y_label.iloc[0]
        except KeyError:
            print(f"Warning: Subject {subject_id} label not found in {labels_csv}")
            continue

        labels = np.repeat(y_label, epochs.shape[0])
        groups = np.repeat(subject_id, epochs.shape[0])
        all_epochs.append(epochs)
        all_labels.append(labels)
        all_groups.append(groups)

    if not all_epochs:
        print("Warning: No epochs were loaded. Returning empty arrays.")
        return np.array([]), np.array([]), np.array([])

    X = np.concatenate(all_epochs, axis=0)  # (total_epochs, channels, timepoints)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)
    return X, y, groups

if __name__ == "__main__":
    # This block is for demonstration and testing purposes.
    # It assumes the script is run from the project's root directory.
    # The .npz files are expected in 'data/interim/'
    npz_folder = "data/interim"
    # The labels CSV is expected in 'data/labels.csv'. Please update if your path is different.
    labels_csv = "data/raw/labels.csv"
    X, y, groups = load_epochs_and_labels(npz_folder, labels_csv)
    if X.size > 0:
        print(f"Loaded data shape: {X.shape}, labels shape: {y.shape}, groups shape: {groups.shape}")
    else:
        print("No data loaded.")
