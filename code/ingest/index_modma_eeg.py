#!/usr/bin/env python3
"""
repro.py - reproducibility helper
Logs run metadata: timestamp, settings hash, git commit, Python/pkg versions
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import yaml

from .paths import load_settings, get_path

def hash_file(path):
    """Return SHA1 hash of a file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit():
    """Return current git commit hash or None if not a git repo."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return None

def get_environment_info():
    """Collect Python version and key package versions."""
    import importlib.metadata as importlib_metadata
    pkgs = ["numpy", "scipy", "mne", "scikit-learn", "nolds", "peakutils", "pyyaml"]
    versions = {}
    for pkg in pkgs:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }

def log_run(module_name, extra_info=None):
    """
    Create a log file for this run in data/logs.
    module_name: short string, usually __file__ stem
    extra_info: optional dict with extra metadata
    """
    settings_path = Path(__file__).parents[1] / "config" / "settings.yaml"
    cfg_hash = hash_file(settings_path)
    commit = get_git_commit()
    env_info = get_environment_info()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%SZ")

    log_data = {
        "timestamp_utc": timestamp,
        "module": module_name,
        "settings_hash": cfg_hash,
        "git_commit": commit,
        "environment": env_info,
        "extra_info": extra_info or {},
    }

    # Save log
    logs_dir = get_path("logs", mkdir=True)
    log_file = logs_dir / f"{timestamp}_{module_name}.log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    print(f"[REPRO] Logged run metadata to {log_file}")
    return log_file

if __name__ == "__main__":
    # Self-test
    log_run("repro_test", extra_info={"note": "This is a test."})
