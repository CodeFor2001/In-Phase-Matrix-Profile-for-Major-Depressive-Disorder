#!/usr/bin/env python3
"""
paths.py - centralised path + config loader for the MODMA EEG pipeline
"""
import yaml
from pathlib import Path

# Cache config so it's loaded only once
_SETTINGS = None

def load_settings():
    global _SETTINGS
    if _SETTINGS is None:
        cfg_path = Path(__file__).parents[1] / "config" / "settings.yaml"
        with open(cfg_path, "r") as f:
            _SETTINGS = yaml.safe_load(f)
    return _SETTINGS

def get_path(*parts, mkdir=False):
    """
    Resolve a path relative to project root.
    mkdir=True will create the directory if it doesn't exist.
    """
    root = Path(load_settings()["data_root"]).expanduser()
    p = root.joinpath(*parts)
    if mkdir:
        p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_structure():
    """
    Make sure the key folders from settings.yaml exist.
    """
    cfg = load_settings()
    # Core folders
    for sub in ["raw", "interim", "processed", "logs"]:
        get_path(sub, mkdir=True)

if __name__ == "__main__":
    ensure_structure()
    print(f"✅ Data root confirmed at: {get_path()}")
