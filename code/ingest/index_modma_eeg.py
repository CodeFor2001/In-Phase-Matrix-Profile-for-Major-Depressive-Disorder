#!/usr/bin/env python3
"""
index_modma_eeg.py - index MODMA .mat resting-state EEG files and join with subject metadata.
"""

import re
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/MODMA_EEG")
EXCEL_PATH = RAW_DIR / "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx"
OUT_CSV = Path("data/interim/modma_eeg_index.csv")

def load_subject_sheet(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Sheet1", engine="openpyxl")
    colmap = {
        "subject id": "subject_id",
        "type": "label",
        "age": "age",
        "gender": "gender",
        "education（years）": "education_years",
        "PHQ-9": "PHQ9",
        "CTQ-SF": "CTQ_SF",
        "LES": "LES",
        "SSRS": "SSRS",
        "GAD-7": "GAD7",
        "PSQI": "PSQI",
    }
    df = df.rename(columns=colmap)
    keep = list(colmap.values())
    df = df[keep]
    df["subject_id"] = df["subject_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    return df

def parse_filename(fname: str):
    # Expected like: a02010002rest_20150416_1017mat.mat or 02010002rest 20150416 1017.mat
    # Extract first 8 digits
    m = re.search(r"(\d{8})", fname)
    raw_id = m.group(1) if m else None

    # Extract date/time if pattern present
    m2 = re.search(r"rest[_\s]+(\d{8})[_\s]+(\d{4})", fname)
    date = m2.group(1) if m2 else None
    time = m2.group(2) if m2 else None
    return raw_id, date, time

def normalize_subject_id(raw_id: str) -> str:
    try:
        return str(int(raw_id))  # drop leading zeros
    except Exception:
        return raw_id

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    subj_df = load_subject_sheet(EXCEL_PATH)
    subj_map = subj_df.set_index("subject_id").to_dict(orient="index")

    rows = []
    for p in RAW_DIR.glob("*.mat"):
        fname = p.name
        raw_id, date, time = parse_filename(fname)
        if not raw_id:
            print(f"[WARN] Could not parse subject id from {fname}")
            continue
        subject_id = normalize_subject_id(raw_id)
        meta = subj_map.get(subject_id, {})
        row = {
            "subject_id": subject_id,
            "filepath": str(p.resolve()),
            "filetype": "mat",
            "date": date,
            "time": time,
            "label": meta.get("label"),
            "age": meta.get("age"),
            "gender": meta.get("gender"),
            "education_years": meta.get("education_years"),
            "PHQ9": meta.get("PHQ9"),
            "CTQ_SF": meta.get("CTQ_SF"),
            "LES": meta.get("LES"),
            "SSRS": meta.get("SSRS"),
            "GAD7": meta.get("GAD7"),
            "PSQI": meta.get("PSQI"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ No .mat files indexed. Check RAW_DIR and extensions.")
        return

    missing = df["label"].isna().sum()
    if missing:
        print(f"[WARN] {missing} files missing label (MDD/HC) from Excel. They will still be indexed.")

    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Indexed {len(df)} .mat files → {OUT_CSV}")

if __name__ == "__main__":
    main()
