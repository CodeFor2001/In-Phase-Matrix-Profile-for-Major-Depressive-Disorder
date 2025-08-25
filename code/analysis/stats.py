#!/usr/bin/env python3

import re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

# Configurable paths (relative to project root)
PROJ_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJ_ROOT / "data" / "processed" / "features_pmp_hfd.csv"
LABELS_CSV = PROJ_ROOT / "data" / "raw" / "labels.csv"
OUT_DIR = PROJ_ROOT / "data" / "processed"
OUT_FULL = OUT_DIR / "stats_channelwise.csv"
OUT_SUMMARY = OUT_DIR / "stats_summary.csv"

# Feature suffixes to evaluate
FEATURES = ["pMP", "HFD"]

def discover_channels(df, feature_suffixes):
    """
    Discover channel indices by scanning df columns that match r'^Ch(\d+)_(feat)$'
    for any feat in feature_suffixes, returning a sorted, de-duplicated list of ints.
    """
    chan_set = set()
    # Precompile case-insensitive patterns once
    pats = [re.compile(rf"^Ch(\d+)_({re.escape(feat)})$", re.IGNORECASE) for feat in feature_suffixes]
    for col in df.columns:
        c2 = col.strip().replace("\ufeff", "").replace("\u200b", "")
        for pat in pats:
            m = pat.match(c2)
            if m:
                chan_set.add(int(m.group(1)))
                break
    return sorted(chan_set)

def mw_auc_from_u(u_stat, n1, n2):
    # AUC == Common Language Effect Size for MWU
    return u_stat / (n1 * n2)

def modified_bonferroni(pvals_array, alpha=0.05):
    import numpy as np
    p = np.array(pvals_array, dtype=float)
    p_bonf = np.full_like(p, np.nan, dtype=float)
    frontier_mask = np.zeros(len(p), dtype=bool)

    # FIX: np.where returns a tuple, so take the [0] element to get the indices array
    finite_idx = np.where(np.isfinite(p))[0]
    if finite_idx.size == 0:
        return p_bonf, frontier_mask

    p_finite = p[finite_idx]
    order = np.argsort(p_finite)  # ascending raw p
    p_sorted = p_finite[order]
    t = len(p_sorted)

    # corrected p-values in sorted order
    p_bonf_sorted = np.empty_like(p_sorted)
    for rank, rawp in enumerate(p_sorted, start=1):
        p_bonf_sorted[rank - 1] = rawp * (t + 1 - rank)

    # map corrected p back to original position order
    p_bonf_finite = np.empty_like(p_sorted)
    p_bonf_finite[order.argsort()] = p_bonf_sorted
    p_bonf[finite_idx] = p_bonf_finite

    # frontier up to first corrected p > alpha (in ascending raw p)
    crossed = False
    for raw_order_idx in order:
        global_idx = finite_idx[raw_order_idx]
        if crossed:
            break
        if np.isfinite(p_bonf[global_idx]) and p_bonf[global_idx] <= alpha:
            frontier_mask[global_idx] = True
        else:
            crossed = True

    return p_bonf, frontier_mask

def main():
    # Load data
    df_feat = pd.read_csv(FEATURES_CSV)
    df_lab = pd.read_csv(LABELS_CSV)

    # Normalize subject keys
    df_feat["subject"] = df_feat["subject"].astype(str).str.strip()
    df_lab["subject"] = df_lab["subject"].astype(str).str.strip()

    # Align subjects present in both files
    keep = set(df_feat["subject"]).intersection(set(df_lab["subject"]))
    df_feat = df_feat[df_feat["subject"].isin(keep)].copy()
    df_lab = df_lab[df_lab["subject"].isin(keep)].copy()

    # Merge labels
    df = df_feat.merge(df_lab[["subject", "group"]], on="subject", how="inner")

    # Identify groups
    groups = df["group"].unique().tolist()
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups, found {groups}")
    if "control" in groups and "MDD" in groups:
        g1, g2 = "control", "MDD"
    else:
        groups.sort()
        g1, g2 = groups[0], groups[1]

    # Discover channel indices from header across all requested features
    ch_indices = discover_channels(df, FEATURES)
    if not ch_indices:
        raise ValueError("No channel feature columns found matching Ch_(pMP|HFD).")

    # Per-channel stats
    rows = []
    for ch in ch_indices:
        ch_row = {"channel": ch}
        for feat_name in FEATURES:
            col = f"Ch{ch}_{feat_name}"
            if col not in df.columns:
                # Skip if this feature column not present for this channel
                continue
            # Extract per-group data (drop NaNs)
            x = df.loc[df["group"] == g1, col].dropna().values
            y = df.loc[df["group"] == g2, col].dropna().values
            n1, n2 = len(x), len(y)
            if n1 < 3 or n2 < 3:
                u_stat = np.nan
                p_val = np.nan
                auc = np.nan
                med1 = np.nan
                med2 = np.nan
            else:
                u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided")
                auc = mw_auc_from_u(u_stat, n1, n2)
                med1 = float(np.median(x))
                med2 = float(np.median(y))
            ch_row.update({
                f"{feat_name}_U": u_stat,
                f"{feat_name}_p": p_val,
                f"{feat_name}_AUC": auc,
                f"{feat_name}_median_{g1}": med1,
                f"{feat_name}_median_{g2}": med2,
                f"{feat_name}_n_{g1}": n1,
                f"{feat_name}_n_{g2}": n2,
            })
        rows.append(ch_row)

    df_stats = pd.DataFrame(rows).sort_values("channel").reset_index(drop=True)

    # Apply modified Bonferroni within each feature across channels
    alpha = 0.05
    for feat_name in FEATURES:
        raw_p = df_stats.get(f"{feat_name}_p", pd.Series(dtype=float)).values
        if raw_p.size == 0:
            # Create empty columns if none exist for this feature
            df_stats[f"{feat_name}_pBonf"] = np.nan
            df_stats[f"{feat_name}_sig_p<0.05"] = False
            df_stats[f"{feat_name}_sig_pBonf<0.05"] = False
            df_stats[f"{feat_name}_sig_frontier"] = False
            continue
        p_bonf, frontier = modified_bonferroni(raw_p, alpha=alpha)
        df_stats[f"{feat_name}_pBonf"] = p_bonf
        df_stats[f"{feat_name}_sig_p<0.05"] = (df_stats[f"{feat_name}_p"] < alpha)
        df_stats[f"{feat_name}_sig_pBonf<0.05"] = (df_stats[f"{feat_name}_pBonf"] < alpha)
        df_stats[f"{feat_name}_sig_frontier"] = frontier

    # Save full table
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(OUT_FULL, index=False)

    # Build compact summary
    summary_rows = []
    for feat_name in FEATURES:
        pcol = f"{feat_name}_p"
        auc_col = f"{feat_name}_AUC"
        avail = int(df_stats[pcol].notna().sum()) if pcol in df_stats else 0
        sig_unc = int(df_stats.get(f"{feat_name}_sig_p<0.05", pd.Series(False)).fillna(False).sum())
        sig_corr = int(df_stats.get(f"{feat_name}_sig_pBonf<0.05", pd.Series(False)).fillna(False).sum())
        sig_front = int(df_stats.get(f"{feat_name}_sig_frontier", pd.Series(False)).sum())
        if auc_col in df_stats and np.isfinite(df_stats[auc_col]).any():
            best_idx = df_stats[auc_col].idxmax()
            best_auc = float(df_stats.loc[best_idx, auc_col])
            best_ch = int(df_stats.loc[best_idx, "channel"])
        else:
            best_auc = np.nan
            best_ch = np.nan
        summary_rows.append({
            "feature": feat_name,
            "channels_tested": avail,
            "significant_p<0.05": sig_unc,
            "significant_pBonf<0.05": sig_corr,
            "significant_frontier": sig_front,
            "best_channel_by_AUC": best_ch,
            "best_AUC": best_auc,
        })
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

    # Console report
    print("Groups:", g1, "vs", g2)
    print("Subjects in analysis:", len(df["subject"].unique()))
    print("Saved:")
    print(" -", OUT_FULL)
    print(" -", OUT_SUMMARY)

if __name__ == "__main__":
    main()
