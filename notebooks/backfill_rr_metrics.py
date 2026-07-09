"""Backfill median_rr_ms and pct_near_floor into all_animals_features.csv.

Re-runs R-peak detection (identical pipeline to notebook 02) for every animal
in per_file_audit.csv, computes the two new metrics from the resulting RR
interval array, and patches them into all_animals_features.csv.

Rules:
- Additive only: no other column is touched, no row is added or removed.
- No status reclassification.
- Covers all animals currently in all_animals_features.csv (status=OK).
- Uses identical constants, filter, and detector as notebook 02 so the
  new columns are guaranteed consistent with rr_mean_ms / rr_std_ms.

Run once from the project root:
    python notebooks/backfill_rr_metrics.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

# ── Constants — must match notebook 02 exactly ────────────────────────────────
FS               = 1000
BANDPASS_LOW_HZ  = 0.5
BANDPASS_HIGH_HZ = 150.0
BANDPASS_ORDER   = 2
REFRACTORY_MS    = 55
HR_ACCEPT_LOW    = 300
HR_ACCEPT_HIGH   = 700

FLOOR_MS         = 55.0
FLOOR_ZONE_MS    = FLOOR_MS + 10.0   # RR <= 65 ms counts as "near floor"


# ── Helper functions (identical to notebook 02) ───────────────────────────────

def parse_header_and_layout(path):
    info = {}
    n_header = 0
    first_data_row = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.lstrip()
            if not s or s[0].isdigit() or s[0] in "+-.":
                first_data_row = line.rstrip("\n")
                break
            n_header += 1
            if "=" in line:
                key, _, rest = line.partition("=")
                info[key.strip()] = rest.strip("\n").strip("\t").split("\t")
    titles     = [t.strip() for t in info.get("ChannelTitle", []) if t.strip()]
    n_channels = max(len(titles), 1)
    n_cols_first_row = len(first_data_row.split("\t")) if first_data_row else (n_channels + 1)
    lead_cols = max(1, n_cols_first_row - n_channels)
    ch3_idx = next((i for i, t in enumerate(titles) if t.lower() == "channel 3"), 0)
    ecg_col = lead_cols + ch3_idx
    return n_header, lead_cols, ecg_col, titles


def load_ecg_segment(path, n_header, ecg_col, base_start, base_end):
    """Stream the file and return only the baseline segment as a float array."""
    voltages = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(n_header):
            f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= ecg_col:
                voltages.append(np.nan)
                continue
            try:
                voltages.append(float(parts[ecg_col]))
            except ValueError:
                voltages.append(np.nan)
    voltage = np.asarray(voltages, dtype=float)
    if np.isnan(voltage).any():
        idx = np.arange(len(voltage))
        good = ~np.isnan(voltage)
        if good.any():
            voltage = np.interp(idx, idx[good], voltage[good])
    end = min(int(base_end), len(voltage))
    return voltage[int(base_start):end]


def bandpass_filter(x):
    nyq = 0.5 * FS
    b, a = butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ / nyq, BANDPASS_HIGH_HZ / nyq], btype="band")
    y = filtfilt(b, a, x)
    for f0 in (50, 100, 150):
        bn, an = iirnotch(f0 / (FS / 2), Q=30)
        y = filtfilt(bn, an, y)
    return y


def _peaks_with_mad(sig):
    if sig.size == 0:
        return np.array([], dtype=int), np.nan
    med = float(np.median(sig))
    mad = float(np.median(np.abs(sig - med)))
    if mad <= 0:
        return np.array([], dtype=int), np.nan
    height     = med + 4 * mad
    prominence = max(0.3 * mad, 0.005)
    distance   = int(REFRACTORY_MS * FS / 1000)
    peaks, _   = find_peaks(sig, height=height, distance=distance, prominence=prominence)
    return peaks, height


def _hr_from_peaks(peaks):
    if len(peaks) < 2:
        return np.nan
    return 60000.0 / float(np.mean(np.diff(peaks) * (1000.0 / FS)))


def detect_r_peaks(sig):
    pos_peaks, _ = _peaks_with_mad(sig)
    pos_hr = _hr_from_peaks(pos_peaks)
    if not np.isnan(pos_hr) and HR_ACCEPT_LOW <= pos_hr <= HR_ACCEPT_HIGH:
        return pos_peaks, False, pos_hr
    inv_peaks, _ = _peaks_with_mad(-sig)
    inv_hr = _hr_from_peaks(inv_peaks)
    if not np.isnan(inv_hr) and HR_ACCEPT_LOW <= inv_hr <= HR_ACCEPT_HIGH:
        return inv_peaks, True, inv_hr
    def _dist(hr):
        if np.isnan(hr): return np.inf
        return max(0, HR_ACCEPT_LOW - hr, hr - HR_ACCEPT_HIGH)
    if _dist(inv_hr) < _dist(pos_hr):
        return inv_peaks, True, inv_hr
    return pos_peaks, False, pos_hr


# ── Main backfill ─────────────────────────────────────────────────────────────

def main():
    audit_path   = OUTPUTS_DIR / "per_file_audit.csv"
    features_path = OUTPUTS_DIR / "all_animals_features.csv"

    audit    = pd.read_csv(audit_path)
    features = pd.read_csv(features_path)

    # Only process animals that are in the features CSV (status=OK)
    ok_ids = set(features["animal_id"].tolist())
    audit_ok = audit[audit["animal_id"].isin(ok_ids)].copy()

    print(f"Animals in all_animals_features.csv : {len(ok_ids)}")
    print(f"Audit rows matched                  : {len(audit_ok)}")
    print()

    results = {}   # animal_id -> (median_rr_ms, pct_near_floor)
    errors  = []

    for _, row in audit_ok.iterrows():
        animal_id  = int(row["animal_id"])
        fname      = str(row["file"])
        base_start = int(row["base_start"])
        base_end   = int(row["base_end"])
        ecg_col    = int(row["ecg_column"])

        # Find the data file (audit stores just the filename, not the path)
        candidates = list(DATA_DIR.glob(fname))
        if not candidates:
            # Try case-insensitive match (Windows glob is already case-insensitive,
            # but be explicit on other platforms)
            candidates = [p for p in DATA_DIR.glob("*.txt") if p.name == fname]
        if not candidates:
            errors.append(f"  MISSING FILE  A{animal_id}: {fname}")
            continue

        path = candidates[0]
        try:
            n_header, _, _, _ = parse_header_and_layout(path)
            segment = load_ecg_segment(path, n_header, ecg_col, base_start, base_end)

            if segment.size < 2 * FS:
                errors.append(f"  SHORT SEGMENT A{animal_id}: {segment.size} samples")
                continue

            filtered = bandpass_filter(segment)
            peaks, inverted, hr = detect_r_peaks(filtered)
            if inverted:
                filtered = -filtered
                peaks, _, hr = detect_r_peaks(filtered)  # re-detect on corrected signal

            if len(peaks) < 2:
                errors.append(f"  TOO FEW PEAKS A{animal_id}: {len(peaks)} peaks")
                continue

            rr_ms = np.diff(peaks) * (1000.0 / FS)
            median_rr  = float(np.median(rr_ms))
            pct_floor  = float(100.0 * np.mean(rr_ms <= FLOOR_ZONE_MS))
            results[animal_id] = (median_rr, pct_floor)

        except Exception as exc:
            errors.append(f"  ERROR A{animal_id}: {type(exc).__name__}: {exc}")

    print(f"Successfully computed : {len(results)} animals")
    if errors:
        print(f"Errors / skipped      : {len(errors)}")
        for e in errors:
            print(e)
    print()

    # ── Patch features CSV ────────────────────────────────────────────────────
    median_col = []
    pct_col    = []
    for aid in features["animal_id"]:
        aid = int(aid)
        if aid in results:
            median_col.append(results[aid][0])
            pct_col.append(results[aid][1])
        else:
            median_col.append(float("nan"))
            pct_col.append(float("nan"))

    features.insert(
        features.columns.get_loc("rr_std_ms") + 1,
        "median_rr_ms",
        median_col,
    )
    features.insert(
        features.columns.get_loc("median_rr_ms") + 1,
        "pct_near_floor",
        pct_col,
    )

    n_populated = sum(1 for v in median_col if not np.isnan(v))
    n_missing   = sum(1 for v in median_col if np.isnan(v))
    print(f"median_rr_ms / pct_near_floor populated : {n_populated} rows")
    print(f"Still NaN (file missing or error)       : {n_missing} rows")
    print()

    features.to_csv(features_path, index=False)
    print(f"Saved: {features_path}")

    # ── Cross-check against previously-reported values ────────────────────────
    print()
    print("=== Spot-check: reported vs newly-computed ===")
    prev = {
        # animal: (previously_reported_median_ms, previously_reported_pct)
        115: (80.0, 1.2),
        201: (121.0, 1.4),
        146: (151.0, 0.7),
        106: (138.0, None),
    }
    for aid, (exp_med, exp_pct) in prev.items():
        if aid in results:
            got_med, got_pct = results[aid]
            med_ok = abs(got_med - exp_med) <= 5
            pct_ok = (exp_pct is None) or (abs(got_pct - exp_pct) <= 2)
            flag = "OK" if (med_ok and pct_ok) else "MISMATCH"
            pct_str = f"{got_pct:.2f}%" if exp_pct is not None else f"{got_pct:.2f}% (no prior)"
            print(f"  A{aid}: median={got_med:.1f}ms (expect ~{exp_med})  "
                  f"pct_near_floor={pct_str}  [{flag}]")
        else:
            print(f"  A{aid}: NOT COMPUTED (check errors above)")


if __name__ == "__main__":
    main()
