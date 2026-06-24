"""Diagnostic export: QT T-end placement for selected animals.

Reuses NB02 functions verbatim. Does NOT modify NB01, NB02, or any pipeline
constant. Outputs qt_template_export.csv (waveforms) and
qt_template_summary.csv (one row per animal) to outputs/.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

# --- Named constants -------------------------------------------------------
ANIMALS      = [115, 125, 157, 201, 232, 249]
DIAG_PRE_MS  = 100   # widened beat window before R (ms)
DIAG_POST_MS = 200   # widened beat window after R (ms)

# --- Project root (dynamic) ------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR    = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Pipeline constants — verbatim from NB02 (do not edit)
# ---------------------------------------------------------------------------
FS                  = 1000
BANDPASS_LOW_HZ     = 0.5
BANDPASS_HIGH_HZ    = 150.0
BANDPASS_ORDER      = 2
REFRACTORY_MS       = 55
MIN_RR_MS           = 75
BEATS_TO_AVERAGE    = 4
PRE_R_MS            = 100
POST_R_MS           = 150
HR_ACCEPT_LOW       = 300
HR_ACCEPT_HIGH      = 700

# Marker regex used by load_ecg_file
_MARKER_RE = re.compile(r"#([*1-3])")

# ---------------------------------------------------------------------------
# Helper functions — verbatim copies from NB02 cells
# ---------------------------------------------------------------------------

def parse_header_and_layout(path: Path):
    """Read header lines, then the first data row, and figure out:
         n_header   : number of header lines
         lead_cols  : number of leading time/date columns
         ecg_col    : 0-based column index of Channel 3 (the ECG mV channel)
       Lead cols are computed as (n_columns_in_first_data_row - n_channels_in_header)."""
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

    titles = [t.strip() for t in info.get("ChannelTitle", []) if t.strip()]
    n_channels = max(len(titles), 1)
    n_cols_first_row = len(first_data_row.split("\t")) if first_data_row else (n_channels + 1)
    lead_cols = max(1, n_cols_first_row - n_channels)

    ch3_idx = None
    for i, t in enumerate(titles):
        if t.lower() == "channel 3":
            ch3_idx = i
            break
    if ch3_idx is None:
        ch3_idx = 0
    ecg_col = lead_cols + ch3_idx
    return n_header, lead_cols, ecg_col, titles


def load_ecg_file(path: Path, n_header: int, ecg_col: int):
    """Stream the file. Return:
         voltage     : 1-D float mV array
         markers     : list of (row_index, full_annotation_text, prefix_char)
                       where prefix_char is '*', '1', '2', or '3'."""
    voltages = []
    markers  = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(n_header):
            f.readline()
        for line in f:
            stripped = line.rstrip("\n")
            parts = stripped.split("\t")
            if len(parts) <= ecg_col:
                continue
            try:
                v = float(parts[ecg_col])
            except ValueError:
                v = np.nan
            voltages.append(v)
            m = _MARKER_RE.search(stripped)
            if m:
                prefix = m.group(1)
                ann_text = stripped[m.start():].strip()
                markers.append((len(voltages) - 1, ann_text, prefix))

    voltage = np.asarray(voltages, dtype=float)
    if np.isnan(voltage).any():
        idx = np.arange(len(voltage))
        good = ~np.isnan(voltage)
        if good.any():
            voltage = np.interp(idx, idx[good], voltage[good])
    return voltage, markers


def bandpass_filter(x, fs=FS, low=BANDPASS_LOW_HZ, high=BANDPASS_HIGH_HZ, order=BANDPASS_ORDER):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    y = filtfilt(b, a, x)
    # Notch out 50/100/150 Hz mains interference (Irish mains = 50 Hz)
    for f0 in (50, 100, 150):
        bn, an = iirnotch(f0 / (fs / 2), Q=30)
        y = filtfilt(bn, an, y)
    return y


def _peaks_with_mad(sig, fs=FS, refractory_ms=REFRACTORY_MS):
    """Median + 4*MAD adaptive threshold (Adaptive Physiology-Driven Framework)."""
    if sig.size == 0:
        return np.array([], dtype=int), np.nan, np.nan
    med = float(np.median(sig))
    mad = float(np.median(np.abs(sig - med)))
    if mad <= 0:
        return np.array([], dtype=int), med, mad
    height     = med + 4 * mad
    prominence = max(0.3 * mad, 0.005)
    distance   = int(refractory_ms * fs / 1000)
    peaks, _   = find_peaks(sig, height=height, distance=distance, prominence=prominence)
    return peaks, height, prominence


def detect_r_peaks(sig, fs=FS):
    """Robust R-peak detection. Returns (peaks, inverted_flag, hr_bpm, threshold)."""
    def _hr(peaks):
        if len(peaks) < 2:
            return np.nan
        rr_ms = np.diff(peaks) * (1000.0 / fs)
        return 60000.0 / np.mean(rr_ms)

    pos_peaks, pos_h, _ = _peaks_with_mad(sig, fs)
    pos_hr = _hr(pos_peaks)
    pos_ok = HR_ACCEPT_LOW <= pos_hr <= HR_ACCEPT_HIGH if not np.isnan(pos_hr) else False
    if pos_ok:
        return pos_peaks, False, pos_hr, pos_h

    inv_peaks, inv_h, _ = _peaks_with_mad(-sig, fs)
    inv_hr = _hr(inv_peaks)
    inv_ok = HR_ACCEPT_LOW <= inv_hr <= HR_ACCEPT_HIGH if not np.isnan(inv_hr) else False
    if inv_ok:
        return inv_peaks, True, inv_hr, inv_h

    def _dist(hr):
        if np.isnan(hr):
            return np.inf
        if hr < HR_ACCEPT_LOW:
            return HR_ACCEPT_LOW - hr
        if hr > HR_ACCEPT_HIGH:
            return hr - HR_ACCEPT_HIGH
        return 0.0

    if _dist(inv_hr) < _dist(pos_hr):
        return inv_peaks, True, inv_hr, inv_h
    return pos_peaks, False, pos_hr, pos_h


def average_beats(sig, r_peaks, fs=FS,
                  pre_ms=PRE_R_MS, post_ms=POST_R_MS,
                  group_size=BEATS_TO_AVERAGE):
    pre  = int(pre_ms  * fs / 1000)
    post = int(post_ms * fs / 1000)
    win_len = pre + post
    t_ms = (np.arange(win_len) - pre) * (1000.0 / fs)
    beats = [sig[r - pre: r + post] for r in r_peaks
             if r - pre >= 0 and r + post <= len(sig)]
    if not beats:
        return None, t_ms, None
    beats = np.vstack(beats)
    n_full_groups = len(beats) // group_size
    if n_full_groups == 0:
        return beats.mean(axis=0), t_ms, beats
    grouped = beats[:n_full_groups * group_size].reshape(n_full_groups, group_size, win_len)
    return grouped.mean(axis=1).mean(axis=0), t_ms, beats


def _measure_one(sig, t_ms):
    """Measure R amplitude, QRS-FWHM, J wave, T wave, RT and QT on one beat
       (or the averaged template). Identical logic for template and per-beat."""
    r_idx = int(np.argmax(sig))
    r_amp = float(sig[r_idx])
    half = r_amp / 2.0
    left = r_idx
    while left > 0 and sig[left] > half:
        left -= 1
    right = r_idx
    while right < len(sig) - 1 and sig[right] > half:
        right += 1
    qrs_ms = float(t_ms[right] - t_ms[left])
    def window_max(s_ms, e_ms):
        mask = (t_ms >= s_ms) & (t_ms <= e_ms)
        if not mask.any():
            return np.nan, np.nan
        seg = sig[mask]; ts = t_ms[mask]
        k = int(np.argmax(seg))
        return float(seg[k]), float(ts[k])
    j_amp, _      = window_max(10, 30)
    t_amp, t_time = window_max(40, 80)
    rt_ms = float(t_time) if not np.isnan(t_time) else np.nan
    baseline = float(np.median(sig[t_ms < -70])) if (t_ms < -70).any() else 0.0
    tol = 0.05 * r_amp
    qt_ms = np.nan
    for i in np.where(t_ms >= 40)[0]:
        if abs(sig[i] - baseline) <= tol:
            qt_ms = float(t_ms[i])
            break
    return r_amp, qrs_ms, j_amp, t_amp, rt_ms, qt_ms


# ---------------------------------------------------------------------------
# Diagnostic processing
# ---------------------------------------------------------------------------

def process_animal(row):
    """Load file, reproduce the pipeline window, return (waveform_df, summary)."""
    path = DATA_DIR / str(row["source_file"])

    n_header, _lead_cols, ecg_col, _titles = parse_header_and_layout(path)
    voltage, _markers = load_ecg_file(path, n_header, ecg_col)

    # Slice the exact same window recorded in features_fallback
    ws = int(row["window_start"])
    we = int(row["window_end"])
    segment = voltage[ws:we]

    filtered = bandpass_filter(segment)

    peaks, inverted, hr, _ = detect_r_peaks(filtered)
    if inverted:
        filtered = -filtered

    rr_intervals = np.diff(peaks) * (1000.0 / FS)
    rr_mean = float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else np.nan

    # Widened window for this export only — pipeline unchanged
    template, t_ms, _beats = average_beats(
        filtered, peaks, pre_ms=DIAG_PRE_MS, post_ms=DIAG_POST_MS
    )

    r_amp, _qrs, _j, _t, _rt, qt_current = _measure_one(template, t_ms)
    baseline_mv = float(np.median(template[t_ms < -70])) if (t_ms < -70).any() else 0.0
    tol_mv = 0.05 * r_amp

    aid = int(row["animal_id"])
    wf_df = pd.DataFrame({
        "animal_id":  aid,
        "time_ms":    np.round(t_ms, 3),
        "voltage_mv": np.round(template, 6),
    })

    summary = {
        "animal_id":        aid,
        "n_beats":          int(len(peaks)),
        "rr_ms":            round(rr_mean, 2),
        "hr_bpm":           round(float(hr), 1),
        "r_amplitude_mv":   round(float(r_amp), 4),
        "baseline_mv":      round(baseline_mv, 4),
        "tol_mv":           round(tol_mv, 4),
        "qt_current_ms":    qt_current if not np.isnan(qt_current) else None,
    }
    return wf_df, summary


def main():
    fb = pd.read_csv(OUTPUTS_DIR / "features_fallback.csv")
    fb = fb[fb["animal_id"].isin(ANIMALS)].set_index("animal_id")

    wf_parts     = []
    summary_rows = []

    for aid in ANIMALS:
        row = fb.loc[aid]
        # squeeze in case loc returns a 1-row DataFrame
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        row = row.copy()
        row["animal_id"] = aid

        wf_df, summary = process_animal(row)
        wf_parts.append(wf_df)
        summary_rows.append(summary)
        print(f"  animal {aid:3d}  beats={summary['n_beats']:3d}  "
              f"HR={summary['hr_bpm']:.1f} bpm  "
              f"baseline={summary['baseline_mv']:.4f} mV  "
              f"tol={summary['tol_mv']:.4f} mV  "
              f"qt_current={summary['qt_current_ms']} ms")

    wf_all      = pd.concat(wf_parts, ignore_index=True)
    summary_df  = pd.DataFrame(summary_rows)

    wf_path      = OUTPUTS_DIR / "qt_template_export.csv"
    summary_path = OUTPUTS_DIR / "qt_template_summary.csv"

    wf_all.to_csv(wf_path,      index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nqt_template_export.csv   : {len(wf_all)} rows")
    print(f"qt_template_summary.csv  : {len(summary_df)} rows")
    print()
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_columns", 20, "display.width", 200):
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
