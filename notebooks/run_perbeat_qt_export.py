"""Diagnostic export: per-beat QT distribution for selected animals.

Reuses NB02 functions verbatim. Does NOT modify NB01, NB02, or any pipeline
constant. Outputs perbeat_qt_distribution.csv to outputs/.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

# --- Named constants -------------------------------------------------------
ANIMALS     = [125, 157, 201, 249, 232]   # 115 excluded — known window-overlap outlier
PRE_BEAT_MS  = 100    # beat window before R — matches pipeline PRE_R_MS
POST_BEAT_MS = 150    # beat window after R  — matches pipeline POST_R_MS

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
HR_ACCEPT_LOW       = 300
HR_ACCEPT_HIGH      = 700

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


def _qt_per_beat(beat, t_ms):
    """Per-beat QT using Notebook 01's method: find the T-wave peak 30-100 ms
       after R, then the first return to (pre-R baseline + 0.01 mV). Returns ms
       (accepted only if 30-100 ms), else nan. R peak = sample where t_ms == 0."""
    n30  = int(30  * FS / 1000); n100 = int(100 * FS / 1000)
    n40  = int(40  * FS / 1000); n50  = int(50  * FS / 1000); n20 = int(20 * FS / 1000)
    r_idx = int(np.argmin(np.abs(t_ms)))
    s = r_idx + n30
    e = min(len(beat), r_idx + n100)
    if e <= s:
        return np.nan
    t_peak = s + int(np.argmax(beat[s:e]))
    after_t = beat[t_peak: min(len(beat), t_peak + n40)]
    lo = max(0, r_idx - n50); hi = max(lo + 1, r_idx - n20)
    baseline = float(np.mean(beat[lo:hi]))
    near = np.where(after_t <= baseline + 0.01)[0]
    if len(near) == 0:
        return np.nan
    qt = (t_peak + int(near[0]) - r_idx) * 1000.0 / FS
    return float(qt) if 30 <= qt <= 100 else np.nan


# ---------------------------------------------------------------------------
# Diagnostic processing
# ---------------------------------------------------------------------------

def process_animal(row):
    """Extract per-beat QT for one animal using its features_fallback window."""
    path = DATA_DIR / str(row["source_file"])

    n_header, _lead, ecg_col, _titles = parse_header_and_layout(path)
    voltage, _markers = load_ecg_file(path, n_header, ecg_col)

    ws = int(row["window_start"])
    we = int(row["window_end"])
    segment = voltage[ws:we]

    filtered = bandpass_filter(segment)

    peaks, inverted, _hr, _ = detect_r_peaks(filtered)
    if inverted:
        filtered = -filtered

    pre  = int(PRE_BEAT_MS  * FS / 1000)
    post = int(POST_BEAT_MS * FS / 1000)
    t_ms = (np.arange(pre + post) - pre) * (1000.0 / FS)

    rr_intervals = np.diff(peaks) * (1000.0 / FS)   # length = n_peaks - 1

    rows = []
    aid = int(row["animal_id"])
    for i, r in enumerate(peaks):
        if r - pre < 0 or r + post > len(filtered):
            continue
        beat = filtered[r - pre: r + post]
        qt   = _qt_per_beat(beat, t_ms)
        rr   = float(rr_intervals[i]) if i < len(rr_intervals) else np.nan
        rows.append({
            "animal_id":      aid,
            "beat_index":     i,
            "qt_ms":          qt,
            "rr_ms":          round(rr, 3),
            "r_amplitude_mv": round(float(np.max(beat)), 4),
        })
    return rows


def _stats(series):
    """Return (n_valid, median, q25, q75, mn, mx) over non-NaN values."""
    v = series.dropna()
    if len(v) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan
    q25, q75 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
    return len(v), float(np.median(v)), q25, q75, float(v.min()), float(v.max())


def main():
    fb = pd.read_csv(OUTPUTS_DIR / "features_fallback.csv")
    fb = fb[fb["animal_id"].isin(ANIMALS)].set_index("animal_id")

    all_rows = []
    print(f"{'animal':>8}  {'beats':>6}  {'valid_qt':>8}  "
          f"{'median':>8}  {'IQR':>14}  {'min':>8}  {'max':>8}")
    print("-" * 74)

    for aid in ANIMALS:
        row = fb.loc[aid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        row = row.copy()
        row["animal_id"] = aid

        beat_rows = process_animal(row)
        all_rows.extend(beat_rows)

        df_a = pd.DataFrame(beat_rows)
        n_valid, med, q25, q75, mn, mx = _stats(df_a["qt_ms"])
        print(f"{aid:>8}  {len(beat_rows):>6}  {n_valid:>8}  "
              f"{med:>8.1f}  [{q25:6.1f}, {q75:6.1f}]  {mn:>8.1f}  {mx:>8.1f}")

    out_df = pd.DataFrame(all_rows)
    out_path = OUTPUTS_DIR / "perbeat_qt_distribution.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\nperbeat_qt_distribution.csv : {len(out_df)} rows total")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
