"""Diagnostic: tangent-method QT estimation for selected animals.

Standalone validation script — does NOT modify NB01, NB02, or any pipeline
constant. Reuses NB02 helper functions verbatim for loading, filtering,
R-peak detection, beat extraction, and baseline estimation.
Outputs qt_tangent_results.csv and qt_tangent_construction.png to outputs/.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

# ---------------------------------------------------------------------------
# Named constants — tangent algorithm (each justified by a comment)
# ---------------------------------------------------------------------------
T_SEARCH_START_MS  = 15     # start T-peak search after QRS/J-wave complex
T_SEARCH_END_FRAC  = 0.85   # end search at 0.85×RR to stay clear of next beat
DERIV_SMOOTH_MS    = 3      # boxcar smoothing for derivative (mouse signal is small)
T_MIN_PROMINENCE   = 3.0    # T-peak must exceed baseline by ≥3× pre-R noise std

# ---------------------------------------------------------------------------
# Animal list — 115 is processed but flagged OUTLIER (known window-overlap issue)
# ---------------------------------------------------------------------------
ANIMALS         = [125, 157, 201, 249, 232, 115]
OUTLIER_ANIMALS = {115}

# ---------------------------------------------------------------------------
# Reference values from previous exports (for the comparison columns)
# ---------------------------------------------------------------------------
_QT_OLD_TEMPLATE  = {115: 85.0, 125: 41.0, 157: 47.0, 201: 44.0, 232: 40.0, 249: 40.0}
_QT_OLD_PERBEAT   = {115: 65.91, 125: 91.21, 157: 68.07, 201: 70.36, 232: 96.35, 249: 87.68}

# ---------------------------------------------------------------------------
# Project root (dynamic)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR    = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Pipeline constants — verbatim from NB02 (do not edit)
# ---------------------------------------------------------------------------
FS               = 1000
BANDPASS_LOW_HZ  = 0.5
BANDPASS_HIGH_HZ = 150.0
BANDPASS_ORDER   = 2
REFRACTORY_MS    = 55
HR_ACCEPT_LOW    = 300
HR_ACCEPT_HIGH   = 700
BEATS_TO_AVERAGE = 4
PRE_R_MS         = 100
POST_R_MS        = 150

_MARKER_RE = re.compile(r"#([*1-3])")

# ---------------------------------------------------------------------------
# NB02 helper functions — verbatim copies
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
        idx  = np.arange(len(voltage))
        good = ~np.isnan(voltage)
        if good.any():
            voltage = np.interp(idx, idx[good], voltage[good])
    return voltage, markers


def bandpass_filter(x, fs=FS, low=BANDPASS_LOW_HZ, high=BANDPASS_HIGH_HZ, order=BANDPASS_ORDER):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    y = filtfilt(b, a, x)
    for f0 in (50, 100, 150):          # notch out Irish mains harmonics
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
        return 60000.0 / np.mean(np.diff(peaks) * (1000.0 / fs))

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
        return max(0.0, HR_ACCEPT_LOW - hr) + max(0.0, hr - HR_ACCEPT_HIGH)

    if _dist(inv_hr) < _dist(pos_hr):
        return inv_peaks, True, inv_hr, inv_h
    return pos_peaks, False, pos_hr, pos_h


def average_beats(sig, r_peaks, fs=FS,
                  pre_ms=PRE_R_MS, post_ms=POST_R_MS,
                  group_size=BEATS_TO_AVERAGE):
    pre     = int(pre_ms  * fs / 1000)
    post    = int(post_ms * fs / 1000)
    win_len = pre + post
    t_ms    = (np.arange(win_len) - pre) * (1000.0 / fs)
    beats   = [sig[r - pre: r + post] for r in r_peaks
               if r - pre >= 0 and r + post <= len(sig)]
    if not beats:
        return None, t_ms, None
    beats = np.vstack(beats)
    n_full_groups = len(beats) // group_size
    if n_full_groups == 0:
        return beats.mean(axis=0), t_ms, beats
    grouped = beats[:n_full_groups * group_size].reshape(n_full_groups, group_size, win_len)
    return grouped.mean(axis=1).mean(axis=0), t_ms, beats


# ---------------------------------------------------------------------------
# Tangent method
# ---------------------------------------------------------------------------

def measure_qt_tangent(template, t_ms, baseline, rr_ms):
    """Tangent method T-end detection. Returns a dict with qt_tangent_ms and diagnostics."""
    # pre-R noise floor (same region used for baseline)
    noise_mask = t_ms < -70
    noise_std  = float(np.std(template[noise_mask])) if noise_mask.any() else np.nan

    _nan = dict(qt_tangent_ms=np.nan, t_peak_ms=np.nan, t_peak_amp_mv=np.nan,
                t_steep_ms=np.nan, noise_std_mv=noise_std)

    # Step 2 — T-search window
    t_end_search = T_SEARCH_END_FRAC * rr_ms
    search_idx   = np.where((t_ms >= T_SEARCH_START_MS) & (t_ms <= t_end_search))[0]
    if len(search_idx) < 3:
        return {**_nan, "reason": "search_window_empty"}

    # Step 3 — T-peak: maximum absolute deviation from baseline inside search window
    seg        = template[search_idx]
    dev        = seg - baseline
    peak_local = int(np.argmax(np.abs(dev)))
    peak_abs   = int(search_idx[peak_local])
    t_peak_ms  = float(t_ms[peak_abs])
    t_peak_amp = float(template[peak_abs])
    t_polarity = int(np.sign(dev[peak_local]))   # +1 positive T, -1 negative T

    # Step 4 — Prominence guard
    if np.isnan(noise_std) or noise_std < 1e-9:
        return {**_nan, "t_peak_ms": t_peak_ms, "t_peak_amp_mv": t_peak_amp,
                "reason": "no_noise_estimate"}
    if abs(t_peak_amp - baseline) < T_MIN_PROMINENCE * noise_std:
        return {**_nan, "t_peak_ms": t_peak_ms, "t_peak_amp_mv": t_peak_amp,
                "reason": "no_measurable_T"}

    # Step 5 — Downslope: T-peak to end of search window
    down_end = int(np.searchsorted(t_ms, t_end_search, side="right"))
    down_idx = np.arange(peak_abs, min(down_end, len(template)))
    if len(down_idx) < 3:
        return {**_nan, "t_peak_ms": t_peak_ms, "t_peak_amp_mv": t_peak_amp,
                "reason": "downslope_too_short"}

    # Smooth derivative over DERIV_SMOOTH_MS samples (mV/ms at 1 kHz)
    smooth_n   = max(1, int(DERIV_SMOOTH_MS * FS / 1000))
    deriv_full = np.gradient(template)
    if smooth_n > 1:
        deriv_full = np.convolve(deriv_full, np.ones(smooth_n) / smooth_n, mode="same")

    # Step 6 — Steepest-return = max |derivative| on the downslope
    down_derivs = deriv_full[down_idx]
    steep_local = int(np.argmax(np.abs(down_derivs)))
    steep_abs   = int(down_idx[steep_local])
    t_steep_ms  = float(t_ms[steep_abs])
    v_steep     = float(template[steep_abs])
    slope       = float(deriv_full[steep_abs])   # signed mV/ms

    if abs(slope) < 1e-9:
        return {**_nan, "t_peak_ms": t_peak_ms, "t_peak_amp_mv": t_peak_amp,
                "t_steep_ms": t_steep_ms, "noise_std_mv": noise_std, "reason": "zero_slope"}

    # Step 7 — Tangent intersection with baseline
    t_end = t_steep_ms + (baseline - v_steep) / slope

    # Step 8 — Sanity: T-end must be after the T-peak and within 1.1×RR
    if t_end <= t_peak_ms or t_end > rr_ms * 1.1:
        return {**_nan, "t_peak_ms": t_peak_ms, "t_peak_amp_mv": t_peak_amp,
                "t_steep_ms": t_steep_ms, "noise_std_mv": noise_std,
                "reason": f"t_end_out_of_range({t_end:.1f}ms)"}

    return {
        "qt_tangent_ms": float(t_end),
        "t_peak_ms":     t_peak_ms,
        "t_peak_amp_mv": t_peak_amp,
        "t_steep_ms":    t_steep_ms,
        "noise_std_mv":  noise_std,
        "reason":        "ok",
        "_slope":        slope,        # kept for figure only, not in CSV
        "_v_steep":      v_steep,
    }


def mitchell_qtc(qt_ms, rr_ms):
    """Mitchell QTc = QT / sqrt(RR/100), QT and RR in ms."""
    if np.isnan(qt_ms) or np.isnan(rr_ms) or rr_ms <= 0:
        return np.nan
    return float(qt_ms / np.sqrt(rr_ms / 100.0))


# ---------------------------------------------------------------------------
# Per-animal pipeline
# ---------------------------------------------------------------------------

def process_animal(fb_row):
    """Load, filter, detect peaks, build template, estimate baseline. Returns dict."""
    path = DATA_DIR / str(fb_row["source_file"])

    n_header, _lead, ecg_col, _titles = parse_header_and_layout(path)
    voltage, _markers = load_ecg_file(path, n_header, ecg_col)

    ws      = int(fb_row["window_start"])
    we      = int(fb_row["window_end"])
    segment = voltage[ws:we]

    filtered = bandpass_filter(segment)

    peaks, inverted, hr, _ = detect_r_peaks(filtered)
    if inverted:
        filtered = -filtered

    rr_intervals = np.diff(peaks) * (1000.0 / FS)
    rr_mean = float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else np.nan

    template, t_ms, _beats = average_beats(filtered, peaks)

    # baseline from pre-R median — same formula as _measure_one in NB02
    noise_mask = t_ms < -70
    baseline   = float(np.median(template[noise_mask])) if noise_mask.any() else 0.0

    return dict(
        n_beats  = int(len(peaks)),
        hr_bpm   = float(hr),
        rr_ms    = round(rr_mean, 2),
        template = template,
        t_ms     = t_ms,
        baseline = baseline,
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _tangent_line_y(t_arr, t_steep, v_steep, slope):
    """Evaluate the tangent line at each time in t_arr."""
    return v_steep + slope * (t_arr - t_steep)


def make_figure(fig_records, out_path):
    """One panel per animal showing template + tangent construction."""
    n    = len(fig_records)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4 * nrow), squeeze=False)
    axes_flat = axes.flatten()

    for ax, rec in zip(axes_flat, fig_records):
        template  = rec["template"]
        t_ms      = rec["t_ms"]
        baseline  = rec["baseline"]
        res       = rec["tangent_result"]
        aid       = rec["animal_id"]
        rr_ms     = rec["rr_ms"]
        outlier   = rec["outlier"]

        qt    = res["qt_tangent_ms"]
        t_pk  = res["t_peak_ms"]
        t_pk_amp = res["t_peak_amp_mv"]
        t_st  = res["t_steep_ms"]
        slope = res.get("_slope", np.nan)
        v_st  = res.get("_v_steep", np.nan)
        reason = res["reason"]

        # Template waveform
        ax.plot(t_ms, template, color="steelblue", linewidth=1.2, label="template")

        # Baseline
        ax.axhline(baseline, color="gray", linewidth=0.8, linestyle="--", label=f"baseline {baseline:.3f} mV")

        if reason == "ok" and not np.isnan(qt):
            # T-peak marker
            ax.plot(t_pk, t_pk_amp, "rv", markersize=8, label=f"T-peak {t_pk:.0f} ms")

            # Steepest-return marker
            v_steep_plot = float(template[np.argmin(np.abs(t_ms - t_st))])
            ax.plot(t_st, v_steep_plot, "bs", markersize=7, label=f"steepest {t_st:.0f} ms")

            # Tangent line: draw from 15 ms before steep point to 10 ms after T-end
            t_tang_start = max(t_pk, t_st - 15.0)
            t_tang_end   = min(qt + 10.0, float(t_ms[-1]))
            t_line = np.linspace(t_tang_start, t_tang_end, 200)
            v_line = _tangent_line_y(t_line, t_st, v_st, slope)
            ax.plot(t_line, v_line, color="darkorange", linewidth=1.5,
                    linestyle="-", label="tangent")

            # T-end vertical
            ax.axvline(qt, color="green", linewidth=1.2, linestyle="--",
                       label=f"T-end {qt:.1f} ms")

            qt_over_rr = qt / rr_ms if rr_ms > 0 else np.nan
            title_str = (f"Animal {aid}  |  QT={qt:.1f} ms  |  "
                         f"QT/RR={qt_over_rr:.2f}")
        else:
            title_str = f"Animal {aid}  |  QT=NaN  ({reason})"

        if outlier:
            title_str = "[OUTLIER]  " + title_str
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)

        ax.set_title(title_str, fontsize=8.5, pad=4)
        ax.set_xlabel("Time from R-peak (ms)", fontsize=8)
        ax.set_ylabel("Voltage (mV)", fontsize=8)
        ax.axvline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)

    # hide unused panels
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Tangent-method QT construction — averaged-beat template per animal",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fb = pd.read_csv(OUTPUTS_DIR / "features_fallback.csv").set_index("animal_id")

    csv_rows   = []
    fig_records = []

    for aid in ANIMALS:
        fb_row      = fb.loc[aid]
        if isinstance(fb_row, pd.DataFrame):
            fb_row = fb_row.iloc[0]
        fb_row      = fb_row.copy()
        fb_row["animal_id"] = aid

        proc = process_animal(fb_row)
        res  = measure_qt_tangent(
            proc["template"], proc["t_ms"], proc["baseline"], proc["rr_ms"]
        )

        qt     = res["qt_tangent_ms"]
        rr     = proc["rr_ms"]
        outlier = aid in OUTLIER_ANIMALS

        row = {
            "animal_id":         aid,
            "n_beats":           proc["n_beats"],
            "rr_ms":             rr,
            "hr_bpm":            round(proc["hr_bpm"], 1),
            "baseline_mv":       round(proc["baseline"], 4),
            "noise_std_mv":      round(res["noise_std_mv"], 5) if not np.isnan(res["noise_std_mv"]) else np.nan,
            "t_peak_ms":         res["t_peak_ms"],
            "t_peak_amp_mv":     round(res["t_peak_amp_mv"], 4) if not np.isnan(res.get("t_peak_amp_mv", np.nan)) else np.nan,
            "t_steep_ms":        res["t_steep_ms"],
            "qt_tangent_ms":     round(qt, 2) if not np.isnan(qt) else np.nan,
            "qtc_tangent_ms":    round(mitchell_qtc(qt, rr), 2) if not np.isnan(qt) else np.nan,
            "qt_over_rr":        round(qt / rr, 3) if (not np.isnan(qt) and rr > 0) else np.nan,
            "qt_oldtemplate_ms": _QT_OLD_TEMPLATE.get(aid),
            "qt_oldperbeat_ms":  _QT_OLD_PERBEAT.get(aid),
            "reason":            ("OUTLIER/" + res["reason"]) if outlier else res["reason"],
            "outlier":           outlier,
        }
        csv_rows.append(row)

        fig_records.append({
            "animal_id":     aid,
            "template":      proc["template"],
            "t_ms":          proc["t_ms"],
            "baseline":      proc["baseline"],
            "rr_ms":         rr,
            "tangent_result": res,
            "outlier":       outlier,
        })

        tag = " [OUTLIER]" if outlier else ""
        t_pk_amp = res.get("t_peak_amp_mv", np.nan)
        n_std    = res.get("noise_std_mv", np.nan)
        print(f"  animal {aid:3d}{tag}  n={proc['n_beats']:3d}  "
              f"HR={proc['hr_bpm']:.1f}  "
              f"T-peak={res['t_peak_ms']} ms  amp={t_pk_amp:.4f}  "
              f"noise_std={n_std:.4f}  "
              f"qt_tangent={round(qt,2) if not np.isnan(qt) else 'NaN'}  "
              f"reason={res['reason']}")

    # --- CSV ---
    out_df = pd.DataFrame(csv_rows)
    csv_path = OUTPUTS_DIR / "qt_tangent_results.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"\nqt_tangent_results.csv : {len(out_df)} rows  -> {csv_path}")

    # --- Figure ---
    fig_path = OUTPUTS_DIR / "figures" / "qt_tangent_construction.png"
    make_figure(fig_records, fig_path)

    # --- Print table ---
    display_cols = [
        "animal_id", "n_beats", "rr_ms", "baseline_mv", "noise_std_mv",
        "t_peak_ms", "t_steep_ms", "qt_tangent_ms", "qtc_tangent_ms",
        "qt_over_rr", "qt_oldtemplate_ms", "qt_oldperbeat_ms", "reason",
    ]
    print()
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.max_columns", 20, "display.width", 220):
        print(out_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
