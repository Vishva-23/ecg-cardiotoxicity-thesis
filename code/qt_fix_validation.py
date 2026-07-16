"""Before/after validation of the QT-per-beat fix on Animal 125.

Reproduces the NB02 pipeline (parse -> baseline window -> bandpass+notch ->
MAD R-peak detection -> average_beats) and computes per-beat QT with the OLD
(unbounded, positive-argmax, fixed 0.01 mV) function and the NEW (RR-bounded,
polarity-aware, relative-threshold) function. Nothing is written; this only
prints a comparison against the manual LabChart value (~44.3 ms mean for 125).

Run:  python qt_fix_validation.py
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

FS = 1000
BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, BANDPASS_ORDER = 0.5, 150.0, 2
REFRACTORY_MS = 55
HR_ACCEPT_LOW, HR_ACCEPT_HIGH = 300, 700
PRE_R_MS, POST_R_MS, BEATS_TO_AVERAGE = 100, 150, 4
DATA_FILE = "../data/2024 11 14 125.txt"
MANUAL_QT_MS = 44.33  # Animal 125 manual mean from validation_table_filled.xlsx


# ---- pipeline helpers (identical to NB02 / backfill) ------------------------
def parse_header_and_layout(path):
    info, n_header, first = {}, 0, None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.lstrip()
            if not s or s[0].isdigit() or s[0] in "+-.":
                first = line.rstrip("\n"); break
            n_header += 1
            if "=" in line:
                k, _, rest = line.partition("=")
                info[k.strip()] = rest.strip("\n").strip("\t").split("\t")
    titles = [t.strip() for t in info.get("ChannelTitle", []) if t.strip()]
    n_channels = max(len(titles), 1)
    ncols = len(first.split("\t")) if first else n_channels + 1
    lead = max(1, ncols - n_channels)
    ch3 = next((i for i, t in enumerate(titles) if t.lower() == "channel 3"), 0)
    return n_header, lead + ch3


def load_voltage_and_markers(path, n_header, ecg_col):
    volts, markers = [], []
    import re
    mre = re.compile(r"#([*1-3])")
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
            volts.append(v)
            m = mre.search(stripped)
            if m:
                markers.append((len(volts) - 1, stripped[m.start():].strip().lower()))
    v = np.asarray(volts, float)
    if np.isnan(v).any():
        idx = np.arange(len(v)); good = ~np.isnan(v)
        v = np.interp(idx, idx[good], v[good])
    return v, markers


def baseline_window(markers):
    def is_base(t):
        return "baseline" in t.replace("basline", "baseline")
    starts = [i for i, t in markers if is_base(t) and not any(k in t for k in ("end", "done", "stop"))]
    ends = [i for i, t in markers if is_base(t) and any(k in t for k in ("end", "done", "stop"))]
    for s in starts:
        for e in ends:
            if 10 * FS <= e - s <= 900 * FS:
                return s, e
    raise RuntimeError("no baseline window found")


def bandpass(x):
    nyq = 0.5 * FS
    b, a = butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ / nyq, BANDPASS_HIGH_HZ / nyq], btype="band")
    y = filtfilt(b, a, x)
    for f0 in (50, 100, 150):
        bn, an = iirnotch(f0 / (FS / 2), Q=30)
        y = filtfilt(bn, an, y)
    return y


def _peaks(sig):
    med = float(np.median(sig)); mad = float(np.median(np.abs(sig - med)))
    if mad <= 0:
        return np.array([], int)
    height = med + 4 * mad
    prom = max(0.3 * mad, 0.005)
    dist = int(REFRACTORY_MS * FS / 1000)
    p, _ = find_peaks(sig, height=height, distance=dist, prominence=prom)
    return p


def _hr(p):
    return np.nan if len(p) < 2 else 60000.0 / float(np.mean(np.diff(p)))


def detect_r_peaks(sig):
    pp = _peaks(sig); ph = _hr(pp)
    if not np.isnan(ph) and HR_ACCEPT_LOW <= ph <= HR_ACCEPT_HIGH:
        return pp, False
    ip = _peaks(-sig); ih = _hr(ip)
    if not np.isnan(ih) and HR_ACCEPT_LOW <= ih <= HR_ACCEPT_HIGH:
        return ip, True
    return pp, False


def average_beats(sig, r_peaks):
    pre = int(PRE_R_MS * FS / 1000); post = int(POST_R_MS * FS / 1000)
    t_ms = (np.arange(pre + post) - pre) * (1000.0 / FS)
    beats = [sig[r - pre:r + post] for r in r_peaks if r - pre >= 0 and r + post <= len(sig)]
    return (np.vstack(beats) if beats else None), t_ms


# ---- OLD QT (as it was in NB02 before the fix) ------------------------------
def qt_old(beat, t_ms):
    n30 = int(30 * FS / 1000); n100 = int(100 * FS / 1000)
    n40 = int(40 * FS / 1000); n50 = int(50 * FS / 1000); n20 = int(20 * FS / 1000)
    r = int(np.argmin(np.abs(t_ms)))
    s = r + n30; e = min(len(beat), r + n100)
    if e <= s:
        return np.nan
    t_peak = s + int(np.argmax(beat[s:e]))
    after_t = beat[t_peak:min(len(beat), t_peak + n40)]
    lo = max(0, r - n50); hi = max(lo + 1, r - n20)
    base = float(np.mean(beat[lo:hi]))
    near = np.where(after_t <= base + 0.01)[0]
    if len(near) == 0:
        return np.nan
    qt = (t_peak + int(near[0]) - r) * 1000.0 / FS
    return float(qt) if 30 <= qt <= 100 else np.nan


# ---- NEW QT (identical to the fix now in NB02 _qt_per_beat) ------------------
def qt_new(beat, t_ms, rr_ms=None):
    QT_MAX_FRAC = 0.5      # physiological QT ceiling as a fraction of RR
    T_END_FRAC = 0.30      # excursion decay that marks the T-wave end
    n30 = int(30 * FS / 1000); n50 = int(50 * FS / 1000)
    n20 = int(20 * FS / 1000); n100 = int(100 * FS / 1000)
    r = int(np.argmin(np.abs(t_ms)))
    lo = max(0, r - n50); hi = max(lo + 1, r - n20)
    base = float(np.mean(beat[lo:hi]))
    s = r + n30
    if rr_ms is not None and np.isfinite(rr_ms) and rr_ms > 0:
        search_end = min(len(beat), r + int(QT_MAX_FRAC * rr_ms * FS / 1000))
    else:
        search_end = min(len(beat), r + n100)
    if search_end <= s:
        return np.nan
    exc = np.abs(beat[s:search_end] - base)
    if exc.size == 0:
        return np.nan
    k = int(np.argmax(exc)); t_peak = s + k; peak_exc = float(exc[k])
    if peak_exc <= 0:
        return np.nan
    tail = np.abs(beat[t_peak:search_end] - base)
    below = np.where(tail <= T_END_FRAC * peak_exc)[0]
    if len(below) == 0:
        return np.nan
    qt = (t_peak + int(below[0]) - r) * 1000.0 / FS
    return float(qt) if 15 <= qt <= 90 else np.nan


def summarize(vals):
    vals = np.array([v for v in vals if not np.isnan(v)])
    if vals.size == 0:
        return "no valid beats"
    return f"n={vals.size:3d}  mean={vals.mean():5.1f}  median={np.median(vals):5.1f}  sd={vals.std(ddof=1):4.1f} ms"


def main():
    n_header, ecg_col = parse_header_and_layout(DATA_FILE)
    v, markers = load_voltage_and_markers(DATA_FILE, n_header, ecg_col)
    s, e = baseline_window(markers)
    seg = v[s:e]
    filt = bandpass(seg)
    peaks, inverted = detect_r_peaks(filt)
    if inverted:
        filt = -filt
        peaks, _ = detect_r_peaks(filt)
    rr = np.diff(peaks) * (1000.0 / FS)
    rr_med = float(np.median(rr))
    beats, t_ms = average_beats(filt, peaks)

    print("=" * 66)
    print("Animal 125 - QT fix before/after")
    print("=" * 66)
    print(f"baseline window : samples {s}..{e}  ({(e - s) / FS:.1f} s)")
    print(f"R-peaks         : {len(peaks)} (inverted={inverted})")
    print(f"HR              : {60000.0 / rr.mean():.1f} bpm")
    print(f"RR mean/median  : {rr.mean():.1f} / {rr_med:.1f} ms")
    print(f"beats stacked   : {0 if beats is None else len(beats)}")
    print()
    old = [qt_old(b, t_ms) for b in beats]
    new = [qt_new(b, t_ms, rr_med) for b in beats]
    print(f"QT OLD (unbounded, +argmax, 0.01mV) : {summarize(old)}")
    print(f"QT NEW (RR-bounded, polarity-aware) : {summarize(new)}")
    print(f"QT MANUAL (LabChart M-cursor)       : {MANUAL_QT_MS:.1f} ms")
    print()
    om = np.nanmean(old); nm = np.nanmean(new)
    print(f"old vs manual : {(om - MANUAL_QT_MS) / MANUAL_QT_MS * 100:+.0f}%")
    print(f"new vs manual : {(nm - MANUAL_QT_MS) / MANUAL_QT_MS * 100:+.0f}%")


if __name__ == "__main__":
    main()
