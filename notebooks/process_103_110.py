"""
Task 1: Process animals 103 and 110 through the full NB02 pipeline.
Reports HR and beat count first, then extracts features.
Does NOT modify any CSV yet — caller decides after reviewing results.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

# ── project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
while not (ROOT / "data").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
OUT      = ROOT / "outputs"

# ── pipeline constants (do not change) ──────────────────────────────────────
FS               = 1000
BANDPASS_LOW_HZ  = 0.5
BANDPASS_HIGH_HZ = 150.0
BANDPASS_ORDER   = 2
REFRACTORY_MS    = 55
MIN_RR_MS        = 75
BEATS_TO_AVERAGE = 4
PRE_R_MS         = 100
POST_R_MS        = 150
HR_ACCEPT_LOW    = 300
HR_ACCEPT_HIGH   = 700
HR_FLAG_LOW      = 400
HR_FLAG_HIGH     = 700
GAP_MIN_S        = 10
GAP_MAX_S        = 120
GAP_MAX_LONG_S   = 900
LONG_USE_S       = 60
SINGLE_FALLBACK_S = 30
EXCLUDE_KEYWORDS = ("oc 1", "oc 2", "oc1", "oc2", "jugular", "jugcan", "injection", "inject")
END_KEYWORDS     = ("end", "done", "stop")
START_HINTS      = ("baseline", "ecg", "start")
QTC_FORMULA_NAME = "Mitchell"
_MARKER_RE       = re.compile(r"#([*1-3])")

TARGET_ANIMALS   = {103, 110}

# ── helper functions (verbatim from NB02) ────────────────────────────────────

def parse_filename(path):
    nums = re.findall(r"\d+", path.stem)
    try:
        if len(nums) >= 4:
            y, m, d, a = int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
            if y > 3000: y = y % 10000  # fix typo e.g. 22024 -> 2024
            return a, pd.Timestamp(year=y, month=m, day=d)
        if len(nums) == 3:
            y, m, a = int(nums[0]), int(nums[1]), int(nums[2])
            if y > 3000: y = y % 10000
            return a, pd.Timestamp(year=y, month=m, day=1)
    except (ValueError, TypeError):
        pass
    return None, None


def parse_header_and_layout(path):
    info = {}; n_header = 0; first_data_row = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.lstrip()
            if not s or s[0].isdigit() or s[0] in "+-.":
                first_data_row = line.rstrip("\n"); break
            n_header += 1
            if "=" in line:
                key, _, rest = line.partition("=")
                info[key.strip()] = rest.strip("\n").strip("\t").split("\t")
    titles = [t.strip() for t in info.get("ChannelTitle", []) if t.strip()]
    n_channels = max(len(titles), 1)
    n_cols = len(first_data_row.split("\t")) if first_data_row else (n_channels + 1)
    lead_cols = max(1, n_cols - n_channels)
    ch3_idx = next((i for i, t in enumerate(titles) if t.lower() == "channel 3"), 0)
    return n_header, lead_cols, lead_cols + ch3_idx, titles


def load_ecg_file(path, n_header, ecg_col):
    voltages = []; markers = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(n_header): f.readline()
        for line in f:
            stripped = line.rstrip("\n"); parts = stripped.split("\t")
            if len(parts) <= ecg_col: continue
            try: v = float(parts[ecg_col])
            except ValueError: v = np.nan
            voltages.append(v)
            m = _MARKER_RE.search(stripped)
            if m:
                markers.append((len(voltages)-1, stripped[m.start():].strip(), m.group(1)))
    voltage = np.asarray(voltages, dtype=float)
    if np.isnan(voltage).any():
        idx = np.arange(len(voltage)); good = ~np.isnan(voltage)
        if good.any(): voltage = np.interp(idx, idx[good], voltage[good])
    return voltage, markers


def _looks_like_end(t):      return any(k in t.lower() for k in END_KEYWORDS)
def _looks_like_excluded(t): return any(k in t.lower() for k in EXCLUDE_KEYWORDS)
def _looks_like_baseline(t): return "baseline" in t.lower().replace("basline", "baseline")


def find_baseline_window(markers, n_samples, fs=FS):
    MIN_GAP = GAP_MIN_S * fs; MAX_GAP = GAP_MAX_S * fs
    LONG_GAP = GAP_MAX_LONG_S * fs; LONG_USE = LONG_USE_S * fs
    FALLBACK = SINGLE_FALLBACK_S * fs
    tagged = []
    for idx, text, prefix in markers:
        if _looks_like_excluded(text): continue
        tagged.append({"idx": idx, "text": text, "prefix": prefix, "end": _looks_like_end(text)})
    starts = [m for m in tagged if not m["end"]]
    ends   = [m for m in tagged if m["end"]]

    def _try(ss, es, lo, hi, label):
        for s in ss:
            for e in es:
                if e["idx"] <= s["idx"]: continue
                gap = e["idx"] - s["idx"]
                if lo <= gap <= hi: return s, e, label
        return None

    for pre in ("*", "1", "2", "3"):
        r = _try([m for m in starts if m["prefix"] == pre],
                 [m for m in ends   if m["prefix"] == pre], MIN_GAP, MAX_GAP, f"A_same_{pre}")
        if r: s,e,l = r; return s["idx"], min(e["idx"], n_samples), l, s["text"], e["text"]

    r = _try(starts, ends, MIN_GAP, MAX_GAP, "B_cross")
    if r: s,e,l = r; return s["idx"], min(e["idx"], n_samples), l, s["text"], e["text"]

    bl_s = [m for m in starts if _looks_like_baseline(m["text"])]
    bl_e = [m for m in ends   if _looks_like_baseline(m["text"])]
    r = _try(bl_s, bl_e, MIN_GAP, LONG_GAP, "C_long_baseline")
    if r:
        s,e,l = r
        return s["idx"], min(s["idx"] + LONG_USE, e["idx"], n_samples), l, s["text"], e["text"]

    candidate = next((m for m in starts if any(k in m["text"].lower() for k in START_HINTS)), None)
    if candidate is None and starts: candidate = starts[0]
    if candidate is None and tagged: candidate = tagged[0]
    if candidate is not None:
        return candidate["idx"], min(candidate["idx"] + FALLBACK, n_samples), "D_single_30s", candidate["text"], None

    return None, None, "no_markers", None, None


def bandpass_filter(x, fs=FS):
    nyq = 0.5 * fs
    b, a = butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ/nyq, BANDPASS_HIGH_HZ/nyq], btype="band")
    y = filtfilt(b, a, x)
    for f0 in (50, 100, 150):
        bn, an = iirnotch(f0 / (fs/2), Q=30)
        y = filtfilt(bn, an, y)
    return y


def _peaks_with_mad(sig, fs=FS):
    if sig.size == 0: return np.array([], dtype=int), np.nan, np.nan
    med = float(np.median(sig)); mad = float(np.median(np.abs(sig - med)))
    if mad <= 0: return np.array([], dtype=int), med, mad
    height = med + 4 * mad; prominence = max(0.3 * mad, 0.005)
    distance = int(REFRACTORY_MS * fs / 1000)
    peaks, _ = find_peaks(sig, height=height, distance=distance, prominence=prominence)
    return peaks, height, prominence


def detect_r_peaks(sig, fs=FS):
    def _hr(p): return 60000.0 / np.mean(np.diff(p) * 1000.0/fs) if len(p) >= 2 else np.nan
    pos_peaks, pos_h, _ = _peaks_with_mad(sig, fs)
    pos_hr = _hr(pos_peaks)
    if not np.isnan(pos_hr) and HR_ACCEPT_LOW <= pos_hr <= HR_ACCEPT_HIGH:
        return pos_peaks, False, pos_hr, pos_h
    inv_peaks, inv_h, _ = _peaks_with_mad(-sig, fs)
    inv_hr = _hr(inv_peaks)
    if not np.isnan(inv_hr) and HR_ACCEPT_LOW <= inv_hr <= HR_ACCEPT_HIGH:
        return inv_peaks, True, inv_hr, inv_h
    def _dist(hr):
        if np.isnan(hr): return np.inf
        return max(0, HR_ACCEPT_LOW - hr, hr - HR_ACCEPT_HIGH)
    if _dist(inv_hr) < _dist(pos_hr): return inv_peaks, True, inv_hr, inv_h
    return pos_peaks, False, pos_hr, pos_h


def average_beats(sig, r_peaks, fs=FS):
    pre = int(PRE_R_MS * fs / 1000); post = int(POST_R_MS * fs / 1000)
    win_len = pre + post; t_ms = (np.arange(win_len) - pre) * (1000.0 / fs)
    beats = [sig[r-pre:r+post] for r in r_peaks if r-pre >= 0 and r+post <= len(sig)]
    if not beats: return None, t_ms, None
    beats = np.vstack(beats)
    n_full = len(beats) // BEATS_TO_AVERAGE
    if n_full == 0: return beats.mean(axis=0), t_ms, beats
    grouped = beats[:n_full*BEATS_TO_AVERAGE].reshape(n_full, BEATS_TO_AVERAGE, win_len)
    return grouped.mean(axis=1).mean(axis=0), t_ms, beats


def _measure_one(sig, t_ms):
    r_idx = int(np.argmax(sig)); r_amp = float(sig[r_idx]); half = r_amp / 2.0
    left = r_idx
    while left > 0 and sig[left] > half: left -= 1
    right = r_idx
    while right < len(sig)-1 and sig[right] > half: right += 1
    qrs_ms = float(t_ms[right] - t_ms[left])
    def window_max(s_ms, e_ms):
        mask = (t_ms >= s_ms) & (t_ms <= e_ms)
        if not mask.any(): return np.nan, np.nan
        seg = sig[mask]; ts = t_ms[mask]; k = int(np.argmax(seg))
        return float(seg[k]), float(ts[k])
    j_amp, _      = window_max(10, 30)
    t_amp, t_time = window_max(40, 80)
    rt_ms = float(t_time) if not np.isnan(t_time) else np.nan
    baseline = float(np.median(sig[t_ms < -70])) if (t_ms < -70).any() else 0.0
    tol = 0.05 * r_amp; qt_ms = np.nan
    for i in np.where(t_ms >= 40)[0]:
        if abs(sig[i] - baseline) <= tol: qt_ms = float(t_ms[i]); break
    return r_amp, qrs_ms, j_amp, t_amp, rt_ms, qt_ms


def _qt_per_beat(beat, t_ms):
    n30 = int(30*FS/1000); n100 = int(100*FS/1000); n40 = int(40*FS/1000)
    n50 = int(50*FS/1000); n20 = int(20*FS/1000)
    r_idx = int(np.argmin(np.abs(t_ms)))
    s = r_idx + n30; e = min(len(beat), r_idx + n100)
    if e <= s: return np.nan
    t_peak = s + int(np.argmax(beat[s:e]))
    after_t = beat[t_peak: min(len(beat), t_peak + n40)]
    lo = max(0, r_idx - n50); hi = max(lo+1, r_idx - n20)
    baseline = float(np.mean(beat[lo:hi]))
    near = np.where(after_t <= baseline + 0.01)[0]
    if len(near) == 0: return np.nan
    qt = (t_peak + int(near[0]) - r_idx) * 1000.0 / FS
    return float(qt) if 30 <= qt <= 100 else np.nan


def extract_morphology_features(template, t_ms, beats=None):
    keys = ["r_amplitude_mv", "qrs_duration_ms", "j_wave_amplitude_mv",
            "t_wave_amplitude_mv", "rt_interval_ms", "qt_ms",
            "qt_std_ms", "qrs_std_ms", "r_amplitude_std_mv"]
    if template is None: return dict.fromkeys(keys, np.nan)
    r_amp, qrs_ms, j_amp, t_amp, rt_ms, qt_template = _measure_one(template, t_ms)
    qt_ms = qt_template; qt_std_ms = qrs_std_ms = r_amplitude_std_mv = np.nan
    if beats is not None and getattr(beats, "ndim", 0) == 2 and len(beats) > 1:
        r_amplitudes, qrs_durations, qt_estimates = [], [], []
        for b in beats:
            br, bqrs, _, _, _, _ = _measure_one(b, t_ms)
            r_amplitudes.append(br); qrs_durations.append(bqrs)
            bqt = _qt_per_beat(b, t_ms)
            if not np.isnan(bqt): qt_estimates.append(bqt)
        if qt_estimates: qt_ms = float(np.mean(qt_estimates))
        qt_std_ms          = float(np.std(qt_estimates, ddof=1))  if len(qt_estimates)  > 1 else np.nan
        qrs_std_ms         = float(np.std(qrs_durations, ddof=1)) if len(qrs_durations) > 1 else np.nan
        r_amplitude_std_mv = float(np.std(r_amplitudes, ddof=1))  if len(r_amplitudes)  > 1 else np.nan
    return {"r_amplitude_mv": r_amp, "qrs_duration_ms": qrs_ms,
            "j_wave_amplitude_mv": j_amp, "t_wave_amplitude_mv": t_amp,
            "rt_interval_ms": rt_ms, "qt_ms": qt_ms,
            "qt_std_ms": qt_std_ms, "qrs_std_ms": qrs_std_ms,
            "r_amplitude_std_mv": r_amplitude_std_mv}


def mitchell_qtc(qt_ms, rr_ms):
    if np.isnan(qt_ms) or np.isnan(rr_ms) or rr_ms <= 0: return np.nan
    return float(qt_ms / np.sqrt(rr_ms / 100.0))


# ── main: scan for target animals ───────────────────────────────────────────
print("=" * 64)
print("Processing animals 103 and 110")
print("=" * 64)

results = {}
files = sorted(DATA_DIR.glob("*.txt"))

for path in files:
    animal_id, rec_date = parse_filename(path)
    if animal_id not in TARGET_ANIMALS:
        continue

    print(f"\n{'-'*50}")
    print(f"Animal {animal_id}  |  file: {path.name}")
    print(f"{'-'*50}")

    try:
        n_header, lead_cols, ecg_col, titles = parse_header_and_layout(path)
        voltage, markers = load_ecg_file(path, n_header, ecg_col)
        print(f"  ECG column  : {ecg_col}  |  signal length: {len(voltage)} samples ({len(voltage)/FS:.1f} s)")
        print(f"  Markers     : {len(markers)}")
        for mk in markers:
            print(f"    @ {mk[0]/FS:.1f}s  [{mk[2]}]  {mk[1][:80]}")

        med_abs = float(np.median(np.abs(voltage)))
        print(f"  Median |V|  : {med_abs:.3f}  (expected < 50 mV)")
        if med_abs > 50:
            print("  ERROR: voltage looks like BPM column — wrong column selected")
            results[animal_id] = {"status": "FAILED_COLUMN"}
            continue

        start, end, rule, s_text, e_text = find_baseline_window(markers, len(voltage))

        # Fallback for no markers: use first 60 s of recording
        if start is None:
            fallback_len = min(60 * FS, len(voltage))
            if fallback_len >= 2 * FS:
                start, end = 0, fallback_len
                rule = "E_no_marker_first60s"
                s_text, e_text = "NO MARKERS — first 60s used", None
                print(f"  WARNING: no markers found — using first 60s of recording as fallback")
            else:
                print("  ERROR: no baseline window found and recording too short")
                results[animal_id] = {"status": "FAILED_ANNOTATION"}
                continue

        print(f"  Window rule : {rule}")
        print(f"  Window      : {start} - {end}  ({(end-start)/FS:.1f} s)")
        print(f"  Start ann   : {s_text}")
        print(f"  End ann     : {e_text}")

        segment = voltage[start:end]
        if segment.size < 2 * FS:
            print(f"  ERROR: window too short ({segment.size} samples)")
            results[animal_id] = {"status": "FAILED_ANNOTATION"}
            continue

        filtered = bandpass_filter(segment)
        peaks, inverted, hr, _ = detect_r_peaks(filtered)
        if inverted:
            filtered = -filtered
            print(f"  Signal inverted: YES — flipped for processing")

        rr_ms = np.diff(peaks) * (1000.0 / FS)
        rr_mean = float(np.mean(rr_ms)) if len(rr_ms) > 0 else np.nan

        print(f"\n  *** HR         : {hr:.1f} bpm")
        print(f"  *** Beat count : {len(peaks)}")
        print(f"  *** RR mean    : {rr_mean:.1f} ms")
        hr_flag = "" if HR_FLAG_LOW <= hr <= HR_FLAG_HIGH else "  <<< OUTSIDE 400-700 bpm FLAG"
        print(f"  *** HR in 400-700 range: {'YES' if HR_FLAG_LOW <= hr <= HR_FLAG_HIGH else 'NO' + hr_flag}")

        if len(peaks) < 10:
            print("  ERROR: fewer than 10 beats detected")
            results[animal_id] = {"status": "FAILED_PEAKS", "n_beats": len(peaks), "hr": hr}
            continue

        template, t_ms, beats = average_beats(filtered, peaks)
        feats = extract_morphology_features(template, t_ms, beats)
        feats["qtc_ms"]      = mitchell_qtc(feats["qt_ms"], rr_mean)
        feats["qtc_formula"] = QTC_FORMULA_NAME

        record = {
            "animal_id":      animal_id,
            "recording_date": rec_date,
            "n_beats":        int(len(peaks)),
            "heart_rate_bpm": float(hr),
            "rr_mean_ms":     rr_mean,
            "rr_std_ms":      float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0,
            "window_rule":    rule,
            "inverted":       inverted,
            **feats,
        }
        results[animal_id] = {"status": "OK", "record": record, "rec_date": rec_date}

        print(f"\n  QRS duration : {feats['qrs_duration_ms']:.1f} ms")
        print(f"  R amplitude  : {feats['r_amplitude_mv']:.3f} mV")
        print(f"  QT (mean)    : {feats['qt_ms']:.1f} ms")
        print(f"  QTc (Mitchell): {feats['qtc_ms']:.1f} ms")

    except Exception as exc:
        import traceback
        print(f"  EXCEPTION: {exc}")
        traceback.print_exc()
        results[animal_id] = {"status": "FAILED_EXCEPTION", "error": str(exc)}

print("\n" + "=" * 64)
print("SUMMARY")
print("=" * 64)
for aid in sorted(TARGET_ANIMALS):
    r = results.get(aid, {"status": "NOT_FOUND"})
    if r["status"] == "OK":
        rec = r["record"]
        flag = "" if HR_FLAG_LOW <= rec["heart_rate_bpm"] <= HR_FLAG_HIGH else "  <<< FLAG"
        print(f"  Animal {aid}: OK  |  HR={rec['heart_rate_bpm']:.1f} bpm{flag}  |  beats={rec['n_beats']}")
    else:
        print(f"  Animal {aid}: {r['status']}")

# Save records for Task 1 append (used by next script)
import pickle
pickle_path = OUT / "_new_animals_103_110.pkl"
to_save = {aid: r for aid, r in results.items() if r["status"] == "OK"}
with open(pickle_path, "wb") as f:
    pickle.dump(to_save, f)
print(f"\nFeature records saved to: {pickle_path}")
print("(Run append_and_cluster.py next to complete Tasks 1-3)")
