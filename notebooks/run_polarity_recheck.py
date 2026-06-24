"""
Polarity fix validation script.
Runs OLD and NEW detect_r_peaks on every .txt file in data/,
prints a before/after table, and writes polarity_recheck.csv.

OLD logic : HR-window only, greedy (matches current NB02).
NEW logic  : non-greedy; resolves ties with signal skewness.
  Tier 1 (both in HR window)  -> amplitude comparison
  Tier 2 (only inv in range)  -> require skewness < 0 to trust it
  Tier 3 (neither in range)   -> skewness is final arbiter

Do NOT import from the notebook — all helpers are copied verbatim from
NB02 Cell 1 so this script is self-contained and auditable.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
while not (ROOT / "data").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA_DIR    = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

# ── constants (identical to NB02) ────────────────────────────────────────────
FS               = 1000
BANDPASS_LOW_HZ  = 0.5
BANDPASS_HIGH_HZ = 150.0
BANDPASS_ORDER   = 2
REFRACTORY_MS    = 55
HR_ACCEPT_LOW    = 300
HR_ACCEPT_HIGH   = 700
GAP_MIN_S        = 10
GAP_MAX_S        = 120
GAP_MAX_LONG_S   = 900
LONG_USE_S       = 60
SINGLE_FALLBACK_S = 30
EXCLUDE_KEYWORDS = ("oc 1","oc 2","oc1","oc2","jugular","jugcan","injection","inject")
END_KEYWORDS     = ("end","done","stop")
START_HINTS      = ("baseline","ecg","start")
_MARKER_RE       = re.compile(r"#([*1-3])")


# ── helpers (verbatim from NB02 Cell 1) ──────────────────────────────────────

def parse_filename(path):
    nums = re.findall(r"\d+", path.stem)
    try:
        if len(nums) >= 4:
            y, m, d, a = int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
            return a, pd.Timestamp(year=y, month=m, day=d)
        if len(nums) == 3:
            y, m, a = int(nums[0]), int(nums[1]), int(nums[2])
            return a, pd.Timestamp(year=y, month=m, day=1)
    except (ValueError, TypeError):
        pass
    return None, None


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


def _looks_like_end(text):      return any(k in text.lower() for k in END_KEYWORDS)
def _looks_like_excluded(text): return any(k in text.lower() for k in EXCLUDE_KEYWORDS)
def _looks_like_baseline(text): return "baseline" in text.lower().replace("basline","baseline")


def load_ecg_file(path, n_header, ecg_col):
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


def find_baseline_window(markers, n_samples, fs=FS):
    MIN_GAP  = GAP_MIN_S      * fs
    MAX_GAP  = GAP_MAX_S      * fs
    LONG_GAP = GAP_MAX_LONG_S * fs
    LONG_USE = LONG_USE_S     * fs
    FALLBACK = SINGLE_FALLBACK_S * fs

    tagged = []
    for idx, text, prefix in markers:
        if _looks_like_excluded(text):
            continue
        end_like = _looks_like_end(text)
        tagged.append({"idx": idx, "text": text, "prefix": prefix, "end": end_like})

    starts = [m for m in tagged if not m["end"]]
    ends   = [m for m in tagged if m["end"]]

    def _try(_starts, _ends, lo, hi, label):
        for s in _starts:
            for e in _ends:
                if e["idx"] <= s["idx"]:
                    continue
                gap = e["idx"] - s["idx"]
                if lo <= gap <= hi:
                    return s, e, label
        return None

    for pre in ("*", "1", "2", "3"):
        result = _try([m for m in starts if m["prefix"]==pre],
                      [m for m in ends   if m["prefix"]==pre],
                      MIN_GAP, MAX_GAP, f"A_same_{pre}")
        if result:
            s, e, label = result
            return s["idx"], min(e["idx"], n_samples), label, s["text"], e["text"]

    result = _try(starts, ends, MIN_GAP, MAX_GAP, "B_cross")
    if result:
        s, e, label = result
        return s["idx"], min(e["idx"], n_samples), label, s["text"], e["text"]

    bl_starts = [m for m in starts if _looks_like_baseline(m["text"])]
    bl_ends   = [m for m in ends   if _looks_like_baseline(m["text"])]
    result = _try(bl_starts, bl_ends, MIN_GAP, LONG_GAP, "C_long_baseline")
    if result:
        s, e, label = result
        end_idx = min(s["idx"] + LONG_USE, e["idx"], n_samples)
        return s["idx"], end_idx, label, s["text"], e["text"]

    candidate = None
    for m in starts:
        if any(k in m["text"].lower() for k in START_HINTS):
            candidate = m
            break
    if candidate is None and starts:
        candidate = starts[0]
    if candidate is None and tagged:
        candidate = tagged[0]
    if candidate is not None:
        end_idx = min(candidate["idx"] + FALLBACK, n_samples)
        return candidate["idx"], end_idx, "D_single_30s", candidate["text"], None

    return None, None, "no_markers", None, None


def bandpass_filter(x, fs=FS, low=BANDPASS_LOW_HZ, high=BANDPASS_HIGH_HZ, order=BANDPASS_ORDER):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    y = filtfilt(b, a, x)
    for f0 in (50, 100, 150):
        bn, an = iirnotch(f0 / (fs / 2), Q=30)
        y = filtfilt(bn, an, y)
    return y


def _peaks_with_mad(sig, fs=FS, refractory_ms=REFRACTORY_MS):
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


def _hr(peaks, fs=FS):
    if len(peaks) < 2:
        return np.nan
    return 60000.0 / np.mean(np.diff(peaks) * (1000.0 / fs))


# ── OLD detect_r_peaks (current NB02 logic) ──────────────────────────────────

def detect_r_peaks_old(sig, fs=FS):
    pos_peaks, pos_h, _ = _peaks_with_mad(sig, fs)
    pos_hr = _hr(pos_peaks, fs)
    pos_ok = HR_ACCEPT_LOW <= pos_hr <= HR_ACCEPT_HIGH if not np.isnan(pos_hr) else False
    if pos_ok:
        return pos_peaks, False, pos_hr, pos_h

    inv_peaks, inv_h, _ = _peaks_with_mad(-sig, fs)
    inv_hr = _hr(inv_peaks, fs)
    inv_ok = HR_ACCEPT_LOW <= inv_hr <= HR_ACCEPT_HIGH if not np.isnan(inv_hr) else False
    if inv_ok:
        return inv_peaks, True, inv_hr, inv_h

    def _dist(hr):
        if np.isnan(hr): return np.inf
        if hr < HR_ACCEPT_LOW: return HR_ACCEPT_LOW - hr
        if hr > HR_ACCEPT_HIGH: return hr - HR_ACCEPT_HIGH
        return 0.0

    if _dist(inv_hr) < _dist(pos_hr):
        return inv_peaks, True, inv_hr, inv_h
    return pos_peaks, False, pos_hr, pos_h


# ── NEW detect_r_peaks (proposed fix) ────────────────────────────────────────

def detect_r_peaks_new(sig, fs=FS):
    """Non-greedy polarity detection with signal skewness confirmation.

    Tier 1  both in HR window  -> amplitude: true R-peaks are taller than
                                   spurious T-wave detections.
    Tier 2  only inv in range  -> require skewness < 0 to trust it; a flat/
                                   noisy window can accidentally pass the HR
                                   test in the inverted direction.
    Tier 3  neither in range   -> skewness is the final arbiter; robust even
                                   on noisy/partial windows because asymmetric
                                   R-peak outliers shift the distribution.
    """
    pos_peaks, pos_h, _ = _peaks_with_mad(sig, fs)
    pos_hr = _hr(pos_peaks, fs)
    pos_ok = HR_ACCEPT_LOW <= pos_hr <= HR_ACCEPT_HIGH if not np.isnan(pos_hr) else False

    inv_peaks, inv_h, _ = _peaks_with_mad(-sig, fs)
    inv_hr = _hr(inv_peaks, fs)
    inv_ok = HR_ACCEPT_LOW <= inv_hr <= HR_ACCEPT_HIGH if not np.isnan(inv_hr) else False

    # Signal skewness: negative = dominant outliers point down (inverted R-peaks).
    # Upright ECG -> positive skew. Flat/noise window -> near-zero skew.
    _std = float(np.std(sig))
    skewness = (float(np.mean(((sig - np.mean(sig)) / _std) ** 3))
                if _std > 1e-9 else 0.0)

    # Tier 1: both in HR window — HR alone cannot discriminate.
    if pos_ok and inv_ok:
        pos_amp = float(np.median(sig[pos_peaks]))
        inv_amp = float(np.median(-sig[inv_peaks]))
        if inv_amp > pos_amp:
            return inv_peaks, True, inv_hr, inv_h
        return pos_peaks, False, pos_hr, pos_h

    # Tier 2: upright is unambiguous (HR + implied positive skew).
    if pos_ok:
        return pos_peaks, False, pos_hr, pos_h

    # Tier 2: inverted in range — only trust it if skewness agrees.
    if inv_ok and skewness < 0:
        return inv_peaks, True, inv_hr, inv_h
    # inv_ok with skewness >= 0: flat/noisy window falsely passing the inverted
    # HR test — fall through to Tier 3.

    # Tier 3: neither in range (or inv_ok overridden above).
    # Skewness decides; returns inv_peaks even if inv_hr is out of range so the
    # polarity flag is correct and the animal goes to NEEDS_REVIEW for HR.
    if skewness < 0:
        return inv_peaks, True, inv_hr, inv_h
    return pos_peaks, False, pos_hr, pos_h


# ── Main loop ────────────────────────────────────────────────────────────────

def process_file(path, fn):
    """Run detect_r_peaks function fn on one file. Returns dict with diagnostics."""
    animal_id, _ = parse_filename(path)
    if animal_id is None:
        return None

    try:
        n_header, _, ecg_col, _ = parse_header_and_layout(path)
        voltage, markers = load_ecg_file(path, n_header, ecg_col)
        if voltage.size == 0:
            return None

        med_abs = float(np.median(np.abs(voltage)))
        if med_abs > 50:
            return None

        start, end, rule, _, _ = find_baseline_window(markers, len(voltage))
        if start is None:
            return None

        segment = voltage[start:end]
        if segment.size < 2 * FS:
            return None

        filtered = bandpass_filter(segment)

        # Compute diagnostics for logging
        _std = float(np.std(filtered))
        skewness = (float(np.mean(((filtered - np.mean(filtered)) / _std) ** 3))
                    if _std > 1e-9 else 0.0)

        pos_peaks_d, pos_h_d, _ = _peaks_with_mad(filtered, FS)
        pos_hr_d = _hr(pos_peaks_d)
        pos_ok_d = HR_ACCEPT_LOW <= pos_hr_d <= HR_ACCEPT_HIGH if not np.isnan(pos_hr_d) else False

        inv_peaks_d, inv_h_d, _ = _peaks_with_mad(-filtered, FS)
        inv_hr_d = _hr(inv_peaks_d)
        inv_ok_d = HR_ACCEPT_LOW <= inv_hr_d <= HR_ACCEPT_HIGH if not np.isnan(inv_hr_d) else False

        pos_amp_d = float(np.median(filtered[pos_peaks_d])) if len(pos_peaks_d) > 0 else np.nan
        inv_amp_d = float(np.median(-filtered[inv_peaks_d])) if len(inv_peaks_d) > 0 else np.nan

        _, inv_old, hr_old, _ = fn(filtered)  # only care about old inverted flag

        return {
            "animal_id":  animal_id,
            "inverted":   inv_old,
            "hr_bpm":     round(float(hr_old), 1) if not np.isnan(hr_old) else float("nan"),
            "pos_ok":     pos_ok_d,
            "inv_ok":     inv_ok_d,
            "pos_hr":     round(float(pos_hr_d), 1) if not np.isnan(pos_hr_d) else float("nan"),
            "inv_hr":     round(float(inv_hr_d), 1) if not np.isnan(inv_hr_d) else float("nan"),
            "pos_amp_mv": round(pos_amp_d, 4) if not np.isnan(pos_amp_d) else float("nan"),
            "inv_amp_mv": round(inv_amp_d, 4) if not np.isnan(inv_amp_d) else float("nan"),
            "skewness":   round(skewness, 3),
        }
    except Exception as e:
        return {"animal_id": animal_id, "error": str(e)}


files = sorted(DATA_DIR.glob("*.txt"))
print(f"Files found: {len(files)}\n")

old_rows, new_rows = [], []
for path in files:
    r_old = process_file(path, detect_r_peaks_old)
    r_new = process_file(path, detect_r_peaks_new)
    if r_old is not None and "error" not in r_old:
        old_rows.append(r_old)
    if r_new is not None and "error" not in r_new:
        new_rows.append(r_new)

df_old = pd.DataFrame(old_rows).set_index("animal_id")
df_new = pd.DataFrame(new_rows).set_index("animal_id")

# ── Before/after comparison ──────────────────────────────────────────────────
all_ids = sorted(set(df_old.index) | set(df_new.index))
rows = []
for aid in all_ids:
    old_inv = df_old.loc[aid, "inverted"] if aid in df_old.index else None
    new_inv = df_new.loc[aid, "inverted"] if aid in df_new.index else None
    old_hr  = df_old.loc[aid, "hr_bpm"]  if aid in df_old.index else float("nan")
    new_hr  = df_new.loc[aid, "hr_bpm"]  if aid in df_new.index else float("nan")
    changed = (old_inv != new_inv)
    rows.append({
        "animal_id":    aid,
        "old_inverted": old_inv,
        "new_inverted": new_inv,
        "old_hr_bpm":   old_hr,
        "new_hr_bpm":   new_hr,
        "changed":      changed,
        # diagnostics
        "pos_ok":       df_new.loc[aid, "pos_ok"]     if aid in df_new.index else None,
        "inv_ok":       df_new.loc[aid, "inv_ok"]     if aid in df_new.index else None,
        "pos_hr":       df_new.loc[aid, "pos_hr"]     if aid in df_new.index else float("nan"),
        "inv_hr":       df_new.loc[aid, "inv_hr"]     if aid in df_new.index else float("nan"),
        "pos_amp_mv":   df_new.loc[aid, "pos_amp_mv"] if aid in df_new.index else float("nan"),
        "inv_amp_mv":   df_new.loc[aid, "inv_amp_mv"] if aid in df_new.index else float("nan"),
        "skewness":     df_new.loc[aid, "skewness"]   if aid in df_new.index else float("nan"),
    })

comp = pd.DataFrame(rows)
changed_df = comp[comp["changed"]]

print("=" * 70)
print("BEFORE / AFTER COMPARISON  (full table, sorted by animal_id)")
print("=" * 70)
with pd.option_context("display.max_rows", 200, "display.width", 200):
    print(comp[["animal_id","old_inverted","new_inverted","old_hr_bpm",
                "new_hr_bpm","changed","skewness","pos_ok","inv_ok",
                "pos_hr","inv_hr","pos_amp_mv","inv_amp_mv"]].to_string(index=False))

print()
print("=" * 70)
print(f"CHANGED ANIMALS  ({len(changed_df)} total)")
print("=" * 70)
if len(changed_df) == 0:
    print("  (none)")
else:
    with pd.option_context("display.max_rows", 50, "display.width", 200):
        print(changed_df[["animal_id","old_inverted","new_inverted","old_hr_bpm",
                           "new_hr_bpm","skewness","pos_ok","inv_ok",
                           "pos_hr","inv_hr","pos_amp_mv","inv_amp_mv"]].to_string(index=False))

# ── Targeted validation: 103, 109, 114 ──────────────────────────────────────
print()
print("=" * 70)
print("TARGETED VALIDATION  (animals 103, 109, 114)")
print("=" * 70)
targets = comp[comp["animal_id"].isin([103, 109, 114])]
for _, r in targets.iterrows():
    aid = int(r["animal_id"])
    ok_inv = {103: True, 109: True, 114: False}[aid]
    hr_ok = (350 <= r["new_hr_bpm"] <= 700) if not np.isnan(r["new_hr_bpm"]) else False
    flag  = "PASS" if (r["new_inverted"] == ok_inv) else "FAIL"
    print(f"  Animal {aid}: new_inverted={r['new_inverted']}  "
          f"(expected {ok_inv})  [{flag}]  "
          f"HR={r['new_hr_bpm']:.1f} bpm  "
          f"HR_in_range={'YES' if hr_ok else 'NO'}  "
          f"skewness={r['skewness']:.3f}")

# ── Save audit ───────────────────────────────────────────────────────────────
out_path = OUTPUTS_DIR / "polarity_recheck.csv"
comp.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
