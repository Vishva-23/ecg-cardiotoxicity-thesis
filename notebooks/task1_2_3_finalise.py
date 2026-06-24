"""
Three tasks in sequence:
  Task 2 : Remove animal 102 from all_animals_features.csv (documented).
  Task 1b: Scan animal 103 full recording for best passing window (HR 350-700, rr_std <= 120).
           Extract features, add row to CSV.
           Animal 110 flagged NEEDS_REVIEW — not included.
  Task 3 : Recompute all six cluster columns on the updated matrix. Regenerate pca_corrected.png.
"""
import re, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

# ── project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
while not (ROOT / "data").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
OUT      = ROOT / "outputs"
FIG      = OUT / "figures"

# ── pipeline constants (NB02-identical) ──────────────────────────────────────
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
GAP_MIN_S        = 10
GAP_MAX_S        = 120
GAP_MAX_LONG_S   = 900
LONG_USE_S       = 60
SINGLE_FALLBACK_S = 30
EXCLUDE_KEYWORDS = ("oc 1","oc 2","oc1","oc2","jugular","jugcan","injection","inject")
END_KEYWORDS     = ("end","done","stop")
START_HINTS      = ("baseline","ecg","start")
_MARKER_RE       = re.compile(r"#([*1-3])")
QTC_FORMULA_NAME = "Mitchell"

# scan-specific gates
HR_GATE_LOW   = 350
HR_GATE_HIGH  = 700
RR_STD_CAP    = 120
WINDOW_S      = 30
STEP_S        = 5

# cluster feature columns — NB02 Cell 19 definition minus rt_interval_ms
# (rt_interval_ms is not written to all_animals_features.csv)
CLUSTER_FEAT_COLS = [
    "heart_rate_bpm", "rr_mean_ms", "rr_std_ms",
    "r_amplitude_mv", "qrs_duration_ms",
    "j_wave_amplitude_mv", "t_wave_amplitude_mv",
    "qt_ms", "qtc_ms",
    "qt_std_ms", "qrs_std_ms", "r_amplitude_std_mv",
    "rr_cv",
]

# ── pipeline helpers (verbatim from NB02 Cell 4) ─────────────────────────────

def parse_filename(path):
    nums = re.findall(r"\d+", path.stem)
    try:
        if len(nums) >= 4:
            y, m, d, a = int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
            if y > 3000: y = y % 10000
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
    return n_header, lead_cols + ch3_idx


def load_ecg_file(path, n_header, ecg_col):
    voltages = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(n_header): f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= ecg_col: continue
            try: v = float(parts[ecg_col])
            except ValueError: v = np.nan
            voltages.append(v)
    voltage = np.asarray(voltages, dtype=float)
    if np.isnan(voltage).any():
        idx = np.arange(len(voltage)); good = ~np.isnan(voltage)
        if good.any(): voltage = np.interp(idx, idx[good], voltage[good])
    return voltage


def bandpass_filter(x):
    nyq = 0.5 * FS
    b, a = butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ/nyq, BANDPASS_HIGH_HZ/nyq], btype="band")
    y = filtfilt(b, a, x)
    for f0 in (50, 100, 150):
        bn, an = iirnotch(f0 / (FS/2), Q=30)
        y = filtfilt(bn, an, y)
    return y


def _peaks_with_mad(sig):
    if sig.size == 0: return np.array([], dtype=int), np.nan, np.nan
    med = float(np.median(sig)); mad = float(np.median(np.abs(sig - med)))
    if mad <= 0: return np.array([], dtype=int), med, mad
    height = med + 4*mad; prominence = max(0.3*mad, 0.005)
    distance = int(REFRACTORY_MS * FS / 1000)
    peaks, _ = find_peaks(sig, height=height, distance=distance, prominence=prominence)
    return peaks, height, prominence


def detect_r_peaks(sig):
    def hr(p): return 60000./np.mean(np.diff(p)*1000./FS) if len(p) >= 2 else np.nan
    pp, ph, _ = _peaks_with_mad(sig); phr = hr(pp)
    if not np.isnan(phr) and HR_ACCEPT_LOW <= phr <= HR_ACCEPT_HIGH:
        return pp, False, phr, ph
    ip, ih, _ = _peaks_with_mad(-sig); ihr = hr(ip)
    if not np.isnan(ihr) and HR_ACCEPT_LOW <= ihr <= HR_ACCEPT_HIGH:
        return ip, True, ihr, ih
    def dist(h):
        if np.isnan(h): return np.inf
        return max(0, HR_ACCEPT_LOW - h, h - HR_ACCEPT_HIGH)
    if dist(ihr) < dist(phr): return ip, True, ihr, ih
    return pp, False, phr, ph


def average_beats(sig, r_peaks):
    pre = int(PRE_R_MS*FS/1000); post = int(POST_R_MS*FS/1000)
    win_len = pre + post; t_ms = (np.arange(win_len)-pre)*(1000./FS)
    beats = [sig[r-pre:r+post] for r in r_peaks if r-pre >= 0 and r+post <= len(sig)]
    if not beats: return None, t_ms, None
    beats = np.vstack(beats)
    n_full = len(beats) // BEATS_TO_AVERAGE
    if n_full == 0: return beats.mean(axis=0), t_ms, beats
    grouped = beats[:n_full*BEATS_TO_AVERAGE].reshape(n_full, BEATS_TO_AVERAGE, win_len)
    return grouped.mean(axis=1).mean(axis=0), t_ms, beats


def _measure_one(sig, t_ms):
    r_idx = int(np.argmax(sig)); r_amp = float(sig[r_idx]); half = r_amp/2.
    left = r_idx
    while left > 0 and sig[left] > half: left -= 1
    right = r_idx
    while right < len(sig)-1 and sig[right] > half: right += 1
    qrs_ms = float(t_ms[right]-t_ms[left])
    def wmax(lo, hi):
        mask = (t_ms >= lo) & (t_ms <= hi)
        if not mask.any(): return np.nan, np.nan
        seg = sig[mask]; ts = t_ms[mask]; k = int(np.argmax(seg))
        return float(seg[k]), float(ts[k])
    j_amp, _      = wmax(10, 30)
    t_amp, t_time = wmax(40, 80)
    rt_ms = float(t_time) if not np.isnan(t_time) else np.nan
    baseline = float(np.median(sig[t_ms < -70])) if (t_ms < -70).any() else 0.
    tol = 0.05*r_amp; qt_ms = np.nan
    for i in np.where(t_ms >= 40)[0]:
        if abs(sig[i]-baseline) <= tol: qt_ms = float(t_ms[i]); break
    return r_amp, qrs_ms, j_amp, t_amp, rt_ms, qt_ms


def _qt_per_beat(beat, t_ms):
    n30=int(30*FS/1000); n100=int(100*FS/1000); n40=int(40*FS/1000)
    n50=int(50*FS/1000); n20=int(20*FS/1000)
    r_idx = int(np.argmin(np.abs(t_ms)))
    s = r_idx+n30; e = min(len(beat), r_idx+n100)
    if e <= s: return np.nan
    t_peak = s+int(np.argmax(beat[s:e]))
    after_t = beat[t_peak: min(len(beat), t_peak+n40)]
    lo = max(0, r_idx-n50); hi = max(lo+1, r_idx-n20)
    baseline = float(np.mean(beat[lo:hi]))
    near = np.where(after_t <= baseline+0.01)[0]
    if len(near) == 0: return np.nan
    qt = (t_peak+int(near[0])-r_idx)*1000./FS
    return float(qt) if 30 <= qt <= 100 else np.nan


def extract_morphology_features(template, t_ms, beats=None):
    keys = ["r_amplitude_mv","qrs_duration_ms","j_wave_amplitude_mv",
            "t_wave_amplitude_mv","rt_interval_ms","qt_ms",
            "qt_std_ms","qrs_std_ms","r_amplitude_std_mv"]
    if template is None: return dict.fromkeys(keys, np.nan)
    r_amp, qrs_ms, j_amp, t_amp, rt_ms, qt_template = _measure_one(template, t_ms)
    qt_ms = qt_template; qt_std_ms = qrs_std_ms = r_amplitude_std_mv = np.nan
    if beats is not None and getattr(beats,"ndim",0) == 2 and len(beats) > 1:
        r_amps, qrs_durs, qt_est = [], [], []
        for b in beats:
            br, bqrs, _, _, _, _ = _measure_one(b, t_ms)
            r_amps.append(br); qrs_durs.append(bqrs)
            bqt = _qt_per_beat(b, t_ms)
            if not np.isnan(bqt): qt_est.append(bqt)
        if qt_est: qt_ms = float(np.mean(qt_est))
        qt_std_ms          = float(np.std(qt_est,  ddof=1)) if len(qt_est)  > 1 else np.nan
        qrs_std_ms         = float(np.std(qrs_durs,ddof=1)) if len(qrs_durs)> 1 else np.nan
        r_amplitude_std_mv = float(np.std(r_amps,  ddof=1)) if len(r_amps)  > 1 else np.nan
    return {"r_amplitude_mv": r_amp, "qrs_duration_ms": qrs_ms,
            "j_wave_amplitude_mv": j_amp, "t_wave_amplitude_mv": t_amp,
            "rt_interval_ms": rt_ms, "qt_ms": qt_ms,
            "qt_std_ms": qt_std_ms, "qrs_std_ms": qrs_std_ms,
            "r_amplitude_std_mv": r_amplitude_std_mv}


def mitchell_qtc(qt_ms, rr_ms):
    if np.isnan(qt_ms) or np.isnan(rr_ms) or rr_ms <= 0: return np.nan
    return float(qt_ms / np.sqrt(rr_ms/100.))


# ── TASK 2: Remove animal 102 ─────────────────────────────────────────────────
print("="*64)
print("TASK 2 — Remove animal 102 from all_animals_features.csv")
print("="*64)

CSV_PATH = OUT / "all_animals_features.csv"
df = pd.read_csv(CSV_PATH)
print(f"  Rows before : {len(df)}")
mask_102 = df["animal_id"] == 102
if mask_102.sum() == 0:
    print("  Animal 102 already removed — skipping.")
else:
    assert mask_102.sum() == 1, f"Expected 1 row for animal 102, found {mask_102.sum()}"
    df = df[~mask_102].reset_index(drop=True)
    print(f"  Rows after  : {len(df)}")
    print(f"  Removal reason: excluded — persistent 50 Hz mains interference from heated mat, "
          f"no clean window recoverable, confirmed by supervisor Roisin Kelly-Laubscher.")
    # write immediately so Task 1b appends to the already-clean file
    df.to_csv(CSV_PATH, index=False)
    print(f"  Saved: {CSV_PATH}")


# ── TASK 1b: Quality scan — Animal 103 ───────────────────────────────────────
print("\n" + "="*64)
print("TASK 1b — Quality scan for animal 103 (full recording)")
print("="*64)

path_103 = next(DATA_DIR.glob("*103.txt"), None)
assert path_103 is not None, "Animal 103 file not found in data/"
print(f"  File: {path_103.name}")

n_header, ecg_col = parse_header_and_layout(path_103)
voltage_103 = load_ecg_file(path_103, n_header, ecg_col)
n_samples = len(voltage_103)
rec_s = n_samples / FS
print(f"  Recording length: {rec_s:.1f} s  ({rec_s/60:.2f} min)  |  {n_samples:,} samples")

WIN  = int(WINDOW_S * FS)
STEP = int(STEP_S * FS)

scan_rows = []
c = 0
while c + WIN <= n_samples:
    seg = voltage_103[c : c+WIN]
    if seg.size < 2*FS:
        c += STEP; continue
    filtered = bandpass_filter(seg)
    peaks, inv, hr, _ = detect_r_peaks(filtered)
    rr_ms_arr = np.diff(peaks) * (1000./FS) if len(peaks) > 1 else np.array([])
    rr_std = float(np.std(rr_ms_arr, ddof=1)) if len(rr_ms_arr) > 1 else np.nan
    passes = (len(peaks) >= 10
              and not np.isnan(hr)
              and HR_GATE_LOW <= hr <= HR_GATE_HIGH
              and not np.isnan(rr_std)
              and rr_std <= RR_STD_CAP)
    scan_rows.append(dict(
        start_s=round(c/FS, 1), end_s=round((c+WIN)/FS, 1),
        hr=round(hr, 1) if not np.isnan(hr) else np.nan,
        rr_std=round(rr_std, 1) if not np.isnan(rr_std) else np.nan,
        n_beats=int(len(peaks)), inverted=inv, passes=passes
    ))
    c += STEP

scan_df = pd.DataFrame(scan_rows)
passing = scan_df[scan_df["passes"]]
print(f"  Windows scanned : {len(scan_df)}")
print(f"  PASSING windows : {len(passing)}  (HR {HR_GATE_LOW}-{HR_GATE_HIGH} bpm, rr_std <= {RR_STD_CAP} ms)")

if len(passing) == 0:
    print("  >> ZERO passing windows — animal 103 UNUSABLE by this gate.")
    print("  >> Flagging as NEEDS_REVIEW, not adding to features CSV.")
    ANIMAL_103_STATUS = "NEEDS_REVIEW"
else:
    # best = lowest rr_std
    best = passing.sort_values("rr_std").iloc[0]
    print(f"\n  Best window: start={best['start_s']:.1f} s  "
          f"HR={best['hr']:.1f} bpm  rr_std={best['rr_std']:.1f} ms  beats={best['n_beats']}")

    # Show top 5 windows
    print("\n  Top 5 passing windows (sorted by rr_std):")
    top5 = passing.sort_values("rr_std").head(5)
    for _, row in top5.iterrows():
        print(f"    t={row.start_s:.1f}-{row.end_s:.1f} s  HR={row.hr:.1f}  "
              f"rr_std={row.rr_std:.1f}  beats={row.n_beats}")

    # Extract features from best window
    ws = int(best["start_s"] * FS)
    we = ws + WIN
    seg_best = voltage_103[ws:we]
    filt_best = bandpass_filter(seg_best)
    peaks_best, inv_best, hr_best, _ = detect_r_peaks(filt_best)
    if inv_best: filt_best = -filt_best
    rr_arr = np.diff(peaks_best) * (1000./FS)
    rr_mean = float(np.mean(rr_arr)); rr_std_val = float(np.std(rr_arr, ddof=1)) if len(rr_arr)>1 else 0.
    template, t_ms, beats_mat = average_beats(filt_best, peaks_best)
    feats = extract_morphology_features(template, t_ms, beats_mat)
    feats["qtc_ms"]      = mitchell_qtc(feats["qt_ms"], rr_mean)
    feats["qtc_formula"] = QTC_FORMULA_NAME

    # recording date from filename
    _, rec_date = parse_filename(path_103)
    study_guess = "acute"  # animal 103 is in the 100s (acute group)

    new_row = {
        "animal_id":      103,
        "recording_date": rec_date.date() if rec_date else "2024-11-01",
        "study_guess":    study_guess,
        "n_beats":        int(len(peaks_best)),
        "heart_rate_bpm": float(hr_best),
        "rr_mean_ms":     rr_mean,
        "rr_std_ms":      rr_std_val,
        **feats,
        "rr_cv":          rr_std_val / rr_mean if rr_mean > 0 else np.nan,
        "window_source":  f"quality_scan_{best['start_s']:.0f}s",
    }
    # cluster columns will be filled in Task 3 after recompute
    for col in ["cluster_k2","cluster_k5","cluster_hier_k2",
                "cluster_hier_k5","cluster_gmm_k2","cluster_gmm_k5"]:
        new_row[col] = np.nan

    print(f"\n  Feature row for animal 103:")
    print(f"    HR={new_row['heart_rate_bpm']:.1f} bpm  n_beats={new_row['n_beats']}  "
          f"rr_std={new_row['rr_std_ms']:.1f} ms  QTc={new_row['qtc_ms']:.1f} ms")
    print(f"    window_source: {new_row['window_source']}")

    # Append to CSV
    df = pd.read_csv(CSV_PATH)
    # guard: don't double-add
    if 103 in df["animal_id"].values:
        df = df[df["animal_id"] != 103].reset_index(drop=True)
        print("  (replaced existing animal 103 row)")
    new_df = pd.DataFrame([new_row])
    # align columns
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[df.columns]
    df = pd.concat([df, new_df], ignore_index=True).sort_values("animal_id").reset_index(drop=True)
    print(f"  Rows after adding 103: {len(df)}")
    ANIMAL_103_STATUS = "OK"

    # write (cluster columns will be overwritten in Task 3)
    df.to_csv(CSV_PATH, index=False)
    print(f"  Saved (pre-cluster): {CSV_PATH}")


# ── Animal 110: flag NEEDS_REVIEW, do not include ────────────────────────────
print("\n" + "="*64)
print("TASK 1b — Animal 110: NEEDS_REVIEW, not included")
print("="*64)
print("  Reason: no annotation markers in file; first-60s fallback gives HR=188.5 bpm")
print("  (below 350 bpm gate); signal inverted; recording quality uncertain.")
print("  Action: flag for Roisin manual inspection. Not added to features CSV.")


# ── TASK 3: Recompute all cluster columns ────────────────────────────────────
print("\n" + "="*64)
print("TASK 3 — Recompute cluster columns on updated feature matrix")
print("="*64)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster

df = pd.read_csv(CSV_PATH)
print(f"  Animals in matrix: {len(df)}  (IDs: {sorted(df['animal_id'].tolist())})")

# compute rr_cv if not present or has nulls
df["rr_cv"] = df["rr_std_ms"] / df["rr_mean_ms"]

# build feature matrix
X_raw    = df[CLUSTER_FEAT_COLS].copy()
X_filled = X_raw.fillna(X_raw.median(numeric_only=True))
X_scaled = StandardScaler().fit_transform(X_filled)
print(f"  Feature matrix: {X_scaled.shape[0]} x {X_scaled.shape[1]}")

# KMeans
df["cluster_k2"] = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X_scaled)
df["cluster_k5"] = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(X_scaled)
print(f"  KMeans k=2 sizes: {dict(pd.Series(df['cluster_k2']).value_counts().sort_index())}")
print(f"  KMeans k=5 sizes: {dict(pd.Series(df['cluster_k5']).value_counts().sort_index())}")

# Ward hierarchical
Z = linkage(X_scaled, method="ward")
df["cluster_hier_k2"] = fcluster(Z, t=2, criterion="maxclust")
df["cluster_hier_k5"] = fcluster(Z, t=5, criterion="maxclust")
print(f"  Ward k=2 sizes  : {dict(pd.Series(df['cluster_hier_k2']).value_counts().sort_index())}")
print(f"  Ward k=5 sizes  : {dict(pd.Series(df['cluster_hier_k5']).value_counts().sort_index())}")

# GMM
df["cluster_gmm_k2"] = GaussianMixture(n_components=2, random_state=42).fit_predict(X_scaled)
df["cluster_gmm_k5"] = GaussianMixture(n_components=5, random_state=42).fit_predict(X_scaled)
print(f"  GMM k=2 sizes   : {dict(pd.Series(df['cluster_gmm_k2']).value_counts().sort_index())}")
print(f"  GMM k=5 sizes   : {dict(pd.Series(df['cluster_gmm_k5']).value_counts().sort_index())}")

# Save final CSV
df.to_csv(CSV_PATH, index=False)
print(f"\n  Saved final CSV: {CSV_PATH}  ({len(df)} rows)")


# ── PCA figure — pca_corrected.png ───────────────────────────────────────────
print("\n  Generating pca_corrected.png ...")

pca2 = PCA(n_components=2, random_state=42)
coords = pca2.fit_transform(X_scaled)
pc1_pct, pc2_pct = pca2.explained_variance_ratio_[:2] * 100
ids = df["animal_id"].astype(int).values

# highlight animal 103 as new addition; 115 as persistent outlier
NEW_IDS      = {103}
OUTLIER_IDS  = {115}

fig, ax = plt.subplots(figsize=(15, 11))
for i, (x, y) in enumerate(coords):
    aid = int(ids[i])
    if aid in OUTLIER_IDS:
        ax.scatter(x, y, s=160, color="crimson", edgecolor="black", lw=1.3, zorder=5)
    elif aid in NEW_IDS:
        ax.scatter(x, y, s=120, color="gold", edgecolor="black", lw=1.3, zorder=5, marker="D")
    else:
        ax.scatter(x, y, s=40, color="steelblue", alpha=0.75, zorder=3)
    size = 9 if (aid in OUTLIER_IDS or aid in NEW_IDS) else 6
    weight = "bold" if (aid in OUTLIER_IDS or aid in NEW_IDS) else "normal"
    col = "black" if (aid in OUTLIER_IDS or aid in NEW_IDS) else "dimgray"
    ax.text(x+0.12, y+0.10, str(aid), fontsize=size, fontweight=weight, color=col)

# custom legend
from matplotlib.lines import Line2D
n_new = sum(1 for aid in ids if aid in NEW_IDS)
n_out = sum(1 for aid in ids if aid in OUTLIER_IDS)
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="steelblue",
           markersize=8, label=f"animals in analysis (n={len(df)-n_out})"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="crimson", markeredgecolor="black",
           markersize=10, label="persistent outlier (animal 115) — supervisor review"),
]
if n_new > 0:
    legend_elements.insert(1, Line2D([0],[0], marker="D", color="w",
           markerfacecolor="gold", markeredgecolor="black",
           markersize=9, label=f"newly added ({', '.join(str(a) for a in NEW_IDS)})"))

ax.legend(handles=legend_elements, fontsize=10)
ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% variance)", fontsize=12)
status_103 = "added" if ANIMAL_103_STATUS == "OK" else "NEEDS_REVIEW"
ax.set_title(
    f"Baseline ECG PCA — {len(df)} animals "
    f"(102 removed; 103 {status_103}; 110 NEEDS_REVIEW; 115 flagged outlier)\n"
    f"13-feature matrix, StandardScaler, PCA(random_state=42)",
    fontsize=13)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "pca_corrected.png", dpi=150)
plt.close(fig)
print(f"  Saved: {FIG / 'pca_corrected.png'}")

print("\n" + "="*64)
print("DONE")
print("="*64)
print(f"  all_animals_features.csv : {len(df)} animals")
print(f"  Animal 102 : REMOVED (mains interference, supervisor confirmed)")
print(f"  Animal 103 : {ANIMAL_103_STATUS}")
print(f"  Animal 110 : NEEDS_REVIEW (not in CSV)")
print(f"  Cluster columns recomputed: k2, k5, hier_k2, hier_k5, gmm_k2, gmm_k5")
print(f"  Figure: outputs/figures/pca_corrected.png")
