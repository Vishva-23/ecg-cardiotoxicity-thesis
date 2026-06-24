"""
Window-fallback audit (defensible, no cherry-picking).

ONE rule, IDENTICAL for every one of the 116 animals, with named constants Luke
can audit. Reuses the committed Notebook-02 pipeline functions verbatim
(bandpass_filter, detect_r_peaks, average_beats, extract_morphology_features,
find_baseline_window, mitchell_qtc, loader) — extracted by exec-ing the actual
definition cells; nothing is rewritten.

Outputs (audit only, not committed):
  outputs/features_annotated.csv      Run A: annotated window only
  outputs/features_fallback.csv       Run B: annotated + automatic window fallback
  outputs/pca_before_after.png        side-by-side PCA on a SHARED projection basis
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

NB = "02_beat_averaging_and_clustering.ipynb"
OUTLIERS = [102, 107, 112, 115, 117, 126, 235]

# ===========================================================================
# NEW, AUDITABLE CONSTANTS (the ONLY new knobs; same for every animal)
# ===========================================================================
WINDOW_WIDTH_S = 30     # sliding candidate window width
STEP_S         = 5      # sliding step
HR_GATE_LOW    = 400    # acceptance gate: heart rate lower bound (bpm)
HR_GATE_HIGH   = 650    # acceptance gate: heart rate upper bound (bpm)
RR_STD_CAP     = 120    # acceptance gate: max rr_std (ms) = rhythm regularity
# (committed pipeline params BANDPASS_HIGH_HZ=150, REFRACTORY_MS=55, MIN_RR_MS=75,
#  median+4*MAD threshold, Mitchell QTc are reused untouched.)

# ---------------------------------------------------------------------------
# Reuse the committed pipeline (exec the notebook's own definition cells)
# ---------------------------------------------------------------------------
nb = json.load(open(NB, encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
setup_cell  = next(s for s in codes if s.startswith("import re"))
helper_cell = next(s for s in codes if "def extract_morphology_features" in s)
batch_cell  = next(s for s in codes if "def mitchell_qtc" in s)
pca_cell    = next(s for s in codes if "cluster_feature_cols = [" in s)

P = {"__name__": "__pipeline__"}
exec(compile(setup_cell, "cell_setup", "exec"), P)
exec(compile(helper_cell, "cell_helpers", "exec"), P)
a = batch_cell.index("QTC_FORMULA_NAME ="); b = batch_cell.index("per_animal", a)
exec(compile(batch_cell[a:b], "cell_mitchell", "exec"), P)
# the EXACT feature columns the existing PCA uses
a = pca_cell.index("cluster_feature_cols = ["); b = pca_cell.index("]", a) + 1
exec(compile(pca_cell[a:b], "cell_featcols", "exec"), P)
CLUSTER_FEATURE_COLS = P["cluster_feature_cols"]

FS          = P["FS"]
DATA_DIR    = P["DATA_DIR"]
OUTPUTS_DIR = P["OUTPUTS_DIR"]
FIGURES_DIR = P["FIGURES_DIR"]
parse_filename          = P["parse_filename"]
parse_header_and_layout = P["parse_header_and_layout"]
load_ecg_file           = P["load_ecg_file"]
find_baseline_window    = P["find_baseline_window"]
bandpass_filter         = P["bandpass_filter"]
detect_r_peaks          = P["detect_r_peaks"]
average_beats           = P["average_beats"]
extract_morphology_features = P["extract_morphology_features"]
mitchell_qtc            = P["mitchell_qtc"]

WIN  = int(WINDOW_WIDTH_S * FS)
STEP = int(STEP_S * FS)
print(f"Reused pipeline (BANDPASS_HIGH_HZ={P['BANDPASS_HIGH_HZ']}, REFRACTORY_MS={P['REFRACTORY_MS']}, "
      f"MIN_RR_MS={P['MIN_RR_MS']}, QTc={P['QTC_FORMULA_NAME']})")
print(f"Fallback constants: window={WINDOW_WIDTH_S}s step={STEP_S}s "
      f"HR gate=[{HR_GATE_LOW},{HR_GATE_HIGH}] rr_std_cap={RR_STD_CAP}")
print(f"PCA feature columns ({len(CLUSTER_FEATURE_COLS)}): {CLUSTER_FEATURE_COLS}\n")


# ---------------------------------------------------------------------------
# Helpers built ON TOP of the pipeline (not rewriting it)
# ---------------------------------------------------------------------------
def eval_window(segment):
    """Run a window through the REAL pipeline filter+detector; return HR/rr_std."""
    if segment.size < 2 * FS:
        return dict(hr=np.nan, rr_std=np.nan, n=0, inverted=False)
    filtered = bandpass_filter(segment)
    peaks, inverted, hr, _ = detect_r_peaks(filtered)
    if len(peaks) > 1:
        rr = np.diff(peaks) * (1000.0 / FS)
        rr_std = float(np.std(rr, ddof=1))
    else:
        rr_std = np.nan
    return dict(hr=float(hr) if not np.isnan(hr) else np.nan,
                rr_std=rr_std, n=int(len(peaks)), inverted=bool(inverted))


def passes_gate(ev):
    return (ev["n"] >= 10 and not np.isnan(ev["hr"]) and not np.isnan(ev["rr_std"])
            and HR_GATE_LOW <= ev["hr"] <= HR_GATE_HIGH and ev["rr_std"] <= RR_STD_CAP)


def mains_ratio(raw_segment):
    """Tie-break metric: mains (50/100/150 Hz) power / cardiac (4-45 Hz) power on
       the RAW window (lower = cleaner). Tie-breaks equal-rr_std candidates."""
    f, pxx = welch(raw_segment, fs=FS, nperseg=min(len(raw_segment), 4096))
    _integrate = getattr(np, "trapezoid", np.trapz)   # version-safe (numpy 1.x/2.x)
    def band(lo, hi):
        m = (f >= lo) & (f <= hi)
        return float(_integrate(pxx[m], f[m])) if m.any() else 0.0
    mains = band(49, 51) + band(99, 101) + band(149, 151)
    cardiac = band(4, 45)
    return mains / cardiac if cardiac > 0 else np.inf


def find_baseline_window_with_fallback(signal, annotations):
    """IDENTICAL rule for every animal. Returns a dict describing the chosen window."""
    n = len(signal)
    a_start, a_end, rule, s_text, e_text = find_baseline_window(annotations, n)

    ann_hr = ann_rrstd = np.nan
    if a_start is not None:
        ev = eval_window(signal[a_start:a_end])
        ann_hr, ann_rrstd = ev["hr"], ev["rr_std"]
        if passes_gate(ev):
            return dict(start=a_start, end=a_end, window_source="annotated",
                        reason="annotated window passed gate", rule=rule,
                        start_ann=s_text, end_ann=e_text,
                        annotated_HR=ann_hr, annotated_rr_std=ann_rrstd)

    # annotated failed (or missing): slide 30 s / 5 s over the WHOLE recording
    cands = []
    c = 0
    while c + WIN <= n:
        ev = eval_window(signal[c:c + WIN])
        if passes_gate(ev):
            cands.append((ev["rr_std"], mains_ratio(signal[c:c + WIN]), c, c + WIN, ev["hr"]))
        c += STEP

    if cands:
        cands.sort(key=lambda t: (t[0], t[1]))      # lowest rr_std, tie-break mains ratio
        rr_std, mr, cs, ce, chr_ = cands[0]
        return dict(start=cs, end=ce, window_source="fallback",
                    reason=f"annotated failed gate; slid -> rr_std={rr_std:.1f} ms, HR={chr_:.0f}",
                    rule=rule, start_ann=s_text, end_ann=e_text,
                    annotated_HR=ann_hr, annotated_rr_std=ann_rrstd)

    # nothing passed anywhere: keep annotated, do NOT fabricate
    return dict(start=a_start, end=a_end, window_source="none_passed",
                reason="no 30 s window anywhere passed the gate", rule=rule,
                start_ann=s_text, end_ann=e_text,
                annotated_HR=ann_hr, annotated_rr_std=ann_rrstd)


def classify(hr, rr_std, n):
    if n < 10:
        return "FAILED_PEAKS"
    if (not np.isnan(hr) and not np.isnan(rr_std)
            and HR_GATE_LOW <= hr <= HR_GATE_HIGH and rr_std <= RR_STD_CAP):
        return "OK"
    return "NEEDS_REVIEW"


def process(voltage, markers, animal_id, rec_date, fname, use_fallback):
    """Full pipeline on one animal, choosing the window per the requested mode."""
    rec = {"animal_id": animal_id, "recording_date": rec_date.date() if rec_date is not None else None,
           "source_file": fname, "window_source": None, "window_start": None, "window_end": None,
           "rule_used": None, "start_annotation": None, "end_annotation": None,
           "n_beats": np.nan, "heart_rate_bpm": np.nan, "rr_mean_ms": np.nan, "rr_std_ms": np.nan,
           "r_amplitude_mv": np.nan, "qrs_duration_ms": np.nan, "j_wave_amplitude_mv": np.nan,
           "t_wave_amplitude_mv": np.nan, "rt_interval_ms": np.nan, "qt_ms": np.nan, "qtc_ms": np.nan,
           "qt_std_ms": np.nan, "qrs_std_ms": np.nan, "r_amplitude_std_mv": np.nan, "rr_cv": np.nan,
           "annotated_window_HR": np.nan, "annotated_window_rr_std": np.nan,
           "overrode_inrange_annotation": False, "status": "", "reason": ""}

    if use_fallback:
        fb = find_baseline_window_with_fallback(voltage, markers)
        start, end = fb["start"], fb["end"]
        rec.update(window_source=fb["window_source"], rule_used=fb["rule"],
                   start_annotation=fb["start_ann"], end_annotation=fb["end_ann"],
                   annotated_window_HR=fb["annotated_HR"], annotated_window_rr_std=fb["annotated_rr_std"],
                   reason=fb["reason"])
    else:
        start, end, rule, s_text, e_text = find_baseline_window(markers, len(voltage))
        rec.update(window_source="annotated", rule_used=rule,
                   start_annotation=s_text, end_annotation=e_text)
        if start is not None:
            ev = eval_window(voltage[start:end])
            rec["annotated_window_HR"] = ev["hr"]
            rec["annotated_window_rr_std"] = ev["rr_std"]

    if start is None:
        rec["status"], rec["reason"] = "FAILED", "no baseline window at all"
        return rec
    rec["window_start"], rec["window_end"] = int(start), int(end)
    segment = voltage[start:end]
    if segment.size < 2 * FS:
        rec["status"], rec["reason"] = "FAILED", f"window too short ({segment.size} samples)"
        return rec

    filtered = bandpass_filter(segment)
    peaks, inverted, hr, _ = detect_r_peaks(filtered)
    if inverted:
        filtered = -filtered
    rec["n_beats"] = int(len(peaks))
    rec["heart_rate_bpm"] = round(float(hr), 1) if not np.isnan(hr) else np.nan

    if len(peaks) < 10:
        rec["status"] = "FAILED_PEAKS"
        if not rec["reason"]:
            rec["reason"] = f"only {len(peaks)} R peaks"
        return rec

    rr = np.diff(peaks) * (1000.0 / FS)
    rr_mean = float(np.mean(rr)); rr_std = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
    template, t_ms, beats = average_beats(filtered, peaks)
    feats = extract_morphology_features(template, t_ms, beats)
    qtc = mitchell_qtc(feats["qt_ms"], rr_mean)

    rec.update(rr_mean_ms=round(rr_mean, 2), rr_std_ms=round(rr_std, 2),
               r_amplitude_mv=round(feats["r_amplitude_mv"], 4),
               qrs_duration_ms=round(feats["qrs_duration_ms"], 2),
               j_wave_amplitude_mv=round(feats["j_wave_amplitude_mv"], 4),
               t_wave_amplitude_mv=round(feats["t_wave_amplitude_mv"], 4),
               rt_interval_ms=round(feats["rt_interval_ms"], 2) if not np.isnan(feats["rt_interval_ms"]) else np.nan,
               qt_ms=round(feats["qt_ms"], 2) if not np.isnan(feats["qt_ms"]) else np.nan,
               qtc_ms=round(qtc, 2) if not np.isnan(qtc) else np.nan,
               qt_std_ms=feats["qt_std_ms"], qrs_std_ms=feats["qrs_std_ms"],
               r_amplitude_std_mv=feats["r_amplitude_std_mv"],
               rr_cv=round(rr_std / rr_mean, 4) if rr_mean else np.nan)

    rec["status"] = classify(hr, rr_std, len(peaks))
    if rec["status"] == "OK" and not rec["reason"]:
        rec["reason"] = "passed gate"

    # override audit: fallback replaced an annotation whose HR was already in-range
    if use_fallback and rec["window_source"] == "fallback":
        a_hr = rec["annotated_window_HR"]
        rec["overrode_inrange_annotation"] = bool(
            not np.isnan(a_hr) and HR_GATE_LOW <= a_hr <= HR_GATE_HIGH)
    return rec


# ---------------------------------------------------------------------------
# PART 3 — run BOTH ways on all 116 animals (load each file once)
# ---------------------------------------------------------------------------
files = sorted(DATA_DIR.glob("*.txt"))
recsA, recsB = [], []
print(f"Processing {len(files)} files (Run A annotated, Run B fallback)...")
for k, path in enumerate(files):
    aid, rec_date = parse_filename(path)
    if aid is None:
        continue
    try:
        n_header, lead_cols, ecg_col, titles = parse_header_and_layout(path)
        voltage, markers = load_ecg_file(path, n_header, ecg_col)
        if voltage.size == 0 or float(np.median(np.abs(voltage))) > 50:
            continue
        recsA.append(process(voltage, markers, aid, rec_date, path.name, use_fallback=False))
        recsB.append(process(voltage, markers, aid, rec_date, path.name, use_fallback=True))
        del voltage
    except Exception as e:
        recsA.append({"animal_id": aid, "source_file": path.name, "status": "FAILED",
                      "reason": f"{type(e).__name__}: {e}", "window_source": "annotated",
                      "overrode_inrange_annotation": False})
        recsB.append({"animal_id": aid, "source_file": path.name, "status": "FAILED",
                      "reason": f"{type(e).__name__}: {e}", "window_source": "none_passed",
                      "overrode_inrange_annotation": False})
    if (k + 1) % 25 == 0:
        print(f"  ...{k + 1}/{len(files)}")

cols = ["animal_id", "recording_date", "source_file", "window_source", "window_start", "window_end",
        "rule_used", "start_annotation", "end_annotation", "n_beats", "heart_rate_bpm",
        "rr_mean_ms", "rr_std_ms", "r_amplitude_mv", "qrs_duration_ms", "j_wave_amplitude_mv",
        "t_wave_amplitude_mv", "rt_interval_ms", "qt_ms", "qtc_ms", "qt_std_ms", "qrs_std_ms",
        "r_amplitude_std_mv", "rr_cv", "annotated_window_HR", "annotated_window_rr_std",
        "overrode_inrange_annotation", "status", "reason"]
dfA = pd.DataFrame(recsA).reindex(columns=cols)
dfB = pd.DataFrame(recsB).reindex(columns=cols)
dfA.to_csv(OUTPUTS_DIR / "features_annotated.csv", index=False)
dfB.to_csv(OUTPUTS_DIR / "features_fallback.csv", index=False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def status_counts(df):
    return df["status"].value_counts().to_dict()

fired = dfB["window_source"].eq("fallback").sum()
nopass = dfB["window_source"].eq("none_passed").sum()
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Run A (annotated) status: {status_counts(dfA)}")
print(f"Run B (fallback)  status: {status_counts(dfB)}")
print(f"Fallback fired on        : {fired} animals")
print(f"No window passed anywhere : {nopass} animals (kept annotated, FAILED-style)")

# moved from NEEDS_REVIEW/FAILED* -> OK
a_status = dfA.set_index("animal_id")["status"]
b_status = dfB.set_index("animal_id")["status"]
common = a_status.index.intersection(b_status.index)
moved_to_ok = [aid for aid in common
               if a_status[aid] in ("NEEDS_REVIEW", "FAILED_PEAKS", "FAILED")
               and b_status[aid] == "OK"]
moved_from_ok = [aid for aid in common
                 if a_status[aid] == "OK" and b_status[aid] != "OK"]
print(f"\nMoved NEEDS_REVIEW/FAILED -> OK with fallback: {len(moved_to_ok)} animals {sorted(moved_to_ok)}")
print(f"Moved OK -> not OK (regressions)              : {len(moved_from_ok)} animals {sorted(moved_from_ok)}")

# PART 2 override list
override = dfB[dfB["overrode_inrange_annotation"]].copy()
print("\n--- PART 2: OVERRIDE AUDIT (send to Roisin) ---")
print("Animals where an in-range annotated window was REPLACED (HR was 400-650 but rr_std failed):")
if override.empty:
    print("  (none)")
else:
    for _, r in override.iterrows():
        print(f"  animal {int(r['animal_id'])}: annotated HR={r['annotated_window_HR']:.0f} "
              f"rr_std={r['annotated_window_rr_std']:.0f}ms -> fallback HR={r['heart_rate_bpm']:.0f} "
              f"rr_std={r['rr_std_ms']:.0f}ms")

# 7 outliers HR before vs after
print("\n--- 7 OUTLIERS: HR before (annotated) vs after (fallback) ---")
for aid in OUTLIERS:
    ra = dfA[dfA.animal_id == aid]
    rb = dfB[dfB.animal_id == aid]
    if ra.empty or rb.empty:
        print(f"  animal {aid}: missing"); continue
    ra, rb = ra.iloc[0], rb.iloc[0]
    print(f"  animal {aid}: A HR={ra['heart_rate_bpm']} ({ra['status']})  ->  "
          f"B HR={rb['heart_rate_bpm']} ({rb['status']}, {rb['window_source']})")

# ---------------------------------------------------------------------------
# PART 4 — PCA before/after on a SHARED projection basis (fit on Run A)
# ---------------------------------------------------------------------------
# animals with valid features in BOTH runs (need >=10 beats both times)
valid = [aid for aid in common
         if dfA.set_index("animal_id").loc[aid, "n_beats"] >= 10
         and dfB.set_index("animal_id").loc[aid, "n_beats"] >= 10]
A = dfA[dfA.animal_id.isin(valid)].set_index("animal_id").loc[valid]
B = dfB[dfB.animal_id.isin(valid)].set_index("animal_id").loc[valid]

XA = A[CLUSTER_FEATURE_COLS].astype(float)
XB = B[CLUSTER_FEATURE_COLS].astype(float)
med = XA.median()                      # impute with Run A medians (same for both)
XA = XA.fillna(med); XB = XB.fillna(med)

scaler = StandardScaler().fit(XA.values)          # fit on "before"
pca = PCA(n_components=2, random_state=42).fit(scaler.transform(XA.values))
covA = pca.explained_variance_ratio_[:2] * 100
coordsA = pca.transform(scaler.transform(XA.values))     # before
coordsB = pca.transform(scaler.transform(XB.values))     # after (SAME basis)

ids = list(valid)
is_out = np.array([aid in OUTLIERS for aid in ids])
# "cluster centroid" = centroid of the NON-outlier main cluster in the before frame
centroid = coordsA[~is_out].mean(axis=0)

# figure: side by side, shared axes
allc = np.vstack([coordsA, coordsB])
xpad = 0.05 * (allc[:, 0].max() - allc[:, 0].min())
ypad = 0.05 * (allc[:, 1].max() - allc[:, 1].min())
xlim = (allc[:, 0].min() - xpad, allc[:, 0].max() + xpad)
ylim = (allc[:, 1].min() - ypad, allc[:, 1].max() + ypad)

fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
for ax, coords, ttl in [(axes[0], coordsA, "Annotated windows"),
                        (axes[1], coordsB, "With window fallback")]:
    ax.scatter(coords[~is_out, 0], coords[~is_out, 1], s=28, c="steelblue", alpha=0.6, label="other animals")
    ax.scatter(coords[is_out, 0], coords[is_out, 1], s=130, c="crimson",
               edgecolor="black", linewidth=1.3, zorder=5, label="7 PCA outliers")
    for (x, y), aid, o in zip(coords, ids, is_out):
        if o:
            ax.text(x + 0.1, y + 0.1, str(aid), fontsize=9, fontweight="bold")
    ax.scatter(*centroid, marker="X", s=160, c="black", zorder=6, label="main-cluster centroid")
    ax.set_title(ttl, fontsize=13)
    ax.set_xlabel(f"PC1 ({covA[0]:.1f}% var, shared basis)")
    ax.set_ylabel(f"PC2 ({covA[1]:.1f}% var, shared basis)")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
fig.suptitle("PCA of baseline ECG features — before vs after automatic window fallback "
             "(shared projection basis fit on 'before')", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGURES_DIR / "pca_before_after.png", dpi=150)
plt.close(fig)
print(f"\nSaved PCA figure -> {FIGURES_DIR / 'pca_before_after.png'}")

# outlier movement table
print("\n--- 7 OUTLIERS: PCA position + movement toward main-cluster centroid ---")
print(f"{'animal':>6} {'PC1_bef':>8} {'PC2_bef':>8} {'PC1_aft':>8} {'PC2_aft':>8} "
      f"{'d_bef':>7} {'d_aft':>7} {'closer_by':>9}  source_B")
idx = {aid: i for i, aid in enumerate(ids)}
for aid in OUTLIERS:
    if aid not in idx:
        print(f"{aid:>6}  (excluded: <10 beats in one run)"); continue
    i = idx[aid]
    pb, pa = coordsA[i], coordsB[i]
    db = float(np.hypot(*(pb - centroid))); da = float(np.hypot(*(pa - centroid)))
    src = dfB[dfB.animal_id == aid].iloc[0]["window_source"]
    print(f"{aid:>6} {pb[0]:>8.2f} {pb[1]:>8.2f} {pa[0]:>8.2f} {pa[1]:>8.2f} "
          f"{db:>7.2f} {da:>7.2f} {db - da:>9.2f}  {src}")
print("\n(closer_by > 0 means the animal moved TOWARD the main cluster; < 0 means it moved away.)")
