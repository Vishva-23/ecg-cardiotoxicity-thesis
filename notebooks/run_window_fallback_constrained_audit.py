"""
Constrained window-fallback re-audit (more defensible variant).

Same uniform rule for all 116 animals; pipeline functions reused verbatim from
NB02 (exec-imported). Shared-PCA-basis design kept (fit once on annotated run,
apply identical transform to every run).

Three runs, ALL using the same widened gate so the ONLY difference between the
unconstrained and constrained fallback is the search radius (isolates the
constraint's effect on the PCA, which is the question being asked):
  A  annotated window only
  U  fallback, UNCONSTRAINED (whole recording)        -- comparison baseline
  C  fallback, CONSTRAINED to +/- SEARCH_RADIUS_S     -- the new defensible rule

Outputs (audit only, NOT committed):
  outputs/features_fallback_constrained.csv
  outputs/pca_before_after_constrained.png
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
# AUDITABLE CONSTANTS (same for every animal)
# ===========================================================================
WINDOW_WIDTH_S  = 30      # candidate window width
STEP_S          = 5       # sliding step
SEARCH_RADIUS_S = 120     # NEW: constrained search radius around annotated start
HR_GATE_LOW     = 350     # WIDENED gate (was 400) - mouse isoflurane resting range
HR_GATE_HIGH    = 700     # WIDENED gate (was 650)
RR_STD_CAP      = 120     # rhythm regularity cap (unchanged)
CLEAN_RR_STD    = 40      # NEW override flag: annotated window "clean" if rr_std <= this

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
a = pca_cell.index("cluster_feature_cols = ["); b = pca_cell.index("]", a) + 1
exec(compile(pca_cell[a:b], "cell_featcols", "exec"), P)
FEATCOLS = P["cluster_feature_cols"]

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

WIN    = int(WINDOW_WIDTH_S * FS)
STEP   = int(STEP_S * FS)
RADIUS = int(SEARCH_RADIUS_S * FS)
_integrate = getattr(np, "trapezoid", np.trapz)
print(f"Constants: window={WINDOW_WIDTH_S}s step={STEP_S}s radius=+/-{SEARCH_RADIUS_S}s "
      f"HR gate=[{HR_GATE_LOW},{HR_GATE_HIGH}] rr_std_cap={RR_STD_CAP} clean_rr_std<={CLEAN_RR_STD}\n")


def eval_window(segment):
    if segment.size < 2 * FS:
        return dict(hr=np.nan, rr_std=np.nan, n=0)
    filtered = bandpass_filter(segment)
    peaks, inverted, hr, _ = detect_r_peaks(filtered)
    rr_std = float(np.std(np.diff(peaks) * (1000.0 / FS), ddof=1)) if len(peaks) > 1 else np.nan
    return dict(hr=float(hr) if not np.isnan(hr) else np.nan, rr_std=rr_std, n=int(len(peaks)))


def passes_gate(ev):
    return (ev["n"] >= 10 and not np.isnan(ev["hr"]) and not np.isnan(ev["rr_std"])
            and HR_GATE_LOW <= ev["hr"] <= HR_GATE_HIGH and ev["rr_std"] <= RR_STD_CAP)


def mains_ratio(seg):
    f, pxx = welch(seg, fs=FS, nperseg=min(len(seg), 4096))
    def band(lo, hi):
        m = (f >= lo) & (f <= hi)
        return float(_integrate(pxx[m], f[m])) if m.any() else 0.0
    mains = band(49, 51) + band(99, 101) + band(149, 151)
    cardiac = band(4, 45)
    return mains / cardiac if cardiac > 0 else np.inf


def find_baseline_window_with_fallback(signal, annotations, search_radius_s=None):
    """search_radius_s=None -> unconstrained (whole recording);
       value -> constrained to starts within +/- radius of the annotated start."""
    n = len(signal)
    a_start, a_end, rule, s_text, e_text = find_baseline_window(annotations, n)
    ann_hr = ann_rrstd = np.nan
    if a_start is not None:
        ev = eval_window(signal[a_start:a_end])
        ann_hr, ann_rrstd = ev["hr"], ev["rr_std"]
        if passes_gate(ev):
            return dict(start=a_start, end=a_end, window_source="annotated",
                        reason="annotated passed gate", rule=rule, start_ann=s_text, end_ann=e_text,
                        annotated_HR=ann_hr, annotated_rr_std=ann_rrstd)

    # bounds for the slide
    if search_radius_s is None or a_start is None:
        lo, hi = 0, n - WIN
    else:
        rad = int(search_radius_s * FS)
        lo = max(0, a_start - rad)
        hi = min(n - WIN, a_start + rad)

    cands = []
    c = lo
    while c <= hi:
        ev = eval_window(signal[c:c + WIN])
        if passes_gate(ev):
            cands.append((ev["rr_std"], mains_ratio(signal[c:c + WIN]), c, c + WIN, ev["hr"]))
        c += STEP

    if cands:
        cands.sort(key=lambda t: (t[0], t[1]))
        rr_std, mr, cs, ce, chr_ = cands[0]
        return dict(start=cs, end=ce, window_source="fallback",
                    reason=f"annotated failed; slid -> rr_std={rr_std:.1f} HR={chr_:.0f}",
                    rule=rule, start_ann=s_text, end_ann=e_text,
                    annotated_HR=ann_hr, annotated_rr_std=ann_rrstd)
    return dict(start=a_start, end=a_end, window_source="none_passed",
                reason="no window in range passed the gate", rule=rule,
                start_ann=s_text, end_ann=e_text, annotated_HR=ann_hr, annotated_rr_std=ann_rrstd)


def classify(hr, rr_std, n):
    if n < 10:
        return "FAILED_PEAKS"
    if (not np.isnan(hr) and not np.isnan(rr_std)
            and HR_GATE_LOW <= hr <= HR_GATE_HIGH and rr_std <= RR_STD_CAP):
        return "OK"
    return "NEEDS_REVIEW"


def process(voltage, markers, aid, rec_date, fname, mode):
    """mode in {'annotated','unconstrained','constrained'}."""
    rec = {"animal_id": aid, "recording_date": rec_date.date() if rec_date is not None else None,
           "source_file": fname, "window_source": None, "window_start": None, "window_end": None,
           "rule_used": None, "n_beats": np.nan, "heart_rate_bpm": np.nan, "rr_mean_ms": np.nan,
           "rr_std_ms": np.nan, "r_amplitude_mv": np.nan, "qrs_duration_ms": np.nan,
           "j_wave_amplitude_mv": np.nan, "t_wave_amplitude_mv": np.nan, "rt_interval_ms": np.nan,
           "qt_ms": np.nan, "qtc_ms": np.nan, "qt_std_ms": np.nan, "qrs_std_ms": np.nan,
           "r_amplitude_std_mv": np.nan, "rr_cv": np.nan, "annotated_window_HR": np.nan,
           "annotated_window_rr_std": np.nan, "overrode_clean_annotation": False,
           "status": "", "reason": ""}
    if mode == "annotated":
        start, end, rule, s_text, e_text = find_baseline_window(markers, len(voltage))
        rec.update(window_source="annotated", rule_used=rule)
        if start is not None:
            ev = eval_window(voltage[start:end])
            rec["annotated_window_HR"], rec["annotated_window_rr_std"] = ev["hr"], ev["rr_std"]
    else:
        radius = None if mode == "unconstrained" else SEARCH_RADIUS_S
        fb = find_baseline_window_with_fallback(voltage, markers, search_radius_s=radius)
        start, end = fb["start"], fb["end"]
        rec.update(window_source=fb["window_source"], rule_used=fb["rule"], reason=fb["reason"],
                   annotated_window_HR=fb["annotated_HR"], annotated_window_rr_std=fb["annotated_rr_std"])

    if start is None:
        rec["status"], rec["reason"] = "FAILED", "no baseline window"
        return rec
    rec["window_start"], rec["window_end"] = int(start), int(end)
    segment = voltage[start:end]
    if segment.size < 2 * FS:
        rec["status"], rec["reason"] = "FAILED", "window too short"
        return rec

    filtered = bandpass_filter(segment)
    peaks, inverted, hr, _ = detect_r_peaks(filtered)
    if inverted:
        filtered = -filtered
    rec["n_beats"] = int(len(peaks))
    rec["heart_rate_bpm"] = round(float(hr), 1) if not np.isnan(hr) else np.nan
    if len(peaks) < 10:
        rec["status"] = "FAILED_PEAKS"
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

    # NEW override flag: a clean (rr_std<=40) annotated window was still replaced
    if mode != "annotated" and rec["window_source"] == "fallback":
        a_rr = rec["annotated_window_rr_std"]
        rec["overrode_clean_annotation"] = bool(not np.isnan(a_rr) and a_rr <= CLEAN_RR_STD)
    return rec


# ---------------------------------------------------------------------------
# Run all three modes (load each file once)
# ---------------------------------------------------------------------------
files = sorted(DATA_DIR.glob("*.txt"))
recsA, recsU, recsC = [], [], []
print(f"Processing {len(files)} files x 3 modes...")
for k, path in enumerate(files):
    aid, rec_date = parse_filename(path)
    if aid is None:
        continue
    try:
        n_header, lead_cols, ecg_col, titles = parse_header_and_layout(path)
        voltage, markers = load_ecg_file(path, n_header, ecg_col)
        if voltage.size == 0 or float(np.median(np.abs(voltage))) > 50:
            continue
        recsA.append(process(voltage, markers, aid, rec_date, path.name, "annotated"))
        recsU.append(process(voltage, markers, aid, rec_date, path.name, "unconstrained"))
        recsC.append(process(voltage, markers, aid, rec_date, path.name, "constrained"))
        del voltage
    except Exception as e:
        for L, ws in ((recsA, "annotated"), (recsU, "none_passed"), (recsC, "none_passed")):
            L.append({"animal_id": aid, "source_file": path.name, "status": "FAILED",
                      "reason": f"{type(e).__name__}: {e}", "window_source": ws,
                      "overrode_clean_annotation": False})
    if (k + 1) % 25 == 0:
        print(f"  ...{k + 1}/{len(files)}")

dfA = pd.DataFrame(recsA)
dfU = pd.DataFrame(recsU)
dfC = pd.DataFrame(recsC)
dfC.to_csv(OUTPUTS_DIR / "features_fallback_constrained.csv", index=False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SUMMARY — constrained re-audit (gate widened to [350,700], radius +/-120s)")
print("=" * 72)
print(f"Run A annotated   status: {dfA['status'].value_counts().to_dict()}")
print(f"Run U unconstr.   status: {dfU['status'].value_counts().to_dict()}")
print(f"Run C constrained status: {dfC['status'].value_counts().to_dict()}")
fired_U = int(dfU['window_source'].eq('fallback').sum())
fired_C = int(dfC['window_source'].eq('fallback').sum())
print(f"\nFallback fired:  unconstrained={fired_U}   constrained={fired_C}   "
      f"(constrained expected FEWER)")
print(f"No window passed: unconstrained={int(dfU['window_source'].eq('none_passed').sum())}   "
      f"constrained={int(dfC['window_source'].eq('none_passed').sum())}")

aS = dfA.set_index('animal_id')['status']; cS = dfC.set_index('animal_id')['status']
common = aS.index.intersection(cS.index)
to_ok_C = sorted([a for a in common if aS[a] != 'OK' and cS[a] == 'OK'])
print(f"\nConstrained moved -> OK: {len(to_ok_C)} animals {to_ok_C}")

# override list (new flag)
ovC = dfC[dfC['overrode_clean_annotation']]
print("\n--- OVERRIDE AUDIT (new flag: clean annotated rr_std<=40 still replaced) ---")
if ovC.empty:
    print("  (none) — the widened gate no longer rejects clean baselines like 139/149.")
else:
    for _, r in ovC.iterrows():
        print(f"  animal {int(r['animal_id'])}: annotated HR={r['annotated_window_HR']:.0f} "
              f"rr_std={r['annotated_window_rr_std']:.0f} -> fallback HR={r['heart_rate_bpm']:.0f} "
              f"rr_std={r['rr_std_ms']:.0f}")

# fired animals: annotated vs fallback HR side by side (state-swap check)
print("\n--- CONSTRAINED fired animals: annotated HR vs fallback HR (state-swap check) ---")
ff = dfC[dfC['window_source'] == 'fallback']
if ff.empty:
    print("  (none fired)")
else:
    print(f"{'animal':>6} {'ann_HR':>7} {'ann_rrstd':>9} {'fb_HR':>7} {'fb_rrstd':>8} {'HR_jump':>8}")
    for _, r in ff.iterrows():
        jump = r['heart_rate_bpm'] - r['annotated_window_HR']
        print(f"{int(r['animal_id']):>6} {r['annotated_window_HR']:>7.0f} "
              f"{r['annotated_window_rr_std']:>9.0f} {r['heart_rate_bpm']:>7.0f} "
              f"{r['rr_std_ms']:>8.0f} {jump:>+8.0f}")

# ---------------------------------------------------------------------------
# PCA — shared basis fit on annotated; project A, U, C
# ---------------------------------------------------------------------------
valid = [a for a in common
         if dfA.set_index('animal_id').loc[a, 'n_beats'] >= 10
         and dfU.set_index('animal_id').loc[a, 'n_beats'] >= 10
         and dfC.set_index('animal_id').loc[a, 'n_beats'] >= 10]
A = dfA.set_index('animal_id').loc[valid]
U = dfU.set_index('animal_id').loc[valid]
C = dfC.set_index('animal_id').loc[valid]
XA = A[FEATCOLS].astype(float); XU = U[FEATCOLS].astype(float); XC = C[FEATCOLS].astype(float)
med = XA.median()
XA, XU, XC = XA.fillna(med), XU.fillna(med), XC.fillna(med)

scaler = StandardScaler().fit(XA.values)
pca = PCA(n_components=2, random_state=42).fit(scaler.transform(XA.values))
cov = pca.explained_variance_ratio_[:2] * 100
cA = pca.transform(scaler.transform(XA.values))
cU = pca.transform(scaler.transform(XU.values))
cC = pca.transform(scaler.transform(XC.values))
ids = list(valid)
is_out = np.array([a in OUTLIERS for a in ids])
centroid = cA[~is_out].mean(axis=0)

# figure A vs C, shared axes
allc = np.vstack([cA, cC])
xpad = 0.05 * np.ptp(allc[:, 0]); ypad = 0.05 * np.ptp(allc[:, 1])
xlim = (allc[:, 0].min() - xpad, allc[:, 0].max() + xpad)
ylim = (allc[:, 1].min() - ypad, allc[:, 1].max() + ypad)
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
for ax, coords, ttl in [(axes[0], cA, "Annotated windows"),
                        (axes[1], cC, "With CONSTRAINED window fallback (+/-120s)")]:
    ax.scatter(coords[~is_out, 0], coords[~is_out, 1], s=28, c="steelblue", alpha=0.6, label="other animals")
    ax.scatter(coords[is_out, 0], coords[is_out, 1], s=130, c="crimson",
               edgecolor="black", linewidth=1.3, zorder=5, label="7 PCA outliers")
    for (x, y), aid, o in zip(coords, ids, is_out):
        if o:
            ax.text(x + 0.1, y + 0.1, str(aid), fontsize=9, fontweight="bold")
    ax.scatter(*centroid, marker="X", s=160, c="black", zorder=6, label="main-cluster centroid")
    ax.set_title(ttl, fontsize=13)
    ax.set_xlabel(f"PC1 ({cov[0]:.1f}% var, shared basis)")
    ax.set_ylabel(f"PC2 ({cov[1]:.1f}% var, shared basis)")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("PCA before vs after CONSTRAINED window fallback (shared basis fit on annotated)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGURES_DIR / "pca_before_after_constrained.png", dpi=150)
plt.close(fig)
print(f"\nSaved -> {FIGURES_DIR / 'pca_before_after_constrained.png'}")

# ---------------------------------------------------------------------------
# 7 outliers: movement under UNCONSTRAINED vs CONSTRAINED (the key comparison)
# ---------------------------------------------------------------------------
idx = {a: i for i, a in enumerate(ids)}
print("\n--- 7 OUTLIERS: did the PCA movement survive the constraint? ---")
print(f"{'animal':>6} {'A_HR':>6} {'C_HR':>6} {'d_annot':>8} {'d_uncon':>8} {'d_con':>7} "
      f"{'closer_U':>9} {'closer_C':>9}  C_source")
for aid in OUTLIERS:
    if aid not in idx:
        print(f"{aid:>6}  (excluded: <10 beats somewhere)"); continue
    i = idx[aid]
    dA = float(np.hypot(*(cA[i] - centroid)))
    dU = float(np.hypot(*(cU[i] - centroid)))
    dC = float(np.hypot(*(cC[i] - centroid)))
    a_hr = dfA.set_index('animal_id').loc[aid, 'heart_rate_bpm']
    c_hr = dfC.set_index('animal_id').loc[aid, 'heart_rate_bpm']
    csrc = dfC.set_index('animal_id').loc[aid, 'window_source']
    print(f"{aid:>6} {a_hr:>6.0f} {c_hr:>6.0f} {dA:>8.2f} {dU:>8.2f} {dC:>7.2f} "
          f"{dA - dU:>9.2f} {dA - dC:>9.2f}  {csrc}")
print("\ncloser_U/closer_C > 0 = moved toward cluster under unconstrained/constrained.")
print("If closer_C << closer_U, the unconstrained movement was a search artifact.")
