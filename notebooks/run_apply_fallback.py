"""
Apply confirmed-artefact fallback correction to 6 animals (107,112,115,117,126,235),
leave 102 pending, overwrite all_animals_features.csv, rerun PCA.

Reuses NB02 functions verbatim. Fixed params (FS=1000, BP 0.5-150 + notch,
REFRACTORY_MS=55, MIN_RR_MS=75, PRE/POST 100/150, Mitchell QTc). Nothing committed.
"""
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

NB = "02_beat_averaging_and_clustering.ipynb"
CORRECT = [107, 112, 115, 117, 126, 235]     # confirmed artefacts -> fallback_scan
PENDING = 102                                 # leave row unchanged -> pending_review
WINDOW_WIDTH_S, STEP_S = 30, 5
HR_LOW, HR_HIGH, RR_STD_CAP, STD_CAP, LF_CAP = 400, 650, 120, 0.5, 0.15
RS = 42
PCA_FEATURES = ["heart_rate_bpm", "rr_mean_ms", "rr_std_ms", "r_amplitude_mv",
                "qrs_duration_ms", "j_wave_amplitude_mv", "t_wave_amplitude_mv",
                "qt_ms", "qtc_ms", "qt_std_ms", "qrs_std_ms", "r_amplitude_std_mv",
                "rr_cv", "rt_interval_ms"]   # rt_interval_ms used only if present

# ---- reuse pipeline ----
nb = json.load(open(NB, encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
P = {"__name__": "__p__"}
exec(compile(next(s for s in codes if s.startswith("import re")), "setup", "exec"), P)
exec(compile(next(s for s in codes if "def extract_morphology_features" in s), "helpers", "exec"), P)
bc = next(s for s in codes if "def mitchell_qtc" in s)
a = bc.index("QTC_FORMULA_NAME ="); b = bc.index("per_animal", a)
exec(compile(bc[a:b], "mitchell", "exec"), P)
FS = P["FS"]; DATA_DIR = P["DATA_DIR"]; OUTPUTS_DIR = P["OUTPUTS_DIR"]; FIGURES_DIR = P["FIGURES_DIR"]
parse_filename = P["parse_filename"]; parse_header_and_layout = P["parse_header_and_layout"]
load_ecg_file = P["load_ecg_file"]; bandpass_filter = P["bandpass_filter"]
detect_r_peaks = P["detect_r_peaks"]; average_beats = P["average_beats"]
extract_morphology_features = P["extract_morphology_features"]; mitchell_qtc = P["mitchell_qtc"]
WIN, STEP = int(WINDOW_WIDTH_S * FS), int(STEP_S * FS)
_integrate = getattr(np, "trapezoid", np.trapz)

# map animal_id -> file
id2file = {}
for p in DATA_DIR.glob("*.txt"):
    aid, _ = parse_filename(p)
    if aid is not None:
        id2file[int(aid)] = p


def low_freq_ratio(filt):
    f, pxx = welch(filt, fs=FS, nperseg=min(len(filt), 4096))
    tot = _integrate(pxx, f)
    lo = _integrate(pxx[f < 3], f[f < 3]) if (f < 3).any() else 0.0
    return float(lo / tot) if tot > 0 else np.nan


def scan_cleanest(voltage):
    """Tile 30 s windows (5 s step) across the WHOLE recording. Among HR[400,650]
       & rr_std<=120 windows that are clean (std<0.5 mV, low-freq<0.15) pick the
       LOWEST rr_std. Relax the cleanliness gates only if none qualify."""
    n = len(voltage); strict, relaxed = [], []
    c = 0
    while c + WIN <= n:
        filt = bandpass_filter(voltage[c:c + WIN])
        pk, inv, hr, _ = detect_r_peaks(filt)
        if len(pk) >= 10 and not np.isnan(hr) and HR_LOW <= hr <= HR_HIGH:
            rr_std = float(np.std(np.diff(pk) * (1000.0 / FS), ddof=1))
            if rr_std <= RR_STD_CAP:
                sd = float(np.std(filt)); lf = low_freq_ratio(filt)
                rec = (rr_std, c, hr, sd, lf, inv)
                relaxed.append(rec)
                if sd < STD_CAP and lf < LF_CAP:
                    strict.append(rec)
        c += STEP
    pool = strict if strict else relaxed
    if not pool:
        return None
    best = min(pool, key=lambda r: r[0])     # lowest rr_std
    return dict(rr_std=best[0], start=best[1], hr=best[2], std=best[3],
                lf=best[4], inverted=best[5], strict=bool(strict))


def features_from_window(voltage, start):
    seg = voltage[start:start + WIN]
    filt = bandpass_filter(seg)
    pk, inv, hr, _ = detect_r_peaks(filt)
    if inv:
        filt = -filt
    rr = np.diff(pk) * (1000.0 / FS)
    rr_mean = float(np.mean(rr)); rr_std = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
    template, t_ms, beats = average_beats(filt, pk)
    f = extract_morphology_features(template, t_ms, beats)
    qtc = mitchell_qtc(f["qt_ms"], rr_mean)
    return dict(n_beats=int(len(pk)), heart_rate_bpm=round(float(hr), 4),
                rr_mean_ms=round(rr_mean, 4), rr_std_ms=round(rr_std, 4),
                r_amplitude_mv=round(f["r_amplitude_mv"], 6),
                qrs_duration_ms=round(f["qrs_duration_ms"], 4),
                j_wave_amplitude_mv=round(f["j_wave_amplitude_mv"], 6),
                t_wave_amplitude_mv=round(f["t_wave_amplitude_mv"], 6),
                qt_ms=round(f["qt_ms"], 4), qtc_ms=round(qtc, 4),
                qt_std_ms=round(f["qt_std_ms"], 4) if not np.isnan(f["qt_std_ms"]) else np.nan,
                qrs_std_ms=round(f["qrs_std_ms"], 4) if not np.isnan(f["qrs_std_ms"]) else np.nan,
                r_amplitude_std_mv=round(f["r_amplitude_std_mv"], 6) if not np.isnan(f["r_amplitude_std_mv"]) else np.nan,
                rr_cv=round(rr_std / rr_mean, 6) if rr_mean else np.nan,
                rt_interval_ms=round(f["rt_interval_ms"], 4) if not np.isnan(f["rt_interval_ms"]) else np.nan)


# =====================================================================
df = pd.read_csv(OUTPUTS_DIR / "all_animals_features.csv")
print(f"Loaded master CSV: {len(df)} rows, {len(df.columns)} cols")
shutil.copy(OUTPUTS_DIR / "all_animals_features.csv", OUTPUTS_DIR / "all_animals_features_preFallback.csv")
print("Backup -> all_animals_features_preFallback.csv")

feat_cols = ["n_beats", "heart_rate_bpm", "rr_mean_ms", "rr_std_ms", "r_amplitude_mv",
             "qrs_duration_ms", "j_wave_amplitude_mv", "t_wave_amplitude_mv", "qt_ms",
             "qtc_ms", "qt_std_ms", "qrs_std_ms", "r_amplitude_std_mv", "rr_cv"]

print("\n=== correcting 6 confirmed-artefact animals ===")
print(f"{'id':>4} {'old_HR':>7} {'new_HR':>7} {'old_rrstd':>9} {'new_rrstd':>9} "
      f"{'old_qtc':>7} {'new_qtc':>7} {'win@min':>8} {'gate':>7} {'inv':>4}")
for aid in CORRECT:
    if aid not in id2file:
        print(f"  {aid}: NO FILE — skipped"); continue
    nh, lc, ec, _t = parse_header_and_layout(id2file[aid])
    v, mk = load_ecg_file(id2file[aid], nh, ec)
    best = scan_cleanest(v)
    row_mask = df.animal_id == aid
    old = df.loc[row_mask, ["heart_rate_bpm", "rr_std_ms", "qtc_ms"]].iloc[0]
    if best is None:
        print(f"  {aid}: no HR[400,650] window found — left UNCHANGED"); continue
    newf = features_from_window(v, best["start"])
    for k, val in newf.items():
        if k in df.columns:
            df.loc[row_mask, k] = val
    print(f"{aid:>4} {old['heart_rate_bpm']:>7.0f} {newf['heart_rate_bpm']:>7.0f} "
          f"{old['rr_std_ms']:>9.1f} {newf['rr_std_ms']:>9.1f} {old['qtc_ms']:>7.1f} "
          f"{newf['qtc_ms']:>7.1f} {best['start']/FS/60:>8.1f} "
          f"{'strict' if best['strict'] else 'relax':>7} {str(best['inverted']):>4}")

# window_source column
def src(aid):
    if aid == PENDING:
        return "pending_review"
    if aid in CORRECT:
        return "fallback_scan"
    return "annotation"
df["window_source"] = df["animal_id"].apply(src)
print("\nwindow_source counts:", df["window_source"].value_counts().to_dict())

df.to_csv(OUTPUTS_DIR / "all_animals_features.csv", index=False)
print(f"Overwrote all_animals_features.csv ({len(df)} rows, {len(df.columns)} cols incl window_source)")

# =====================================================================
# PCA on the available features (rt_interval_ms only if present)
feats = [c for c in PCA_FEATURES if c in df.columns]
print(f"\nPCA features ({len(feats)}): {feats}")
X = df[feats].astype(float)
X = X.fillna(X.median())
Xz = StandardScaler().fit_transform(X.values)
pca = PCA(n_components=2, random_state=RS).fit(Xz)
sc = pca.transform(Xz); cov = pca.explained_variance_ratio_[:2] * 100

colors = {"annotation": "steelblue", "fallback_scan": "green", "pending_review": "red"}
sizes = {"annotation": 36, "fallback_scan": 130, "pending_review": 150}
fig, ax = plt.subplots(figsize=(15, 11))
for srcname, col in colors.items():
    m = (df["window_source"] == srcname).values
    if m.any():
        ax.scatter(sc[m, 0], sc[m, 1], s=sizes[srcname], c=col,
                   edgecolor="black" if srcname != "annotation" else "none",
                   linewidth=1.2, alpha=0.85, zorder=4 if srcname != "annotation" else 2,
                   label=f"{srcname} (n={m.sum()})")
for (x, y), aid, srcname in zip(sc, df.animal_id, df.window_source):
    bold = srcname != "annotation"
    ax.text(x + 0.1, y + 0.08, str(int(aid)),
            fontsize=9 if bold else 6, fontweight="bold" if bold else "normal",
            color="black" if bold else "dimgray")
ax.set_xlabel(f"PC1 ({cov[0]:.1f}% variance)"); ax.set_ylabel(f"PC2 ({cov[1]:.1f}% variance)")
ax.set_title("PCA after fallback correction of 6 artefact animals "
             "(102 pending) — animal IDs labelled", fontsize=14)
ax.grid(alpha=0.3); ax.legend(fontsize=10)
fig.tight_layout(); fig.savefig(FIGURES_DIR / "pca_corrected.png", dpi=150); plt.close(fig)
print(f"Saved pca_corrected.png (PC1 {cov[0]:.1f}%, PC2 {cov[1]:.1f}%)")

# show the 7 outliers' new PCA positions
idx = {int(a): i for i, a in enumerate(df.animal_id)}
print("\n7 outliers — PCA position now:")
for aid in CORRECT + [PENDING]:
    i = idx[aid]
    print(f"  {aid} ({df.window_source.iloc[i]}): PC1={sc[i,0]:.2f} PC2={sc[i,1]:.2f}")
