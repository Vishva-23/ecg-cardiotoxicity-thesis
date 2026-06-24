"""
(a) Re-pick 112 & 115 with HR capped at 580 bpm; if no window <580 with rr_std<=120
    exists, mark window_source='unresolvable' and revert that row to its original
    annotation values (do NOT keep a bad >580 window).
(b) Recompute all 6 cluster columns on the updated feature matrix (NB02 params:
    KMeans n_init=10 random_state=42, Ward linkage, GaussianMixture random_state=42).
Then regenerate pca_corrected.png (blue=normal, green=corrected, orange=102 pending,
red=unresolvable). Only all_animals_features.csv + the figure change. Nothing committed.
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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

NB = "02_beat_averaging_and_clustering.ipynb"
REPICK = [112, 115]
HR_LOW, HR_CAP, RR_STD_CAP, STD_CAP, LF_CAP = 400, 580, 120, 0.5, 0.15
WINDOW_WIDTH_S, STEP_S, RS = 30, 5, 42
CLUSTER_FEATS = ["heart_rate_bpm", "rr_mean_ms", "rr_std_ms", "r_amplitude_mv",
                 "qrs_duration_ms", "j_wave_amplitude_mv", "t_wave_amplitude_mv",
                 "qt_ms", "qtc_ms", "qt_std_ms", "qrs_std_ms", "r_amplitude_std_mv", "rr_cv"]
FEAT_COLS = ["n_beats"] + CLUSTER_FEATS

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
id2file = {int(parse_filename(p)[0]): p for p in DATA_DIR.glob("*.txt") if parse_filename(p)[0] is not None}


def low_freq_ratio(filt):
    f, pxx = welch(filt, fs=FS, nperseg=min(len(filt), 4096))
    tot = _integrate(pxx, f)
    return float(_integrate(pxx[f < 3], f[f < 3]) / tot) if tot > 0 and (f < 3).any() else np.nan


def scan_capped(voltage):
    """Windows with HR in [400,580] & rr_std<=120. Prefer clean (std<0.5, lf<0.15);
       pick LOWEST rr_std. Return None if no [400,580] & rr_std<=120 window exists."""
    n = len(voltage); strict, pool = [], []
    c = 0
    while c + WIN <= n:
        filt = bandpass_filter(voltage[c:c + WIN])
        pk, inv, hr, _ = detect_r_peaks(filt)
        if len(pk) >= 10 and not np.isnan(hr) and HR_LOW <= hr <= HR_CAP:
            rr_std = float(np.std(np.diff(pk) * (1000.0 / FS), ddof=1))
            if rr_std <= RR_STD_CAP:
                sd, lf = float(np.std(filt)), low_freq_ratio(filt)
                rec = (rr_std, c, hr, sd, lf, inv)
                pool.append(rec)
                if sd < STD_CAP and lf < LF_CAP:
                    strict.append(rec)
    # note: STEP increment
        c += STEP
    use = strict if strict else pool
    if not use:
        return None
    best = min(use, key=lambda r: r[0])
    return dict(rr_std=best[0], start=best[1], hr=best[2], std=best[3], lf=best[4],
                inverted=best[5], strict=bool(strict))


def features_from_window(voltage, start):
    filt = bandpass_filter(voltage[start:start + WIN])
    pk, inv, hr, _ = detect_r_peaks(filt)
    if inv:
        filt = -filt
    rr = np.diff(pk) * (1000.0 / FS)
    rr_mean = float(np.mean(rr)); rr_std = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
    template, t_ms, beats = average_beats(filt, pk)
    f = extract_morphology_features(template, t_ms, beats)
    qtc = mitchell_qtc(f["qt_ms"], rr_mean)
    g = lambda v, d=4: round(float(v), d) if not (isinstance(v, float) and np.isnan(v)) else np.nan
    return {"n_beats": int(len(pk)), "heart_rate_bpm": g(hr), "rr_mean_ms": g(rr_mean),
            "rr_std_ms": g(rr_std), "r_amplitude_mv": g(f["r_amplitude_mv"], 6),
            "qrs_duration_ms": g(f["qrs_duration_ms"]), "j_wave_amplitude_mv": g(f["j_wave_amplitude_mv"], 6),
            "t_wave_amplitude_mv": g(f["t_wave_amplitude_mv"], 6), "qt_ms": g(f["qt_ms"]),
            "qtc_ms": g(qtc), "qt_std_ms": g(f["qt_std_ms"]), "qrs_std_ms": g(f["qrs_std_ms"]),
            "r_amplitude_std_mv": g(f["r_amplitude_std_mv"], 6), "rr_cv": g(rr_std / rr_mean, 6) if rr_mean else np.nan}


# =====================================================================
df = pd.read_csv(OUTPUTS_DIR / "all_animals_features.csv")
backup = pd.read_csv(OUTPUTS_DIR / "all_animals_features_preFallback.csv").set_index("animal_id")
print(f"Loaded {len(df)} rows. Re-picking {REPICK} with HR cap {HR_CAP}.\n")

# ---- (a) re-pick 112 & 115 ----
for aid in REPICK:
    v, mk = (lambda p: (lambda hdr: load_ecg_file(p, hdr[0], hdr[2]))(parse_header_and_layout(p)))(id2file[aid])
    best = scan_capped(v)
    mask = df.animal_id == aid
    old_hr = df.loc[mask, "heart_rate_bpm"].iloc[0]
    if best is None:
        # unresolvable -> revert to original annotation values, flag it
        for col in FEAT_COLS:
            if col in df.columns and aid in backup.index:
                df.loc[mask, col] = backup.loc[aid, col]
        df.loc[mask, "window_source"] = "unresolvable"
        print(f"  {aid}: NO window <{HR_CAP} bpm with rr_std<=120 -> UNRESOLVABLE "
              f"(reverted to annotation values, flagged)")
    else:
        newf = features_from_window(v, best["start"])
        for col, val in newf.items():
            if col in df.columns:
                df.loc[mask, col] = val
        df.loc[mask, "window_source"] = "fallback_scan"
        print(f"  {aid}: re-picked @ {best['start']/FS/60:.1f} min  HR {old_hr:.0f} -> {newf['heart_rate_bpm']:.0f} "
              f"(rr_std {newf['rr_std_ms']:.1f}, qtc {newf['qtc_ms']:.1f}, "
              f"{'strict' if best['strict'] else 'relax'}, inv={best['inverted']})")

# ---- (b) recompute all 6 cluster columns on updated matrix ----
X = df[CLUSTER_FEATS].astype(float)
X = X.fillna(X.median())
Xz = StandardScaler().fit_transform(X.values)
df["cluster_k2"] = KMeans(2, n_init=10, random_state=RS).fit_predict(Xz)
df["cluster_k5"] = KMeans(5, n_init=10, random_state=RS).fit_predict(Xz)
df["cluster_hier_k2"] = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(Xz)
df["cluster_hier_k5"] = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(Xz)
df["cluster_gmm_k2"] = GaussianMixture(n_components=2, random_state=RS).fit_predict(Xz)
df["cluster_gmm_k5"] = GaussianMixture(n_components=5, random_state=RS).fit_predict(Xz)
print("\nRecomputed cluster columns (13-feature standardised matrix).")
print("  window_source counts:", df["window_source"].value_counts().to_dict())

df.to_csv(OUTPUTS_DIR / "all_animals_features.csv", index=False)
print(f"Overwrote all_animals_features.csv ({len(df)} rows, {len(df.columns)} cols)")

# ---- regenerate pca_corrected.png (4-colour) ----
pca = PCA(n_components=2, random_state=RS).fit(Xz)
sc = pca.transform(Xz); cov = pca.explained_variance_ratio_[:2] * 100
style = {"annotation": ("steelblue", 36, "normal"),
         "fallback_scan": ("green", 130, "corrected fallback"),
         "pending_review": ("orange", 150, "102 pending"),
         "unresolvable": ("red", 150, "unresolvable")}
fig, ax = plt.subplots(figsize=(15, 11))
for srcname, (col, sz, lbl) in style.items():
    m = (df["window_source"] == srcname).values
    if m.any():
        ax.scatter(sc[m, 0], sc[m, 1], s=sz, c=col,
                   edgecolor="black" if srcname != "annotation" else "none",
                   linewidth=1.2, alpha=0.85, zorder=4 if srcname != "annotation" else 2,
                   label=f"{lbl} (n={m.sum()})")
for (x, y), aid, srcname in zip(sc, df.animal_id, df.window_source):
    bold = srcname != "annotation"
    ax.text(x + 0.1, y + 0.08, str(int(aid)), fontsize=9 if bold else 6,
            fontweight="bold" if bold else "normal", color="black" if bold else "dimgray")
ax.set_xlabel(f"PC1 ({cov[0]:.1f}% variance)"); ax.set_ylabel(f"PC2 ({cov[1]:.1f}% variance)")
ax.set_title("PCA after fallback correction + cluster recompute — animal IDs labelled", fontsize=14)
ax.grid(alpha=0.3); ax.legend(fontsize=11)
fig.tight_layout(); fig.savefig(FIGURES_DIR / "pca_corrected.png", dpi=150); plt.close(fig)
print(f"Saved pca_corrected.png (PC1 {cov[0]:.1f}%, PC2 {cov[1]:.1f}%)")

idx = {int(a): i for i, a in enumerate(df.animal_id)}
print("\nOutlier positions now:")
for aid in [107, 112, 115, 117, 126, 235, 102]:
    i = idx[aid]
    print(f"  {aid} ({df.window_source.iloc[i]:>14}): PC1={sc[i,0]:6.2f} PC2={sc[i,1]:6.2f}")
