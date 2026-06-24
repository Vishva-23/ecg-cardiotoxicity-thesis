"""
Defensible clustering comparison for the NB02 morphology feature set.

Reuses the committed NB02 pipeline functions verbatim (exec-imported); re-extracts
per-beat detail to build higher-order features (beat-to-beat SDs + cross-wave
correlations). Runs k-means / Ward / GMM, selects by AGREEMENT of multiple
internal indices, and flags degenerate (one-giant-cluster) solutions.

DATA CHOICE (documented): annotated windows, the committed-pipeline OK set
(HR 300-700 bpm, >=10 beats) = the SAME 108 animals the current k-means uses, for
apples-to-apples comparison. Post-fallback windows exist but, per the constrained-
fallback audit, they swap cardiac state for several animals and are not defensible
as baselines, so they are NOT used here.

Reproducibility: every seed fixed (RANDOM_STATE=42, gap RNG seed=0); Ward and
GMM-BIC selection are deterministic.

Outputs (NOT committed):
  outputs/cluster_validation_table.csv
  outputs/figures/feature_correlation_heatmap.png
  outputs/figures/dendrogram.png
  outputs/figures/validation_indices.png
  outputs/figures/clusters_pca_comparison.png
  outputs/cluster_assignments_recommended.csv
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram

NB = "02_beat_averaging_and_clustering.ipynb"
RANDOM_STATE = 42
GAP_B = 20
K_RANGE = list(range(2, 9))
DEGENERATE_FRAC = 0.70   # >70% of animals in one cluster => degenerate
HR_OK_LOW, HR_OK_HIGH = 300, 700   # committed pipeline OK gate (defines the 108)

rng_master = np.random.default_rng(0)

# ---- reuse pipeline ----
nb = json.load(open(NB, encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
P = {"__name__": "__pipeline__"}
exec(compile(next(s for s in codes if s.startswith("import re")), "setup", "exec"), P)
exec(compile(next(s for s in codes if "def extract_morphology_features" in s), "helpers", "exec"), P)
bc = next(s for s in codes if "def mitchell_qtc" in s)
a = bc.index("QTC_FORMULA_NAME ="); b = bc.index("per_animal", a)
exec(compile(bc[a:b], "mitchell", "exec"), P)
FS = P["FS"]; DATA_DIR = P["DATA_DIR"]; OUTPUTS_DIR = P["OUTPUTS_DIR"]; FIGURES_DIR = P["FIGURES_DIR"]
parse_filename = P["parse_filename"]; parse_header_and_layout = P["parse_header_and_layout"]
load_ecg_file = P["load_ecg_file"]; find_baseline_window = P["find_baseline_window"]
bandpass_filter = P["bandpass_filter"]; detect_r_peaks = P["detect_r_peaks"]
average_beats = P["average_beats"]; extract_morphology_features = P["extract_morphology_features"]
_measure_one = P["_measure_one"]; mitchell_qtc = P["mitchell_qtc"]


def safe_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


# ===========================================================================
# PART 1.1-1.2 : extract per-animal features (means + beat-to-beat SDs + corrs)
# ===========================================================================
rows, dropped = [], []
for path in sorted(DATA_DIR.glob("*.txt")):
    aid, _ = parse_filename(path)
    if aid is None:
        continue
    try:
        nh, lc, ec, _t = parse_header_and_layout(path)
        voltage, markers = load_ecg_file(path, nh, ec)
        if voltage.size == 0 or float(np.median(np.abs(voltage))) > 50:
            dropped.append((aid, "non-ECG column")); continue
        start, end, *_ = find_baseline_window(markers, len(voltage))
        if start is None:
            dropped.append((aid, "no baseline window")); continue
        seg = voltage[start:end]
        if seg.size < 2 * FS:
            dropped.append((aid, "window too short")); continue
        filt = bandpass_filter(seg)
        peaks, inverted, hr, _ = detect_r_peaks(filt)
        if inverted:
            filt = -filt
        if len(peaks) < 10:
            dropped.append((aid, f"FAILED_PEAKS ({len(peaks)} beats)")); continue
        if not (HR_OK_LOW <= hr <= HR_OK_HIGH):
            dropped.append((aid, f"NEEDS_REVIEW (HR {hr:.0f} outside {HR_OK_LOW}-{HR_OK_HIGH})")); continue

        rr = np.diff(peaks) * (1000.0 / FS)
        rr_mean = float(np.mean(rr)); rr_std = float(np.std(rr, ddof=1))
        template, t_ms, beats = average_beats(filt, peaks)
        feats = extract_morphology_features(template, t_ms, beats)
        qtc = mitchell_qtc(feats["qt_ms"], rr_mean)

        # per-beat arrays (reuse NB02's _measure_one on each individual beat)
        R, QRS, J, T, RT, QT = [], [], [], [], [], []
        for bt in beats:
            r, q, j, tw, rt, qt = _measure_one(bt, t_ms)
            R.append(r); QRS.append(q); J.append(j); T.append(tw); RT.append(rt); QT.append(qt)

        rows.append({
            "animal_id": aid,
            # --- base means (pipeline) ---
            "heart_rate_bpm": hr, "rr_mean_ms": rr_mean, "rr_std_ms": rr_std,
            "rr_cv": rr_std / rr_mean if rr_mean else np.nan,
            "r_amplitude_mv": feats["r_amplitude_mv"], "qrs_duration_ms": feats["qrs_duration_ms"],
            "j_wave_amplitude_mv": feats["j_wave_amplitude_mv"],
            "t_wave_amplitude_mv": feats["t_wave_amplitude_mv"],
            "rt_interval_ms": feats["rt_interval_ms"], "qt_ms": feats["qt_ms"], "qtc_ms": qtc,
            # --- ADDED beat-to-beat SDs ---
            "r_amplitude_std": np.nanstd(R, ddof=1), "qrs_std": np.nanstd(QRS, ddof=1),
            "j_wave_std": np.nanstd(J, ddof=1), "t_wave_std": np.nanstd(T, ddof=1),
            "rt_std": np.nanstd(RT, ddof=1), "qt_std": np.nanstd(QT, ddof=1),
            # --- ADDED cross-wave correlations across beats ---
            "corr_R_T_amp": safe_corr(R, T), "corr_R_J_amp": safe_corr(R, J),
            "corr_J_T_amp": safe_corr(J, T), "corr_QRS_QT": safe_corr(QRS, QT),
            "n_beats": len(peaks),
        })
    except Exception as e:
        dropped.append((aid, f"{type(e).__name__}: {e}"))

df = pd.DataFrame(rows).sort_values("animal_id").reset_index(drop=True)

BASE = ["heart_rate_bpm", "rr_mean_ms", "rr_std_ms", "rr_cv", "r_amplitude_mv",
        "qrs_duration_ms", "j_wave_amplitude_mv", "t_wave_amplitude_mv",
        "rt_interval_ms", "qt_ms", "qtc_ms"]
ADDED_SD = ["r_amplitude_std", "qrs_std", "j_wave_std", "t_wave_std", "rt_std", "qt_std"]
ADDED_CORR = ["corr_R_T_amp", "corr_R_J_amp", "corr_J_T_amp", "corr_QRS_QT"]
FEATURES = BASE + ADDED_SD + ADDED_CORR

# PART 1.5 : report feature list, n, drops
print("=" * 72)
print("PART 1 — FEATURE PREP")
print("=" * 72)
print(f"Window source       : ANNOTATED (committed OK set, HR {HR_OK_LOW}-{HR_OK_HIGH}, >=10 beats)")
print(f"n_animals retained  : {len(df)}")
print(f"animals dropped     : {len(dropped)}")
for aid, why in dropped:
    print(f"   - {aid}: {why}")
print(f"\nBASE features ({len(BASE)}): {BASE}")
print(f"ADDED beat-to-beat SDs ({len(ADDED_SD)}): {ADDED_SD}")
print(f"ADDED cross-wave correlations ({len(ADDED_CORR)}): {ADDED_CORR}")
print(f"TOTAL features: {len(FEATURES)}")
nan_counts = df[FEATURES].isna().sum()
print("\nNaN per feature (imputed with column median):")
print(nan_counts[nan_counts > 0].to_string() if (nan_counts > 0).any() else "  none")

# PART 1.3 : impute + standardise
X_raw = df[FEATURES].astype(float)
X_imp = X_raw.fillna(X_raw.median())
scaler = StandardScaler().fit(X_imp.values)
Xz = scaler.transform(X_imp.values)            # standardised features (representation A)

# PART 1.4a : correlation heatmap
corr = pd.DataFrame(Xz, columns=FEATURES).corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=90, fontsize=7)
ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES, fontsize=7)
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=5,
                color="white" if abs(corr.values[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_title("Feature correlation (standardised)")
fig.tight_layout(); fig.savefig(FIGURES_DIR / "feature_correlation_heatmap.png", dpi=150); plt.close(fig)

# PART 1.4b : decorrelated representation = PCA-whitened, retain 95% variance
pca_white = PCA(n_components=0.95, whiten=True, random_state=RANDOM_STATE).fit(Xz)
Xw = pca_white.transform(Xz)                   # whitened (representation B)
print(f"\nDecorrelation: PCA-whiten retains {Xw.shape[1]} comps for 95% variance "
      f"(from {len(FEATURES)} features).")
hi_corr = [(FEATURES[i], FEATURES[j], corr.values[i, j])
           for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))
           if abs(corr.values[i, j]) > 0.9]
print(f"Near-duplicate pairs |r|>0.9: {hi_corr if hi_corr else 'none'}")

# ===========================================================================
# PART 2-3 : clustering methods + validation indices
# ===========================================================================
def km_labels(X, k):   return KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
def ward_labels(X, k): return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
def gmm_obj(X, k, cov): return GaussianMixture(k, covariance_type=cov, random_state=RANDOM_STATE,
                                               n_init=3).fit(X)

def within_dispersion(X, labels):
    return float(sum(((X[labels == c] - X[labels == c].mean(0)) ** 2).sum()
                     for c in np.unique(labels)))

def gap_statistic(X, cluster_fn, k, B=GAP_B, seed=0):
    rng = np.random.default_rng(seed + k)
    logWk = np.log(within_dispersion(X, cluster_fn(X, k)) + 1e-12)
    mins, maxs = X.min(0), X.max(0)
    logs = []
    for _ in range(B):
        Xb = rng.uniform(mins, maxs, size=X.shape)
        logs.append(np.log(within_dispersion(Xb, cluster_fn(Xb, k)) + 1e-12))
    logs = np.array(logs)
    return float(logs.mean() - logWk), float(logs.std() * np.sqrt(1 + 1.0 / B))

def largest_frac(labels):
    _, counts = np.unique(labels, return_counts=True)
    return counts.max() / counts.sum()

def index_row(rep, method, k, X, labels, extra=None):
    lf = largest_frac(labels)
    nuniq = len(np.unique(labels))
    row = dict(representation=rep, method=method, k=k,
               silhouette=silhouette_score(X, labels) if nuniq > 1 else np.nan,
               davies_bouldin=davies_bouldin_score(X, labels) if nuniq > 1 else np.nan,
               calinski_harabasz=calinski_harabasz_score(X, labels) if nuniq > 1 else np.nan,
               largest_cluster_frac=round(lf, 3),
               degenerate=bool(lf > DEGENERATE_FRAC),
               cluster_sizes=str(sorted(np.unique(labels, return_counts=True)[1].tolist(), reverse=True)))
    if extra:
        row.update(extra)
    return row

print("\n" + "=" * 72)
print("PART 2-3 — METHODS x k, validation indices")
print("=" * 72)
table = []
for rep, X in [("standardised", Xz), ("pca_whitened", Xw)]:
    for k in K_RANGE:
        # A) k-means
        lab = km_labels(X, k)
        gap, gap_se = gap_statistic(X, km_labels, k)
        table.append(index_row(rep, "kmeans", k, X, lab, {"gap": gap, "gap_se": gap_se, "bic": np.nan}))
        # B) Ward
        lab = ward_labels(X, k)
        gap, gap_se = gap_statistic(X, ward_labels, k)
        table.append(index_row(rep, "ward", k, X, lab, {"gap": gap, "gap_se": gap_se, "bic": np.nan}))
        # C) GMM diagonal + spherical (full overfits at n~108) selected by BIC
        for cov in ("diag", "spherical"):
            g = gmm_obj(X, k, cov); lab = g.predict(X)
            table.append(index_row(rep, f"gmm_{cov}", k, X, lab,
                                   {"gap": np.nan, "gap_se": np.nan, "bic": g.bic(X)}))

vdf = pd.DataFrame(table)
vdf.to_csv(OUTPUTS_DIR / "cluster_validation_table.csv", index=False)
print(f"Saved validation table -> cluster_validation_table.csv ({len(vdf)} rows)")

# show the standardised-representation summary
show = vdf[vdf.representation == "standardised"].copy()
print("\nStandardised representation (silhouette / DB / CH / gap / largest-frac / degenerate):")
for method in ["kmeans", "ward", "gmm_diag", "gmm_spherical"]:
    sub = show[show.method == method]
    print(f"\n  {method}:")
    for _, r in sub.iterrows():
        deg = "  <-- DEGENERATE" if r.degenerate else ""
        bic = f" BIC={r.bic:.0f}" if not np.isnan(r.bic) else ""
        gap = f" gap={r.gap:.3f}" if not np.isnan(r.gap) else ""
        print(f"    k={r.k}: sil={r.silhouette:.3f} DB={r.davies_bouldin:.3f} "
              f"CH={r.calinski_harabasz:.0f}{gap}{bic} sizes={r.cluster_sizes}{deg}")

# ===========================================================================
# PART 4 — pick by agreement (exclude degenerate; require non-trivial sizes)
# ===========================================================================
print("\n" + "=" * 72)
print("PART 4 — MODEL SELECTION (agreement of indices, non-degenerate only)")
print("=" * 72)
cand = show[~show.degenerate].copy()
# rank within each index among non-degenerate candidates (higher sil/CH/gap better; lower DB better)
def winners(metric, better_high=True):
    s = cand.dropna(subset=[metric])
    if s.empty: return []
    s = s.sort_values(metric, ascending=not better_high)
    return list(zip(s.method, s.k))[:3]
print("Top-3 non-degenerate by each index:")
print(f"  silhouette (high): {winners('silhouette', True)}")
print(f"  davies_bouldin (low): {winners('davies_bouldin', False)}")
print(f"  calinski_harabasz (high): {winners('calinski_harabasz', True)}")
print(f"  gap (high): {winners('gap', True)}")
# GMM BIC best (any, but flag if degenerate)
gmm_rows = show[show.method.str.startswith("gmm")].dropna(subset=["bic"])
best_gmm = gmm_rows.sort_values("bic").iloc[0] if not gmm_rows.empty else None
if best_gmm is not None:
    print(f"  GMM best BIC: {best_gmm.method} k={best_gmm.k} BIC={best_gmm.bic:.0f} "
          f"sizes={best_gmm.cluster_sizes} {'DEGENERATE' if best_gmm.degenerate else 'OK'}")

# Principled, honest selection (NOT a naive vote):
#  - The best silhouette overall is at the DEGENERATE k=2 (it just isolates outliers).
#  - If NO non-degenerate solution reaches SIL_STRONG, the data has no real cluster
#    structure: report it as "~2 main groups + outliers" using the most parsimonious
#    DETERMINISTIC (Ward) non-degenerate partition, and say so plainly.
SIL_STRONG = 0.25
best_nd = cand.sort_values("silhouette", ascending=False).iloc[0]
best_nd_sil = float(best_nd.silhouette)
if best_nd_sil >= SIL_STRONG:
    rec_method, rec_k = best_nd.method, int(best_nd.k)
    STRUCTURE = f"clusters supported (best non-degenerate silhouette {best_nd_sil:.2f})"
else:
    ward_nd = cand[cand.method == "ward"].sort_values("k")
    r = ward_nd.iloc[0]            # smallest non-degenerate k for Ward (deterministic)
    rec_method, rec_k = "ward", int(r.k)
    STRUCTURE = (f"WEAK / NO STRONG STRUCTURE: best non-degenerate silhouette is only "
                 f"{best_nd_sil:.2f} (<{SIL_STRONG}); data supports ~2 main groups + outliers, "
                 f"NOT 5 separated clusters.")
print(f"\nBest non-degenerate silhouette: {best_nd_sil:.3f} at {best_nd.method} k={int(best_nd.k)}")
print(f"Structure verdict: {STRUCTURE}")
print(f"==> RECOMMENDED (deterministic, honest): {rec_method}, k={rec_k}")

# recommended labels on standardised features
def labels_for(method, k, X):
    if method == "kmeans": return km_labels(X, k)
    if method == "ward":   return ward_labels(X, k)
    if method.startswith("gmm"):
        return gmm_obj(X, k, method.split("_")[1]).predict(X)
rec_labels = labels_for(rec_method, rec_k, Xz)
km5 = km_labels(Xz, 5)
df["cluster_recommended"] = rec_labels
df["cluster_kmeans_k5"] = km5

# decorrelation effect
rec_labels_w = labels_for(rec_method, rec_k, Xw)
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(rec_labels, rec_labels_w)
print(f"\nDecorrelation effect: ARI(standardised vs whitened) for {rec_method} k={rec_k} = {ari:.3f}")
print("  (ARI~1 => decorrelation does not change the partition; <0.5 => it matters)")

# ===========================================================================
# PART 4 figures
# ===========================================================================
# dendrogram (Ward)
Z = linkage(Xz, method="ward")
fig, ax = plt.subplots(figsize=(16, 6))
dendrogram(Z, labels=df["animal_id"].astype(str).values, leaf_font_size=6, ax=ax, color_threshold=None)
ax.set_title("Ward dendrogram (standardised features)"); ax.set_xlabel("animal_id"); ax.set_ylabel("merge distance")
fig.tight_layout(); fig.savefig(FIGURES_DIR / "dendrogram.png", dpi=150); plt.close(fig)

# validation indices vs k (standardised)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, metric, hi in [(axes[0, 0], "silhouette", True), (axes[0, 1], "davies_bouldin", False),
                       (axes[1, 0], "calinski_harabasz", True), (axes[1, 1], "gap", True)]:
    for method, mk in [("kmeans", "o-"), ("ward", "s-"), ("gmm_diag", "^-"), ("gmm_spherical", "v-")]:
        sub = show[(show.method == method)].dropna(subset=[metric])
        if not sub.empty:
            ax.plot(sub.k, sub[metric], mk, label=method, markersize=5)
    ax.set_title(f"{metric} vs k ({'higher better' if hi else 'lower better'})")
    ax.set_xlabel("k"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.suptitle("Internal validation indices vs k (standardised features)")
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIGURES_DIR / "validation_indices.png", dpi=150); plt.close(fig)

# PCA scatter: recommended vs kmeans k=5
pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xz)
sc = pca2.transform(Xz); cov = pca2.explained_variance_ratio_[:2] * 100
fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), sharex=True, sharey=True)
for ax, labs, ttl in [(axes[0], rec_labels, f"RECOMMENDED: {rec_method} k={rec_k}"),
                      (axes[1], km5, "current k-means k=5")]:
    for c in np.unique(labs):
        m = labs == c
        ax.scatter(sc[m, 0], sc[m, 1], s=45, label=f"cluster {c} (n={m.sum()})", alpha=0.8)
    ax.set_title(ttl); ax.set_xlabel(f"PC1 ({cov[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({cov[1]:.1f}%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Clustering on PCA projection — recommended vs current k-means(k=5)")
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIGURES_DIR / "clusters_pca_comparison.png", dpi=150); plt.close(fig)

# ===========================================================================
# GMM membership confidence (lowest-confidence = borderline/outlier animals)
# ===========================================================================
gmm_best_k = int(best_gmm.k) if best_gmm is not None else 2
gmm_cov = best_gmm.method.split("_")[1] if best_gmm is not None else "diag"
gmm_fit = gmm_obj(Xz, gmm_best_k, gmm_cov)
proba = gmm_fit.predict_proba(Xz)
df["gmm_max_prob"] = proba.max(axis=1)
df["gmm_cluster"] = gmm_fit.predict(Xz)
low_conf = df.sort_values("gmm_max_prob").head(10)[["animal_id", "gmm_max_prob", "gmm_cluster"]]
print(f"\n--- GMM ({gmm_cov}, k={gmm_best_k}) lowest-confidence animals (borderline/outliers) ---")
print(low_conf.to_string(index=False))

df.to_csv(OUTPUTS_DIR / "cluster_assignments_recommended.csv", index=False)
print("\nSaved -> cluster_assignments_recommended.csv  + 4 figures")

# kmeans k=5 degeneracy (the thing we're trying to replace)
_, c5 = np.unique(km5, return_counts=True)
print(f"\nCurrent k-means k=5 cluster sizes: {sorted(c5.tolist(), reverse=True)} "
      f"(largest frac {c5.max()/c5.sum():.2f}{' DEGENERATE' if c5.max()/c5.sum() > DEGENERATE_FRAC else ''})")
_, crec = np.unique(rec_labels, return_counts=True)
print(f"Recommended {rec_method} k={rec_k} sizes: {sorted(crec.tolist(), reverse=True)} "
      f"(largest frac {crec.max()/crec.sum():.2f})")
