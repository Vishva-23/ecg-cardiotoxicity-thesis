"""
TASK 1 — clustering comparison with OUTLIERS REMOVED (Roisin's request).

Reuses the SAME 21-feature standardised table from the last run (loaded from
outputs/cluster_assignments_recommended.csv, which stored the 11 base + 6 SD +
4 correlation features). Removes named PCA outliers + any animal >3 SD on PC1/PC2
in the shared-basis PCA, REFITS the scaler on the survivors, and re-runs the full
k-means / Ward / GMM comparison with the same indices.

Reproducibility: RANDOM_STATE=42, gap RNG seeded; Ward and GMM-BIC deterministic.
Outputs (not committed): cluster_validation_table_no_outliers.csv,
figures/{dendrogram,validation_indices,clusters_pca}_no_outliers.png
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram

RANDOM_STATE = 42
GAP_B = 20
K_RANGE = list(range(2, 9))
DEGENERATE_FRAC = 0.70
SIL_STRONG = 0.25
NAMED_OUTLIERS = [102, 107, 112, 115, 117, 126, 235]

ROOT = Path.cwd()
while not (ROOT / "data").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
OUTPUTS_DIR = ROOT / "outputs"; FIGURES_DIR = OUTPUTS_DIR / "figures"

BASE = ["heart_rate_bpm", "rr_mean_ms", "rr_std_ms", "rr_cv", "r_amplitude_mv",
        "qrs_duration_ms", "j_wave_amplitude_mv", "t_wave_amplitude_mv",
        "rt_interval_ms", "qt_ms", "qtc_ms"]
ADDED_SD = ["r_amplitude_std", "qrs_std", "j_wave_std", "t_wave_std", "rt_std", "qt_std"]
ADDED_CORR = ["corr_R_T_amp", "corr_R_J_amp", "corr_J_T_amp", "corr_QRS_QT"]
FEATURES = BASE + ADDED_SD + ADDED_CORR

# ---- load the SAME prepared feature table from the last run ----
df = pd.read_csv(OUTPUTS_DIR / "cluster_assignments_recommended.csv")
missing = [c for c in FEATURES if c not in df.columns]
assert not missing, f"missing features in saved table: {missing}"
df = df[["animal_id"] + FEATURES].copy()
print(f"Loaded prepared 21-feature table: {len(df)} animals (full set).")

# ---- shared-basis PCA outlier detection (fit on FULL standardised set) ----
X_full = df[FEATURES].astype(float).fillna(df[FEATURES].median())
sc_full = StandardScaler().fit(X_full.values)
pca_full = PCA(n_components=2, random_state=RANDOM_STATE).fit(sc_full.transform(X_full.values))
pcs = pca_full.transform(sc_full.transform(X_full.values))
z1 = (pcs[:, 0] - pcs[:, 0].mean()) / pcs[:, 0].std(ddof=0)
z2 = (pcs[:, 1] - pcs[:, 1].mean()) / pcs[:, 1].std(ddof=0)
beyond3 = (np.abs(z1) > 3) | (np.abs(z2) > 3)

removed = []
for i, aid in enumerate(df.animal_id):
    is_named = int(aid) in NAMED_OUTLIERS
    is_3sd = bool(beyond3[i])
    if is_named or is_3sd:
        if is_named and is_3sd:
            reason = f"named outlier (also >3SD: PC1 z={z1[i]:.1f}, PC2 z={z2[i]:.1f})"
        elif is_named:
            reason = "named outlier"
        else:
            reason = f">3SD only (PC1 z={z1[i]:.1f}, PC2 z={z2[i]:.1f})"
        removed.append((int(aid), reason))

removed_ids = {a for a, _ in removed}
print("\n" + "=" * 64)
print("TASK 1 — OUTLIER REMOVAL")
print("=" * 64)
print(f"Removed {len(removed)} animals:")
for aid, why in sorted(removed):
    print(f"   - {aid}: {why}")
keep = df[~df.animal_id.isin(removed_ids)].reset_index(drop=True)
print(f"Remaining n = {len(keep)}  (from {len(df)})")

# ---- REFIT scaler on survivors only ----
Xk = keep[FEATURES].astype(float).fillna(keep[FEATURES].median())
scaler = StandardScaler().fit(Xk.values)
Xz = scaler.transform(Xk.values)

# ---- clustering machinery (identical to the with-outliers run) ----
def km_labels(X, k):   return KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
def ward_labels(X, k): return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
def gmm_obj(X, k, cov): return GaussianMixture(k, covariance_type=cov, random_state=RANDOM_STATE, n_init=3).fit(X)

def within_dispersion(X, labels):
    return float(sum(((X[labels == c] - X[labels == c].mean(0)) ** 2).sum() for c in np.unique(labels)))

def gap_statistic(X, fn, k, B=GAP_B, seed=0):
    rng = np.random.default_rng(seed + k)
    logWk = np.log(within_dispersion(X, fn(X, k)) + 1e-12)
    mins, maxs = X.min(0), X.max(0)
    logs = []
    for _ in range(B):
        Xb = rng.uniform(mins, maxs, size=X.shape)
        logs.append(np.log(within_dispersion(Xb, fn(Xb, k)) + 1e-12))
    logs = np.array(logs)
    return float(logs.mean() - logWk)

def largest_frac(labels):
    _, c = np.unique(labels, return_counts=True); return c.max() / c.sum()

def row(method, k, X, labels, bic=np.nan, gap=np.nan):
    nu = len(np.unique(labels)); lf = largest_frac(labels)
    return dict(method=method, k=k,
                silhouette=silhouette_score(X, labels) if nu > 1 else np.nan,
                gap=gap, davies_bouldin=davies_bouldin_score(X, labels) if nu > 1 else np.nan,
                calinski_harabasz=calinski_harabasz_score(X, labels) if nu > 1 else np.nan,
                bic=bic, largest_cluster_frac=round(lf, 3), degenerate=bool(lf > DEGENERATE_FRAC),
                cluster_sizes=str(sorted(np.unique(labels, return_counts=True)[1].tolist(), reverse=True)))

table = []
for k in K_RANGE:
    table.append(row("kmeans", k, Xz, km_labels(Xz, k), gap=gap_statistic(Xz, km_labels, k)))
    table.append(row("ward", k, Xz, ward_labels(Xz, k), gap=gap_statistic(Xz, ward_labels, k)))
    for cov in ("diag", "spherical"):
        g = gmm_obj(Xz, k, cov)
        table.append(row(f"gmm_{cov}", k, Xz, g.predict(Xz), bic=g.bic(Xz)))
vdf = pd.DataFrame(table)
vdf.to_csv(OUTPUTS_DIR / "cluster_validation_table_no_outliers.csv", index=False)

print("\nValidation (outliers removed):")
for method in ["kmeans", "ward", "gmm_diag", "gmm_spherical"]:
    print(f"\n  {method}:")
    for _, r in vdf[vdf.method == method].iterrows():
        bic = f" BIC={r.bic:.0f}" if not np.isnan(r.bic) else ""
        gap = f" gap={r.gap:.3f}" if not np.isnan(r.gap) else ""
        deg = "  <-- DEGENERATE" if r.degenerate else ""
        print(f"    k={r.k}: sil={r.silhouette:.3f} DB={r.davies_bouldin:.3f} "
              f"CH={r.calinski_harabasz:.0f}{gap}{bic} sizes={r.cluster_sizes}{deg}")

# ---- honest selection (same rule as before) ----
cand = vdf[~vdf.degenerate]
best_nd = cand.sort_values("silhouette", ascending=False).iloc[0]
best_sil = float(best_nd.silhouette)
if best_sil >= SIL_STRONG:
    rec_method, rec_k = best_nd.method, int(best_nd.k)
    verdict = f"clusters supported (best non-degenerate silhouette {best_sil:.2f})"
else:
    r = cand[cand.method == "ward"].sort_values("k").iloc[0]
    rec_method, rec_k = "ward", int(r.k)
    verdict = (f"WEAK / NO STRONG STRUCTURE: best non-degenerate silhouette {best_sil:.2f} "
               f"(<{SIL_STRONG}) even AFTER outlier removal.")
print(f"\nBest non-degenerate silhouette: {best_sil:.3f} at {best_nd.method} k={int(best_nd.k)}")
print(f"Structure verdict: {verdict}")
print(f"==> RECOMMENDED: {rec_method}, k={rec_k}")

def labels_for(method, k):
    if method == "kmeans": return km_labels(Xz, k)
    if method == "ward":   return ward_labels(Xz, k)
    return gmm_obj(Xz, k, method.split("_")[1]).predict(Xz)
rec_labels = labels_for(rec_method, rec_k)

# ---- figures ----
Z = linkage(Xz, "ward")
fig, ax = plt.subplots(figsize=(16, 6))
dendrogram(Z, labels=keep.animal_id.astype(str).values, leaf_font_size=6, ax=ax)
ax.set_title(f"Ward dendrogram — outliers removed (n={len(keep)})"); ax.set_ylabel("merge distance")
fig.tight_layout(); fig.savefig(FIGURES_DIR / "dendrogram_no_outliers.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, metric, hi in [(axes[0, 0], "silhouette", True), (axes[0, 1], "davies_bouldin", False),
                       (axes[1, 0], "calinski_harabasz", True), (axes[1, 1], "gap", True)]:
    for method, mk in [("kmeans", "o-"), ("ward", "s-"), ("gmm_diag", "^-"), ("gmm_spherical", "v-")]:
        sub = vdf[vdf.method == method].dropna(subset=[metric])
        if not sub.empty:
            ax.plot(sub.k, sub[metric], mk, label=method, markersize=5)
    ax.set_title(f"{metric} vs k ({'higher better' if hi else 'lower better'})")
    ax.set_xlabel("k"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.suptitle(f"Validation indices vs k — outliers removed (n={len(keep)})")
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIGURES_DIR / "validation_indices_no_outliers.png", dpi=150)
plt.close(fig)

pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xz)
sc = pca2.transform(Xz); cov = pca2.explained_variance_ratio_[:2] * 100
fig, ax = plt.subplots(figsize=(9, 7.5))
for c in np.unique(rec_labels):
    m = rec_labels == c
    ax.scatter(sc[m, 0], sc[m, 1], s=45, alpha=0.8, label=f"cluster {c} (n={m.sum()})")
ax.set_title(f"Recommended {rec_method} k={rec_k} (outliers removed, PCA refit on n={len(keep)})")
ax.set_xlabel(f"PC1 ({cov[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({cov[1]:.1f}%)")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGURES_DIR / "clusters_pca_no_outliers.png", dpi=150); plt.close(fig)

print("\nSaved: cluster_validation_table_no_outliers.csv + 3 figures")
_, crec = np.unique(rec_labels, return_counts=True)
print(f"Recommended sizes: {sorted(crec.tolist(), reverse=True)} (largest frac {crec.max()/crec.sum():.2f})")
