"""
Two fixes (reuse existing outputs; no re-extraction; nothing committed).

FIX 1: correct the `inverted` flag for 126 (-> True) and 235 (-> False) in every
       CSV that actually has an `inverted` column.
FIX 2: regenerate 3 PCA figures with animal-ID labels on every point.

Reproducibility: StandardScaler + PCA(random_state=42); Ward deterministic.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

ROOT = Path.cwd()
while not (ROOT / "data").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
OUT = ROOT / "outputs"; FIG = OUT / "figures"
OUTLIERS = [102, 107, 112, 115, 117, 126, 235]
INV_FIX = {126: True, 235: False}
RS = 42

# =====================================================================
# FIX 1 — correct inverted flags wherever the column exists
# =====================================================================
print("=" * 64)
print("FIX 1 — inverted-flag correction (126 -> True, 235 -> False)")
print("=" * 64)
named_files = ["cleaned_outliers_summary.csv", "outlier_after_pipeline.csv",
               "features_fallback.csv", "features_annotated.csv",
               "features_fallback_constrained.csv"]
for fn in named_files:
    p = OUT / fn
    if not p.exists():
        print(f"  {fn}: DOES NOT EXIST — nothing to update.")
        continue
    d = pd.read_csv(p)
    if "inverted" not in d.columns:
        print(f"  {fn}: no 'inverted' column — nothing to update.")
        continue
    for aid, newval in INV_FIX.items():
        sel = d["animal_id"] == aid
        if sel.any():
            before = d.loc[sel, "inverted"].tolist()
            d.loc[sel, "inverted"] = newval
            print(f"  {fn}: animal {aid}: inverted {before} -> [{newval}]")
        else:
            print(f"  {fn}: animal {aid} not present.")
    d.to_csv(p, index=False)
    print(f"  {fn}: saved.")

# =====================================================================
# FIX 2 — PCA figures with animal-ID labels
# =====================================================================
print("\n" + "=" * 64)
print("FIX 2 — PCA figures with animal-ID labels")
print("=" * 64)

# ---- shared 21-feature annotated PCA (110 animals) ----
FEAT21 = ["heart_rate_bpm", "rr_mean_ms", "rr_std_ms", "rr_cv", "r_amplitude_mv",
          "qrs_duration_ms", "j_wave_amplitude_mv", "t_wave_amplitude_mv",
          "rt_interval_ms", "qt_ms", "qtc_ms", "r_amplitude_std", "qrs_std",
          "j_wave_std", "t_wave_std", "rt_std", "qt_std",
          "corr_R_T_amp", "corr_R_J_amp", "corr_J_T_amp", "corr_QRS_QT"]
d21 = pd.read_csv(OUT / "cluster_assignments_recommended.csv")
X = d21[FEAT21].astype(float).fillna(d21[FEAT21].median())
scaler = StandardScaler().fit(X.values)
pca = PCA(n_components=2, random_state=RS).fit(scaler.transform(X.values))
coords = pca.transform(scaler.transform(X.values))
cov = pca.explained_variance_ratio_[:2] * 100
ids = d21["animal_id"].astype(int).values
coord = {int(a): coords[i] for i, a in enumerate(ids)}

# save coordinates for future reuse
pd.DataFrame({"animal_id": ids, "PC1": coords[:, 0], "PC2": coords[:, 1]}).to_csv(
    OUT / "pca_coordinates.csv", index=False)
print(f"Saved pca_coordinates.csv (110 animals, 21-feature annotated basis, "
      f"PC1={cov[0]:.1f}% PC2={cov[1]:.1f}%)")


def label_points(ax, xy, id_list):
    """small grey labels for all; larger bold black labels for the 7 outliers."""
    for (x, y), aid in zip(xy, id_list):
        if int(aid) in OUTLIERS:
            ax.text(x + 0.12, y + 0.12, str(int(aid)), fontsize=9, fontweight="bold", color="black")
        else:
            ax.text(x + 0.08, y + 0.06, str(int(aid)), fontsize=6, color="dimgray")


# ---- (c) pca_for_roisin.png : all 110, labelled ----
out_mask = np.array([int(a) in OUTLIERS for a in ids])
fig, ax = plt.subplots(figsize=(15, 11))
ax.scatter(coords[~out_mask, 0], coords[~out_mask, 1], s=42, c="steelblue", alpha=0.75, label="other animals")
ax.scatter(coords[out_mask, 0], coords[out_mask, 1], s=150, c="crimson", edgecolor="black",
           linewidth=1.3, zorder=5, label="named outliers")
label_points(ax, coords, ids)
ax.set_xlabel(f"PC1 ({cov[0]:.1f}% variance)"); ax.set_ylabel(f"PC2 ({cov[1]:.1f}% variance)")
ax.set_title("Baseline ECG PCA — 110 animals (animal IDs labelled)", fontsize=14)
ax.grid(alpha=0.3); ax.legend(fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "pca_for_roisin.png", dpi=150); plt.close(fig)
print("Saved (c) pca_for_roisin.png")

# ---- (b) clusters_pca_no_outliers.png : 102 animals, Ward k=4, labelled ----
# removal set = named 7 + any >3 SD on PC1/PC2 (matches Task 1 -> adds 108)
z1 = (coords[:, 0] - coords[:, 0].mean()) / coords[:, 0].std()
z2 = (coords[:, 1] - coords[:, 1].mean()) / coords[:, 1].std()
beyond3 = (np.abs(z1) > 3) | (np.abs(z2) > 3)
removed = set(OUTLIERS) | {int(ids[i]) for i in np.where(beyond3)[0]}
keep_mask = np.array([int(a) not in removed for a in ids])
keep_ids = ids[keep_mask]
# Ward k=4 on survivors (refit scaler on 102, deterministic)
Xk = d21.set_index("animal_id").loc[keep_ids, FEAT21].astype(float)
Xk = Xk.fillna(Xk.median())
Xkz = StandardScaler().fit_transform(Xk.values)
ward = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(Xkz)
keep_coords = coords[keep_mask]                    # SHARED basis (subset projection)
fig, ax = plt.subplots(figsize=(14, 10))
cmap = plt.cm.tab10
for c in np.unique(ward):
    m = ward == c
    ax.scatter(keep_coords[m, 0], keep_coords[m, 1], s=55, color=cmap(c % 10),
               alpha=0.85, label=f"cluster {c} (n={m.sum()})")
for (x, y), aid in zip(keep_coords, keep_ids):
    ax.text(x + 0.08, y + 0.06, str(int(aid)), fontsize=6, color="dimgray")
ax.set_xlabel(f"PC1 ({cov[0]:.1f}% variance)"); ax.set_ylabel(f"PC2 ({cov[1]:.1f}% variance)")
ax.set_title(f"Outliers removed — Ward k=4  (n={len(keep_ids)}, {len(removed)} removed)", fontsize=14)
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "clusters_pca_no_outliers.png", dpi=150); plt.close(fig)
print(f"Saved (b) clusters_pca_no_outliers.png (removed {sorted(removed)})")

# ---- (a) pca_before_after_constrained.png : 14-feature fallback basis, labelled ----
CF14 = ["heart_rate_bpm", "rr_mean_ms", "rr_std_ms", "r_amplitude_mv", "qrs_duration_ms",
        "j_wave_amplitude_mv", "t_wave_amplitude_mv", "qt_ms", "qtc_ms", "rt_interval_ms",
        "qt_std_ms", "qrs_std_ms", "r_amplitude_std_mv", "rr_cv"]
dA = pd.read_csv(OUT / "features_annotated.csv")
dC = pd.read_csv(OUT / "features_fallback_constrained.csv")
have = [c for c in CF14 if c in dA.columns and c in dC.columns]
# animals with valid features in BOTH
A = dA.dropna(subset=have).set_index("animal_id")
C = dC.dropna(subset=have).set_index("animal_id")
both = [a for a in A.index if a in C.index]
XA = A.loc[both, have].astype(float); XB = C.loc[both, have].astype(float)
med = XA.median(); XA = XA.fillna(med); XB = XB.fillna(med)
sc = StandardScaler().fit(XA.values)
pcaf = PCA(n_components=2, random_state=RS).fit(sc.transform(XA.values))
covf = pcaf.explained_variance_ratio_[:2] * 100
cA = pcaf.transform(sc.transform(XA.values))
cB = pcaf.transform(sc.transform(XB.values))
bid = [int(a) for a in both]
om = np.array([a in OUTLIERS for a in bid])
allc = np.vstack([cA, cB])
xlim = (allc[:, 0].min() - 0.5, allc[:, 0].max() + 0.5)
ylim = (allc[:, 1].min() - 0.5, allc[:, 1].max() + 0.5)
fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharex=True, sharey=True)
for ax, cc, ttl in [(axes[0], cA, "Annotated windows"),
                    (axes[1], cB, "With constrained window fallback (+/-120s)")]:
    ax.scatter(cc[~om, 0], cc[~om, 1], s=36, c="steelblue", alpha=0.7)
    ax.scatter(cc[om, 0], cc[om, 1], s=140, c="crimson", edgecolor="black", linewidth=1.3, zorder=5)
    for (x, y), aid in zip(cc, bid):
        if aid in OUTLIERS:
            ax.text(x + 0.1, y + 0.1, str(aid), fontsize=9, fontweight="bold", color="black")
        else:
            ax.text(x + 0.07, y + 0.05, str(aid), fontsize=6, color="dimgray")
    ax.set_title(ttl, fontsize=13)
    ax.set_xlabel(f"PC1 ({covf[0]:.1f}% var, shared basis)")
    ax.set_ylabel(f"PC2 ({covf[1]:.1f}% var, shared basis)")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(alpha=0.3)
fig.suptitle("PCA before vs after constrained fallback — animal IDs labelled "
             "(14-feature fallback basis)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG / "pca_before_after_constrained.png", dpi=150); plt.close(fig)
print(f"Saved (a) pca_before_after_constrained.png (n={len(both)} animals, "
      f"PC1={covf[0]:.1f}% PC2={covf[1]:.1f}%)")

print("\nDONE. Updated: outlier_after_pipeline.csv (inverted flags). "
      "Saved: pca_coordinates.csv + 3 labelled figures.")
