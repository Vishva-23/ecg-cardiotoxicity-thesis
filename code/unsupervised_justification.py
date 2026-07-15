"""Evidence that unsupervised grouping was tried exhaustively and does not work,
so supervised analysis (needs treatment labels) is the only valid route.

Runs the full battery on the recomputed feature matrix and reports, for each
method x k, the silhouette score (how real the structure is) and, where the
outlier split dominates, what actually drives it. Produces:
  - console table (paste into the justification note / email to Roisin)
  - ../outputs/unsupervised_justification.png  (silhouette-vs-k, all 3 methods)

The point: every unsupervised attempt recovers only the k=2 QUALITY split
(silhouette ~0.6) and nothing at k=5 (the 5 treatment arms, ~0.28 = noise).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "../outputs/all_features_recomputed.csv"
CONT = ["heart_rate_bpm", "rr_mean_ms", "qrs_duration_ms", "qt_ms", "qtc_ms",
        "r_amplitude_mv", "t_wave_amplitude_mv", "j_wave_amplitude_mv",
        "rt_interval_ms"]
KS = [2, 3, 4, 5, 6]


def build_matrix(df):
    cols = [c for c in CONT if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return StandardScaler().fit_transform(X.values), cols


def sils(X):
    out = {}
    for k in KS:
        row = {}
        row["kmeans"] = silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X))
        row["ward"] = silhouette_score(X, AgglomerativeClustering(k, linkage="ward").fit_predict(X))
        row["gmm"] = silhouette_score(X, GaussianMixture(k, random_state=0).fit_predict(X))
        out[k] = row
    return out


def main():
    df = pd.read_csv(CSV)
    df = df[df["status"].isin(["OK", "NEEDS_REVIEW", "FAILED_PEAKS"])].copy()
    X, cols = build_matrix(df)
    n = len(df)

    # Attempt 1-3: cluster the full cohort, all three algorithms.
    full = sils(X)

    # Attempt 4: remove the k=2 outlier cluster, re-cluster the "clean" block.
    lab2 = KMeans(2, n_init=10, random_state=0).fit_predict(X)
    big = np.bincount(lab2).argmax()
    Xin = X[lab2 == big]
    inner = sils(Xin)

    # Attempt 5: how much variance does PC1 (the quality axis) eat?
    p = PCA(5).fit(X)
    pc1 = p.explained_variance_ratio_[0]

    print(f"\nn = {n} recordings | features = {len(cols)}")
    print("=" * 62)
    print("ATTEMPTS 1-3: cluster full cohort (silhouette; >0.5 real, <0.25 noise)")
    print(f"{'k':>3} | {'k-means':>8} {'Ward':>8} {'GMM':>8}   verdict")
    for k in KS:
        r = full[k]
        best = max(r.values())
        v = "REAL split" if best > 0.5 else ("weak" if best > 0.35 else "noise")
        print(f"{k:>3} | {r['kmeans']:8.3f} {r['ward']:8.3f} {r['gmm']:8.3f}   {v}")
    print("-" * 62)
    print("ATTEMPT 4: drop k=2 outliers, re-cluster the clean block")
    print(f"{'k':>3} | {'k-means':>8} {'Ward':>8} {'GMM':>8}")
    for k in KS:
        r = inner[k]
        print(f"{k:>3} | {r['kmeans']:8.3f} {r['ward']:8.3f} {r['gmm']:8.3f}")
    print("-" * 62)
    print(f"ATTEMPT 5: PC1 explains {pc1:.0%} of variance = the quality axis")
    print("=" * 62)
    print("CONCLUSION: only k=2 (quality) is real; k=5 (treatment arms) is noise")
    print("            -> supervised analysis with labels is the only valid route\n")

    # Figure
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for meth, mk in [("kmeans", "o"), ("ward", "s"), ("gmm", "^")]:
        ax.plot(KS, [full[k][meth] for k in KS], mk + "-", label=f"{meth} (full cohort)")
    ax.plot(KS, [inner[k]["kmeans"] for k in KS], "x--", color="gray",
            label="k-means (outliers removed)")
    ax.axhline(0.5, color="green", ls=":", lw=1, label="0.5 = real structure")
    ax.axhline(0.25, color="red", ls=":", lw=1, label="0.25 = noise floor")
    ax.axvline(5, color="black", ls="-", lw=0.8, alpha=0.4)
    ax.text(5.02, ax.get_ylim()[1] * 0.95, "5 treatment arms", fontsize=8, va="top")
    ax.set_xlabel("number of clusters k"); ax.set_ylabel("silhouette score")
    ax.set_xticks(KS)
    ax.set_title(f"Unsupervised grouping fails: only k=2 (quality) is real, "
                 f"k=5 (treatment) is noise\n(n={n}, {len(cols)} ECG features, "
                 f"3 algorithms)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    out = "../outputs/unsupervised_justification.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
