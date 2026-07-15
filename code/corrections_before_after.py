"""Show, concretely, how the QT correction changed feature values and moved the PCA.

- QT bug: the return-to-baseline search ran past the next beat and latched onto
  the following P-wave, inflating QT. The fix bounds the search to the RR
  interval. Effect: QT roughly halves into the mouse literature range.
- Because PCA is computed on standardized features, changing QT rotates the
  projection. We show the SAME animals under old-QT vs new-QT features.

Outputs:
  console  : cohort QT before/after summary
  figure   : ../outputs/corrections_before_after.png  (QT hist + PCA old vs new)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEAT = "../outputs/all_features_recomputed.csv"
AUD = "../outputs/qt_backfill_audit.csv"
# features that feed the PCA (continuous)
CONT = ["heart_rate_bpm", "rr_mean_ms", "qrs_duration_ms", "qt_ms", "qtc_ms",
        "r_amplitude_mv", "t_wave_amplitude_mv", "j_wave_amplitude_mv",
        "rt_interval_ms"]


def matrix(df, qt_col, qtc_col):
    d = df.copy()
    d["qt_ms"] = df[qt_col]
    d["qtc_ms"] = df[qtc_col]
    X = d[CONT].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return StandardScaler().fit_transform(X.values)


def main():
    f = pd.read_csv(FEAT)
    a = pd.read_csv(AUD)[["animal_id", "qt_old_ms", "qt_new_ms", "qtc_old_ms", "qtc_new_ms"]]
    df = f.merge(a, on="animal_id", how="inner")
    df = df[df["status"].isin(["OK", "NEEDS_REVIEW"])].copy()
    n = len(df)

    qo, qn = df["qt_old_ms"], df["qt_new_ms"]
    print(f"n = {n} animals")
    print("QT interval (ms)      old  ->  new")
    print(f"  median            {qo.median():5.1f} -> {qn.median():5.1f}")
    print(f"  mean              {qo.mean():5.1f} -> {qn.mean():5.1f}")
    print(f"  mean reduction    {(qo - qn).mean():5.1f} ms  ({(1-qn.mean()/qo.mean())*100:.0f}% lower)")
    print(f"QTc median          {df['qtc_old_ms'].median():5.1f} -> {df['qtc_new_ms'].median():5.1f}"
          f"   (mouse literature ~41 ms)")

    Xold = matrix(df, "qt_old_ms", "qtc_old_ms")
    Xnew = matrix(df, "qt_new_ms", "qtc_new_ms")
    # Fit ONE PCA on the new data, project both, so axes are comparable.
    pca = PCA(2).fit(Xnew)
    Pold, Pnew = pca.transform(Xold), pca.transform(Xnew)
    shift = np.linalg.norm(Pnew - Pold, axis=1)
    print(f"mean PCA shift per animal (old->new): {shift.mean():.2f} PC units; "
          f"max {shift.max():.2f}")

    fig, (axH, axP) = plt.subplots(1, 2, figsize=(14, 6))

    axH.hist(qo, bins=20, alpha=0.6, color="#c9432b", label=f"old QT (median {qo.median():.0f} ms)")
    axH.hist(qn, bins=20, alpha=0.6, color="#2a9d5c", label=f"new QT (median {qn.median():.0f} ms)")
    axH.axvspan(40, 55, color="gray", alpha=0.15, label="mouse literature band")
    axH.set_xlabel("QT interval (ms)"); axH.set_ylabel("animals")
    axH.set_title("QT fix: P-wave latching removed\n-> values drop into physiological range", fontsize=11)
    axH.legend(fontsize=9); axH.grid(alpha=0.25)

    for i in range(n):
        axP.plot([Pold[i, 0], Pnew[i, 0]], [Pold[i, 1], Pnew[i, 1]],
                 color="gray", lw=0.5, alpha=0.4, zorder=1)
    axP.scatter(Pold[:, 0], Pold[:, 1], s=28, c="#c9432b", label="old features", zorder=2, edgecolor="k", linewidth=0.3)
    axP.scatter(Pnew[:, 0], Pnew[:, 1], s=28, c="#2a9d5c", label="new features", zorder=3, edgecolor="k", linewidth=0.3)
    axP.set_xlabel("PC1"); axP.set_ylabel("PC2")
    axP.set_title(f"Same animals move in PCA space after the fix\n(mean shift {shift.mean():.2f} PC units)", fontsize=11)
    axP.legend(fontsize=9); axP.grid(alpha=0.25)

    fig.suptitle("Effect of the QT correction on feature values and the PCA projection",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "../outputs/corrections_before_after.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
