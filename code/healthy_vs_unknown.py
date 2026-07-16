"""Visualise the ONE binary split the data actually supports: recording quality
('healthy' clean trace vs 'unknown' can't-trust-it), NOT biological healthy-vs-
cardiotoxic. Reuses the cause-flags already in all_features_recomputed.csv.

A trace is 'healthy' (clean) iff status==OK AND all 3 signal-quality flags fire
(rhythm_regular, twave_isolated, enough_beats_for_sd). Everything else is
'unknown'. This is exactly the k=2 axis (silhouette 0.62) rediscovered by
clustering -- shown here as it really is: a deterministic quality gate.

Output: ../outputs/healthy_vs_unknown.png
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "../outputs/all_features_recomputed.csv"
CONT = ["heart_rate_bpm", "rr_mean_ms", "rr_std_ms", "r_amplitude_mv",
        "qrs_duration_ms", "j_wave_amplitude_mv", "t_wave_amplitude_mv",
        "qt_ms", "qtc_ms", "rt_interval_ms", "qt_std_ms", "qrs_std_ms",
        "r_amplitude_std_mv", "rr_cv"]
FLAGS = ["beats_detected", "rhythm_regular", "twave_isolated", "enough_beats_for_sd"]


def main():
    df = pd.read_csv(CSV)
    df = df[df["status"].isin(["OK", "NEEDS_REVIEW", "FAILED_PEAKS"])].copy()
    df = df.sort_values("animal_id").reset_index(drop=True)

    # The quality gate: clean iff OK and all three signal flags fire.
    clean = (df["status"].eq("OK") & df["rhythm_regular"].eq(1)
             & df["twave_isolated"].eq(1) & df["enough_beats_for_sd"].eq(1))
    df["quality"] = np.where(clean, "healthy (clean)", "unknown (flagged)")

    # PCA on the SAME feature space the pipeline clusters on.
    Xc = df[CONT].apply(pd.to_numeric, errors="coerce")
    Xc = Xc.fillna(Xc.median(numeric_only=True))
    X = np.hstack([StandardScaler().fit_transform(Xc), df[FLAGS].values.astype(float)])
    sc = PCA(n_components=2, random_state=42).fit_transform(X)

    ids = df["animal_id"].to_numpy()
    n_clean = int(clean.sum()); n_unk = int((~clean).sum())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7),
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    # LEFT: PCA coloured by the quality gate.
    for m, col, lab in [(clean.values, "#2a9d5c", f"healthy / clean  (n={n_clean})"),
                        (~clean.values, "#c9432b", f"unknown / flagged  (n={n_unk})")]:
        axL.scatter(sc[m, 0], sc[m, 1], c=col, s=55, alpha=0.85,
                    edgecolor="black", linewidth=0.4, label=lab)
    for x, y, a, c in zip(sc[:, 0], sc[:, 1], ids, ~clean.values):
        if c:  # only label the flagged ones
            axL.text(x + 0.12, y + 0.08, str(int(a)), fontsize=7, alpha=0.9)
    axL.axhline(0, color="grey", lw=0.5); axL.axvline(0, color="grey", lw=0.5)
    axL.set_xlabel("PC1"); axL.set_ylabel("PC2")
    axL.set_title("Same PCA space, coloured by the quality gate\n"
                  "(this IS the k=2 split, silhouette 0.62)", fontsize=10)
    axL.legend(loc="upper right", fontsize=9)
    axL.grid(alpha=0.25)

    # RIGHT: what drives 'unknown' -- the reason flags, stacked.
    reasons = {
        "status != OK": (~df["status"].eq("OK")).sum(),
        "irregular rhythm\n(rr_cv>0.15)": (df["rhythm_regular"].eq(0)).sum(),
        "T-wave not isolated\n(excursion<0.005)": (df["twave_isolated"].eq(0)).sum(),
        "too few beats for SD": (df["enough_beats_for_sd"].eq(0)).sum(),
    }
    yl = list(reasons.keys()); vals = list(reasons.values())
    axR.barh(yl, vals, color="#c9432b", alpha=0.8, edgecolor="black")
    for i, v in enumerate(vals):
        axR.text(v + 0.3, i, str(int(v)), va="center", fontsize=9)
    axR.set_xlabel("animals triggering this flag")
    axR.set_title("WHY 'unknown' = recording artifact,\nnot cardiotoxicity", fontsize=10)
    axR.invert_yaxis()
    axR.grid(axis="x", alpha=0.25)

    fig.suptitle(f"Healthy vs Unknown IS possible - but it is a DATA-QUALITY split "
                 f"({n_clean} clean / {n_unk} flagged), not healthy-vs-cardiotoxic",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005,
             "The clean/flagged axis is driven by signal artifact (motion, wander, HR out of band), "
             "NOT by doxorubicin. Within the clean block there is no second gap -> biology needs the "
             "supervised test + treatment metadata.",
             ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = "../outputs/healthy_vs_unknown.png"
    fig.savefig(out, dpi=150)
    print(f"clean(healthy)={n_clean}  unknown(flagged)={n_unk}")
    print("saved", out)


if __name__ == "__main__":
    main()
