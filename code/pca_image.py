"""Reproduce NB02's PCA figure on the full cohort, using the current (QT-fixed)
pipeline. Extracts morphology features for every file via NB02's own functions,
applies the same informative-missingness rules (Rule A: rr_cv>0.15 nulls
rr_mean/qt/qtc; Rule B: t_wave_excursion<0.005 nulls t_amp/rt/qt/qtc), builds
the 14 continuous + 4 cause-flag matrix (FLAG_WEIGHT=1.0, median-impute +
StandardScaler on continuous, flags un-scaled), then PCA(2) + KMeans(k=5) and
saves the labelled scatter with outliers highlighted.

Outputs:
  ../outputs/all_features_recomputed.csv   (feature table)
  ../outputs/pca_k5.png                     (the image)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qt_cohort_audit as A

RR_CV_IRREGULAR   = 0.15
T_WAVE_ISO_MIN_MV = 0.005
FLAG_WEIGHT       = 1.0
CLUSTER_COLS = ["heart_rate_bpm", "rr_mean_ms", "rr_std_ms",
                "r_amplitude_mv", "qrs_duration_ms",
                "j_wave_amplitude_mv", "t_wave_amplitude_mv",
                "qt_ms", "qtc_ms", "rt_interval_ms",
                "qt_std_ms", "qrs_std_ms", "r_amplitude_std_mv", "rr_cv"]
FLAG_COLS = ["beats_detected", "rhythm_regular", "twave_isolated", "enough_beats_for_sd"]


def extract_all():
    ns = A.load_pipeline_ns(); g = ns.__getitem__
    FS = g("FS"); HR_LO, HR_HI = g("HR_ACCEPT_LOW"), g("HR_ACCEPT_HIGH"); QWIN = g("QUALITY_WIN_S")
    parse = g("parse_header_and_layout"); load = g("load_ecg_file")
    win = g("find_baseline_window"); bp = g("bandpass_filter")
    detect = g("detect_r_peaks"); qsearch = g("quality_window_search")
    avg = g("average_beats"); morph = g("extract_morphology_features")
    mitchell = lambda qt, rr: (np.nan if (np.isnan(qt) or np.isnan(rr) or rr <= 0)
                               else float(qt / np.sqrt(rr / 100.0)))
    rows = []
    for path in sorted(g("DATA_DIR").glob("*.txt")):
        aid = A.animal_id_from_name(path.stem)
        rec = {"animal_id": aid, "source_file": path.name, "status": "FAILED_FILE"}
        try:
            if aid is None:
                rows.append(rec); continue
            n_h, lead, ecg_col, titles = parse(path)
            v, markers = load(path, n_h, ecg_col)
            if v.size == 0 or float(np.median(np.abs(v))) > 50:
                rec["status"] = "FAILED_COLUMN"; rows.append(rec); continue
            s, e, rule, st, et = win(markers, len(v))
            if s is None or (e - s) < 2 * FS:
                rec["status"] = "FAILED_ANNOTATION"; rows.append(rec); continue
            filt = bp(v[s:e]); peaks, inv, hr, _ = detect(filt)
            if inv: filt = -filt
            if not (HR_LO <= hr <= HR_HI):
                q = qsearch(v, s, len(v))
                if q["quality_flag"] == "solid":
                    s = q["best_start"]; e = s + int(QWIN * FS)
                    filt = bp(v[s:e]); peaks, inv, hr, _ = detect(filt)
                    if inv: filt = -filt
            if len(peaks) < 10:
                rec.update({"status": "FAILED_PEAKS", "heart_rate_bpm": float(hr) if not np.isnan(hr) else np.nan})
                rows.append(rec); continue
            rr = np.diff(peaks) * (1000.0 / FS)
            tmpl, t_ms, beats = avg(filt, peaks)
            feats = morph(tmpl, t_ms, beats, rr_ms=float(np.median(rr)))
            rr_mean = float(np.mean(rr))
            feats["qtc_ms"] = mitchell(feats["qt_ms"], rr_mean)
            rec.update({
                "status": "OK" if HR_LO <= hr <= HR_HI else "NEEDS_REVIEW",
                "heart_rate_bpm": float(hr), "rr_mean_ms": rr_mean,
                "rr_std_ms": float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0,
                **feats})
        except Exception as exc:
            rec["status"] = f"ERROR:{type(exc).__name__}"
        rows.append(rec)
        print(f"  {path.name:32s} A{str(aid):>4} {rec['status']}")
    return pd.DataFrame(rows)


def main():
    df = extract_all()
    df = df[df["status"].isin(["OK", "NEEDS_REVIEW", "FAILED_PEAKS"])].copy()
    df = df.sort_values("animal_id").reset_index(drop=True)

    # rr_cv, then Rule A / Rule B nulling (informative missingness).
    df["rr_cv"] = df["rr_std_ms"] / df["rr_mean_ms"]
    df.loc[df["rr_cv"] > RR_CV_IRREGULAR, ["rr_mean_ms", "qt_ms", "qtc_ms"]] = np.nan
    tw = df["t_wave_excursion_mv"].notna() & (df["t_wave_excursion_mv"] < T_WAVE_ISO_MIN_MV)
    df.loc[tw, ["t_wave_amplitude_mv", "rt_interval_ms", "qt_ms", "qtc_ms"]] = np.nan

    df["beats_detected"]      = (df["status"] != "FAILED_PEAKS").astype(int)
    df["rhythm_regular"]      = (df["rr_cv"].notna() & (df["rr_cv"] <= RR_CV_IRREGULAR)).astype(int)
    df["twave_isolated"]      = (df["t_wave_excursion_mv"].notna() & (df["t_wave_excursion_mv"] >= T_WAVE_ISO_MIN_MV)).astype(int)
    df["enough_beats_for_sd"] = df["qt_std_ms"].notna().astype(int)

    df.to_csv("../outputs/all_features_recomputed.csv", index=False)

    Xc = df[CLUSTER_COLS].apply(pd.to_numeric, errors="coerce")
    Xc = Xc.fillna(Xc.median(numeric_only=True))
    Xc_s = StandardScaler().fit_transform(Xc)
    Xf = df[FLAG_COLS].values.astype(float) * FLAG_WEIGHT
    X = np.hstack([Xc_s, Xf])
    print(f"\nMatrix: {X.shape[0]} animals x {X.shape[1]} features")

    pca = PCA(n_components=2, random_state=42)
    sc = pca.fit_transform(X)
    p1, p2 = pca.explained_variance_ratio_[:2] * 100
    labels = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(X)
    vmin, vmax = int(labels.min()), int(labels.max())
    ids = df["animal_id"].to_numpy()
    out = (sc[:, 0] > 5) | (np.abs(sc[:, 1]) > 3)
    out_ids = [int(a) for a in ids[out]]

    fig, ax = plt.subplots(figsize=(14, 10))
    nrm = ~out
    ax.scatter(sc[nrm, 0], sc[nrm, 1], c=labels[nrm], cmap="tab10", vmin=vmin, vmax=vmax, s=40, alpha=0.85)
    ax.scatter(sc[out, 0], sc[out, 1], c=labels[out], cmap="tab10", vmin=vmin, vmax=vmax,
               s=120, edgecolor="black", linewidth=1.5, zorder=5)
    for x, y, a in zip(sc[:, 0], sc[:, 1], ids):
        ax.text(x + 0.15, y + 0.1, str(int(a)), fontsize=7, alpha=0.9)
    ax.set_xlabel(f"PC1 ({p1:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({p2:.1f}% variance)", fontsize=12)
    ax.grid(alpha=0.3)
    fig.suptitle(f"PCA of baseline ECG features - {len(df)} animals (QT-fixed pipeline)",
                 fontsize=15, fontweight="bold")
    ax.set_title("Colour = k=5 k-means. Point labels = animal ID. Black-edged = statistical "
                 "outliers (PC1>5 or |PC2|>3). Unsupervised - blind to treatment metadata.", fontsize=9)
    ax.text(0.99, 0.01, "Outliers: " + (", ".join(map(str, out_ids)) or "none"),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("../outputs/pca_k5.png", dpi=150)
    print(f"\nPC1={p1:.1f}% PC2={p2:.1f}% | outliers: {out_ids}")
    print("Saved ../outputs/pca_k5.png and ../outputs/all_features_recomputed.csv")


if __name__ == "__main__":
    main()
