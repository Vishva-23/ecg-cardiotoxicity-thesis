"""Raw-vs-filtered triage metrics for the flagged animals.

Computes every statistic over the FULL selected baseline window. An earlier
throwaway version computed corr and %removed over only the first 1 s (the slice
that gets plotted) while computing HR over the full window -- a mixed-window
inconsistency. This script fixes that and reports both, so the difference is
auditable.

Metrics (all over the full baseline window unless stated):
  corr_full  : correlation between raw (de-meaned) and the FILTER-ONLY output.
               Filter-only (no polarity flip) so an inverted recording does not
               produce a spurious negative correlation.
  pct_full   : 100 * RMS(raw - filtered) / RMS(raw); the fraction of the signal
               the filter removed as noise/drift.
  corr_1s / pct_1s : the same, over the first 1 s, for comparison only.
  hr, beats, rr_cv : from R-peak detection over the full window.

Run:  python problem_animal_triage.py
Out:  ../outputs/problem_animal_triage.csv
"""
import numpy as np
import pandas as pd

from qt_cohort_audit import load_pipeline_ns, animal_id_from_name

FEATURES = "../outputs/all_features_recomputed.csv"
OUT = "../outputs/problem_animal_triage.csv"


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def stats(raw, filt):
    """corr + % removed between de-meaned raw and filter-only output."""
    r = raw - np.mean(raw)
    f = filt - np.mean(filt)
    if rms(r) == 0:
        return np.nan, np.nan
    return float(np.corrcoef(r, f)[0, 1]), 100.0 * rms(r - f) / rms(r)


def main():
    ns = load_pipeline_ns()
    g = ns.__getitem__
    FS = g("FS")
    parse = g("parse_header_and_layout"); load = g("load_ecg_file")
    findbw = g("find_baseline_window"); bp = g("bandpass_filter")
    detect = g("detect_r_peaks")

    df = pd.read_csv(FEATURES)
    clean = ((df.status == "OK") & (df.rhythm_regular == 1)
             & (df.twave_isolated == 1) & (df.enough_beats_for_sd == 1))
    flagged = sorted(df[~clean].animal_id.tolist())
    byid = {animal_id_from_name(p.stem): p for p in sorted(g("DATA_DIR").glob("*.txt"))}

    rows = []
    for aid in flagged:
        path = byid.get(aid)
        if path is None:
            continue
        n_header, lead, ecg_col, _ = parse(path)
        voltage, markers = load(path, n_header, ecg_col)
        start, end, *_ = findbw(markers, len(voltage))
        raw = voltage[start:end]
        filt = bp(raw)                      # filter-only: no polarity flip
        peaks, inverted, hr, _ = detect(filt)

        corr_full, pct_full = stats(raw, filt)                 # FULL window
        n1 = min(int(1.0 * FS), len(raw))
        corr_1s, pct_1s = stats(raw[:n1], filt[:n1])           # first 1 s only

        rr = np.diff(peaks) * (1000.0 / FS)
        rr_cv = float(np.std(rr) / np.mean(rr)) if len(peaks) > 2 else np.nan

        rows.append(dict(animal=aid, window_s=round(len(raw) / FS, 1),
                         corr_full=round(corr_full, 3), pct_full=round(pct_full),
                         corr_1s=round(corr_1s, 3), pct_1s=round(pct_1s),
                         hr=round(float(hr)), beats=len(peaks),
                         rr_cv=round(rr_cv, 3) if rr_cv == rr_cv else None,
                         inverted=bool(inverted)))

    out = pd.DataFrame(rows).sort_values("pct_full").reset_index(drop=True)
    out["d_corr"] = (out.corr_full - out.corr_1s).round(3)
    out["d_pct"] = (out.pct_full - out.pct_1s).round()
    out.to_csv(OUT, index=False)

    print(f"n = {len(out)} flagged animals | window length "
          f"{out.window_s.min():.0f}-{out.window_s.max():.0f} s "
          f"(vs the 1 s that gets plotted)\n")
    print(out[["animal", "window_s", "corr_full", "corr_1s", "d_corr",
               "pct_full", "pct_1s", "d_pct", "hr", "beats"]].to_string(index=False))
    print(f"\nMean |change| from using the full window instead of 1 s: "
          f"corr {out.d_corr.abs().mean():.3f}, %removed {out.d_pct.abs().mean():.0f} pts")
    print("saved", OUT)


if __name__ == "__main__":
    main()
