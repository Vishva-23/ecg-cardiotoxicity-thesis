"""Cohort before/after audit for the QT-per-beat fix.

Reuses NB02's OWN functions (execs cell 2 constants + cell 4 functions from
the notebook, with the plotting/sklearn imports stripped) so the audit cannot
drift from the real pipeline. For every data file it runs the exact NB02 path
(parse -> baseline window -> quality guard -> average_beats) and reports:

    qt_ms / qtc_ms with the OLD per-beat method   vs
    qt_ms / qtc_ms with the NEW (RR-bounded, polarity-aware) method

Writes outputs/qt_backfill_audit.csv. Touches NO existing feature CSV.

Run:  python qt_cohort_audit.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

NB = "02_beat_averaging_and_clustering.ipynb"
OUT_CSV = Path("../outputs/qt_backfill_audit.csv")

# Manual LabChart QT means (from validation_table_filled.xlsx) for cross-check.
MANUAL_QT = {232: 48.67, 125: 44.33, 249: 41.0, 201: 43.67, 111: 25.67}


# --- Build a namespace from the notebook's real constants + functions --------
def load_pipeline_ns():
    nb = json.load(open(NB, encoding="utf-8"))
    cell2 = "".join(nb["cells"][2]["source"])
    cell4 = "".join(nb["cells"][4]["source"])
    # Strip imports we don't need / don't have (plotting, sklearn, hierarchy).
    drop = ("matplotlib", "sklearn", "scipy.cluster")
    cell2 = "\n".join(l for l in cell2.splitlines()
                      if not any(d in l for d in drop))
    ns = {}
    exec(compile(cell2, "nb_cell2", "exec"), ns)
    exec(compile(cell4, "nb_cell4", "exec"), ns)
    return ns


# --- OLD per-beat QT (NB02 before the fix), to compare against ---------------
def make_qt_old(FS):
    def qt_old(beat, t_ms):
        n30 = int(30 * FS / 1000); n100 = int(100 * FS / 1000)
        n40 = int(40 * FS / 1000); n50 = int(50 * FS / 1000); n20 = int(20 * FS / 1000)
        r = int(np.argmin(np.abs(t_ms)))
        s = r + n30; e = min(len(beat), r + n100)
        if e <= s:
            return np.nan
        t_peak = s + int(np.argmax(beat[s:e]))
        after_t = beat[t_peak:min(len(beat), t_peak + n40)]
        lo = max(0, r - n50); hi = max(lo + 1, r - n20)
        base = float(np.mean(beat[lo:hi]))
        near = np.where(after_t <= base + 0.01)[0]
        if len(near) == 0:
            return np.nan
        qt = (t_peak + int(near[0]) - r) * 1000.0 / FS
        return float(qt) if 30 <= qt <= 100 else np.nan
    return qt_old


def animal_id_from_name(stem):
    nums = re.findall(r"\d+", stem)
    if len(nums) >= 4:
        return int(nums[3])
    if len(nums) == 3:
        return int(nums[2])
    return None


def main():
    ns = load_pipeline_ns()
    g = ns.__getitem__
    FS = g("FS")
    HR_LO, HR_HI = g("HR_ACCEPT_LOW"), g("HR_ACCEPT_HIGH")
    QWIN = g("QUALITY_WIN_S")
    qt_old = make_qt_old(FS)

    parse_header_and_layout = g("parse_header_and_layout")
    load_ecg_file = g("load_ecg_file")
    find_baseline_window = g("find_baseline_window")
    bandpass_filter = g("bandpass_filter")
    detect_r_peaks = g("detect_r_peaks")
    quality_window_search = g("quality_window_search")
    average_beats = g("average_beats")
    extract_morphology_features = g("extract_morphology_features")
    mitchell = lambda qt, rr: (np.nan if (np.isnan(qt) or np.isnan(rr) or rr <= 0)
                               else float(qt / np.sqrt(rr / 100.0)))

    data_dir = g("DATA_DIR")
    files = sorted(p for p in data_dir.glob("*.txt"))
    print(f"Found {len(files)} .txt files\n")

    rows = []
    for path in files:
        aid = animal_id_from_name(path.stem)
        rec = {"animal_id": aid, "file": path.name, "status": "", "n_beats": None,
               "hr_bpm": None, "rr_mean_ms": None,
               "qt_old_ms": None, "qt_new_ms": None,
               "qtc_old_ms": None, "qtc_new_ms": None}
        try:
            if aid is None:
                rec["status"] = "FAILED_FILE"; rows.append(rec); continue
            n_header, lead, ecg_col, titles = parse_header_and_layout(path)
            voltage, markers = load_ecg_file(path, n_header, ecg_col)
            if voltage.size == 0:
                rec["status"] = "FAILED_FILE"; rows.append(rec); continue
            if float(np.median(np.abs(voltage))) > 50:
                rec["status"] = "FAILED_COLUMN"; rows.append(rec); continue
            start, end, rule, s_t, e_t = find_baseline_window(markers, len(voltage))
            if start is None:
                rec["status"] = "FAILED_ANNOTATION"; rows.append(rec); continue
            segment = voltage[start:end]
            if segment.size < 2 * FS:
                rec["status"] = "FAILED_ANNOTATION"; rows.append(rec); continue

            filtered = bandpass_filter(segment)
            peaks, inverted, hr, _ = detect_r_peaks(filtered)
            if inverted:
                filtered = -filtered

            # Mirror the NB02 quality guard for out-of-band HR.
            if not (HR_LO <= hr <= HR_HI):
                q = quality_window_search(voltage, start, len(voltage))
                if q["quality_flag"] == "solid":
                    start = q["best_start"]; end = start + int(QWIN * FS)
                    segment = voltage[start:end]
                    filtered = bandpass_filter(segment)
                    peaks, inverted, hr, _ = detect_r_peaks(filtered)
                    if inverted:
                        filtered = -filtered

            rec["hr_bpm"] = None if np.isnan(hr) else round(float(hr), 1)
            if len(peaks) < 10:
                rec["status"] = "FAILED_PEAKS"; rec["n_beats"] = int(len(peaks))
                rows.append(rec); continue

            rr_ms = np.diff(peaks) * (1000.0 / FS)
            rr_mean = float(np.mean(rr_ms))
            template, t_ms, beats = average_beats(filtered, peaks)

            # NEW: the notebook's current fixed function.
            feats_new = extract_morphology_features(template, t_ms, beats,
                                                    rr_ms=float(np.median(rr_ms)))
            qt_new = feats_new["qt_ms"]

            # OLD: replicate the pre-fix per-beat mean.
            old_est = [qt_old(b, t_ms) for b in beats] if beats is not None else []
            old_est = [v for v in old_est if not np.isnan(v)]
            qt_old_v = float(np.mean(old_est)) if old_est else np.nan

            rec["status"] = "OK" if HR_LO <= hr <= HR_HI else "NEEDS_REVIEW"
            rec["n_beats"] = int(len(peaks))
            rec["rr_mean_ms"] = round(rr_mean, 1)
            rec["qt_old_ms"] = None if np.isnan(qt_old_v) else round(qt_old_v, 1)
            rec["qt_new_ms"] = None if np.isnan(qt_new) else round(qt_new, 1)
            rec["qtc_old_ms"] = None if np.isnan(mitchell(qt_old_v, rr_mean)) else round(mitchell(qt_old_v, rr_mean), 1)
            rec["qtc_new_ms"] = None if np.isnan(mitchell(qt_new, rr_mean)) else round(mitchell(qt_new, rr_mean), 1)
        except Exception as exc:
            rec["status"] = f"ERROR:{type(exc).__name__}"
        rows.append(rec)
        print(f"  {path.name:32s} A{str(aid):>4}  {rec['status']:14s} "
              f"QT {rec['qt_old_ms']} -> {rec['qt_new_ms']} ms")

    df = pd.DataFrame(rows)
    df["qt_delta_ms"] = df["qt_new_ms"].astype(float) - df["qt_old_ms"].astype(float)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    ok = df[df["status"].isin(["OK", "NEEDS_REVIEW"])].copy()
    print("\n" + "=" * 70)
    print(f"Processed {len(df)} files | measured (OK/NEEDS_REVIEW): {len(ok)}")
    if len(ok):
        o = ok["qt_old_ms"].astype(float); n = ok["qt_new_ms"].astype(float)
        print(f"QT old  : mean {o.mean():.1f}  median {o.median():.1f} ms")
        print(f"QT new  : mean {n.mean():.1f}  median {n.median():.1f} ms")
        print(f"QT delta: mean {(n - o).mean():.1f} ms (new - old)")
    print("\n=== Manual cross-check (new QT vs LabChart M-cursor) ===")
    for aid, man in MANUAL_QT.items():
        r = df[df["animal_id"] == aid]
        if len(r) and r.iloc[0]["qt_new_ms"] is not None and not pd.isna(r.iloc[0]["qt_new_ms"]):
            old = r.iloc[0]["qt_old_ms"]; new = r.iloc[0]["qt_new_ms"]
            print(f"  A{aid}: manual {man:5.1f} | old {old} ({(float(old)-man)/man*100:+.0f}%) "
                  f"-> new {new} ({(float(new)-man)/man*100:+.0f}%)")
        else:
            print(f"  A{aid}: manual {man:5.1f} | not measured in this run")
    print(f"\nSaved: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
