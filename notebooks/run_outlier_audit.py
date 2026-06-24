"""
Outlier audit — run ONLY the 7 PCA-outlier animals through the EXISTING
Notebook 02 pipeline functions (no rewrite).

The pipeline functions are reused verbatim by extracting and exec'ing the
relevant definition cells from 02_beat_averaging_and_clustering.ipynb, then
driving only the 7 files through them in the same order cell 6 uses:
  load -> header/col detection -> find_baseline_window -> bandpass_filter
  -> detect_r_peaks (invert retry) -> average_beats -> extract_morphology_features
  -> mitchell_qtc

Outputs:
  outputs/outlier_after_pipeline.csv   per-animal results table
  outputs/outlier_clean_signals.csv    filtered baseline signal actually used (long)
  outputs/outlier_averaged_beats.png   2x4 grid of averaged beats + overlay
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB = "02_beat_averaging_and_clustering.ipynb"

# Physiological HR band for THIS audit (per supervisor): outside => flag honestly
PHYS_LOW, PHYS_HIGH = 400, 650

OUTLIER_FILES = [
    "2024_11_01_102.txt",   # 102
    "2024_11_01_107.txt",   # 107
    "2024_11_13_112.txt",   # 112
    "2024_11_13_115.txt",   # 115
    "2024_11_14_117.txt",   # 117
    "2024_11_15_126.txt",   # 126
    "2024_12_18_235.txt",   # 235
]

# ---------------------------------------------------------------------------
# 1) Reuse the pipeline: exec the notebook's own definition cells into P.
# ---------------------------------------------------------------------------
nb = json.load(open(NB, encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]

setup_cell = next(s for s in codes if s.startswith("import re"))          # cell 2: imports/paths/constants
helper_cell = next(s for s in codes if "def extract_morphology_features" in s)  # cell 4: all functions
batch_cell = next(s for s in codes if "def mitchell_qtc" in s)            # cell 6: mitchell_qtc lives here

P = {"__name__": "__pipeline__"}
exec(compile(setup_cell, "nb02_cell_setup", "exec"), P)
exec(compile(helper_cell, "nb02_cell_helpers", "exec"), P)
# take only the QTC_FORMULA_NAME + mitchell_qtc definition, not the batch loop
a = batch_cell.index("QTC_FORMULA_NAME =")
b = batch_cell.index("per_animal", a)
exec(compile(batch_cell[a:b], "nb02_mitchell", "exec"), P)

# pull the reused objects
FS            = P["FS"]
DATA_DIR      = P["DATA_DIR"]
OUTPUTS_DIR   = P["OUTPUTS_DIR"]
FIGURES_DIR   = P["FIGURES_DIR"]
parse_filename            = P["parse_filename"]
parse_header_and_layout   = P["parse_header_and_layout"]
load_ecg_file             = P["load_ecg_file"]
find_baseline_window      = P["find_baseline_window"]
bandpass_filter           = P["bandpass_filter"]
detect_r_peaks            = P["detect_r_peaks"]
average_beats             = P["average_beats"]
extract_morphology_features = P["extract_morphology_features"]
mitchell_qtc              = P["mitchell_qtc"]

print(f"Reused pipeline functions from {NB} (parameters: "
      f"BANDPASS_HIGH_HZ={P['BANDPASS_HIGH_HZ']}, REFRACTORY_MS={P['REFRACTORY_MS']}, "
      f"MIN_RR_MS={P['MIN_RR_MS']}, QTc={P['QTC_FORMULA_NAME']})")

# Does the committed bandpass_filter include the 50/100/150 Hz notch cascade?
# (detect from the notebook cell source, since exec'd funcs have no source file)
HAS_NOTCH = "iirnotch" in helper_cell
print(f"bandpass_filter includes notch cascade: {HAS_NOTCH}")

# No-notch comparison filter (pre-notch baseline: identical butter bandpass, no notch).
# Built ONLY for the side-by-side HR column the supervisor asked for — not the pipeline.
from scipy.signal import butter, filtfilt
def bandpass_no_notch(x, fs=FS):
    nyq = 0.5 * fs
    bb, aa = butter(P["BANDPASS_ORDER"],
                    [P["BANDPASS_LOW_HZ"] / nyq, P["BANDPASS_HIGH_HZ"] / nyq], btype="band")
    return filtfilt(bb, aa, x)

# ---------------------------------------------------------------------------
# 2) Drive the 7 files through the pipeline (replicating cell-6 orchestration).
# ---------------------------------------------------------------------------
rows = []
templates = {}        # animal_id -> (template, t_ms, beats, hr, n_beats, inverted)
clean_long = []       # long-format filtered signal rows

for fname in OUTLIER_FILES:
    path = DATA_DIR / fname
    rec = {"animal_id": None, "source_file": fname, "rule_used": None,
           "start_annotation": None, "end_annotation": None,
           "n_beats": None, "heart_rate_bpm": None, "heart_rate_no_notch_bpm": None,
           "rr_mean_ms": None, "rr_std_ms": None, "r_amplitude_mv": None,
           "qrs_duration_ms": None, "j_wave_amplitude_mv": None,
           "t_wave_amplitude_mv": None, "qt_ms": None, "qtc_ms": None,
           "inverted": None, "status": "", "reason": ""}
    rows.append(rec)
    try:
        animal_id, _ = parse_filename(path)
        rec["animal_id"] = animal_id
        if not path.exists():
            rec["status"], rec["reason"] = "FAILED", "file not found in data/"
            continue

        n_header, lead_cols, ecg_col, titles = parse_header_and_layout(path)
        voltage, markers = load_ecg_file(path, n_header, ecg_col)
        if voltage.size == 0:
            rec["status"], rec["reason"] = "FAILED", "no voltage rows parsed"
            continue
        med_abs = float(np.median(np.abs(voltage)))
        if med_abs > 50:
            rec["status"], rec["reason"] = "FAILED", f"column not mV (median|.|={med_abs:.1f})"
            continue

        start, end, rule, s_text, e_text = find_baseline_window(markers, len(voltage))
        rec["rule_used"], rec["start_annotation"], rec["end_annotation"] = rule, s_text, e_text
        if start is None:
            rec["status"], rec["reason"] = "FAILED", f"no baseline window ({rule})"
            continue
        segment = voltage[start:end]
        if segment.size < 2 * FS:
            rec["status"], rec["reason"] = "FAILED", f"baseline window too short ({segment.size} samples)"
            continue

        # --- the animal's signal through the REAL pipeline filter (with notch) ---
        filtered = bandpass_filter(segment)
        peaks, inverted, hr, _ = detect_r_peaks(filtered)
        rec["inverted"] = bool(inverted)
        if inverted:
            filtered = -filtered

        # --- no-notch comparison HR (same detector) ---
        filt_nn = bandpass_no_notch(segment)
        peaks_nn, inv_nn, hr_nn, _ = detect_r_peaks(filt_nn)
        rec["heart_rate_no_notch_bpm"] = round(float(hr_nn), 1) if not np.isnan(hr_nn) else np.nan

        rec["n_beats"] = int(len(peaks))
        rec["heart_rate_bpm"] = round(float(hr), 1) if not np.isnan(hr) else np.nan

        if len(peaks) < 10:
            rec["status"] = "FAILED_PEAKS"
            rec["reason"] = f"only {len(peaks)} R peaks detected (need >=10)"
            # still try a template if >=1 beat for the figure
            template, t_ms, beats = average_beats(filtered, peaks)
            if template is not None:
                templates[animal_id] = (template, t_ms, beats, hr, len(peaks), inverted)
            continue

        rr_ms = np.diff(peaks) * (1000.0 / FS)
        template, t_ms, beats = average_beats(filtered, peaks)
        feats = extract_morphology_features(template, t_ms, beats)
        rr_mean_ms = float(np.mean(rr_ms))
        qtc = mitchell_qtc(feats["qt_ms"], rr_mean_ms)

        rec["rr_mean_ms"]          = round(rr_mean_ms, 2)
        rec["rr_std_ms"]           = round(float(np.std(rr_ms, ddof=1)), 2) if len(rr_ms) > 1 else 0.0
        rec["r_amplitude_mv"]      = round(feats["r_amplitude_mv"], 4)
        rec["qrs_duration_ms"]     = round(feats["qrs_duration_ms"], 2)
        rec["j_wave_amplitude_mv"] = round(feats["j_wave_amplitude_mv"], 4)
        rec["t_wave_amplitude_mv"] = round(feats["t_wave_amplitude_mv"], 4)
        rec["qt_ms"]               = round(feats["qt_ms"], 2) if not np.isnan(feats["qt_ms"]) else np.nan
        rec["qtc_ms"]              = round(qtc, 2) if not np.isnan(qtc) else np.nan

        templates[animal_id] = (template, t_ms, beats, hr, len(peaks), inverted)

        # honest physiological judgement (per supervisor: 400-650 bpm)
        if np.isnan(hr):
            rec["status"], rec["reason"] = "NEEDS_REVIEW", "heart rate undefined"
        elif PHYS_LOW <= hr <= PHYS_HIGH:
            rec["status"], rec["reason"] = "OK", ""
        else:
            side = "below" if hr < PHYS_LOW else "above"
            rec["status"] = "NEEDS_REVIEW"
            rec["reason"] = f"HR {hr:.0f} bpm {side} physiological 400-650 band"

        # clean signal actually used (post-invert filtered)
        t_s = np.arange(len(filtered)) / FS
        clean_long.append(pd.DataFrame({
            "animal_id": animal_id, "time_s": np.round(t_s, 4),
            "mv": np.round(filtered, 6)}))

    except Exception as e:
        rec["status"], rec["reason"] = "FAILED", f"{type(e).__name__}: {e}"

# ---------------------------------------------------------------------------
# 2/3) Tables
# ---------------------------------------------------------------------------
col_order = ["animal_id", "source_file", "rule_used", "start_annotation", "end_annotation",
             "n_beats", "heart_rate_bpm", "heart_rate_no_notch_bpm", "rr_mean_ms", "rr_std_ms",
             "r_amplitude_mv", "qrs_duration_ms", "j_wave_amplitude_mv", "t_wave_amplitude_mv",
             "qt_ms", "qtc_ms", "inverted", "status", "reason"]
table = pd.DataFrame(rows)[col_order]
table.to_csv(OUTPUTS_DIR / "outlier_after_pipeline.csv", index=False)

with pd.option_context("display.max_columns", None, "display.width", 240):
    print("\n=== OUTLIER AUDIT — after pipeline ===")
    print(table[["animal_id", "n_beats", "heart_rate_bpm", "heart_rate_no_notch_bpm",
                 "qt_ms", "qtc_ms", "inverted", "status", "reason"]].to_string(index=False))

# clean signals long CSV
if clean_long:
    clean_df = pd.concat(clean_long, ignore_index=True)
    clean_df.to_csv(OUTPUTS_DIR / "outlier_clean_signals.csv", index=False)
    print(f"\nSaved clean signals: {len(clean_df):,} rows "
          f"({clean_df['animal_id'].nunique()} animals) -> outlier_clean_signals.csv")
else:
    print("\nNo clean signals saved (no animal produced a usable filtered segment).")

# ---------------------------------------------------------------------------
# 4) Figure: 2x4 grid (7 animals + 1 overlay panel)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.reshape(-1)
order = [parse_filename(DATA_DIR / f)[0] for f in OUTLIER_FILES]
cmap = plt.cm.tab10

for ax, aid in zip(axes[:7], order):
    if aid in templates:
        template, t_ms, beats, hr, nb_, inv = templates[aid]
        if beats is not None and getattr(beats, "ndim", 0) == 2:
            for beat in beats:
                ax.plot(t_ms, beat, color="0.7", linewidth=0.4, alpha=0.5)
        ax.plot(t_ms, template, color="crimson", linewidth=2.0)
        ax.axvline(0, color="k", linewidth=0.4, linestyle="--", alpha=0.5)
        inv_flag = "  INV" if inv else ""
        hr_txt = f"{hr:.0f}bpm" if not np.isnan(hr) else "HR?"
        ax.set_title(f"Animal {aid}   {hr_txt}   n={nb_}{inv_flag}", fontsize=10)
    else:
        r = next(x for x in rows if x["animal_id"] == aid)
        ax.text(0.5, 0.5, f"Animal {aid}\n{r['status']}\n{r['reason']}",
                ha="center", va="center", fontsize=9, transform=ax.transAxes)
        ax.set_title(f"Animal {aid}  — no template", fontsize=10)
    ax.set_xlabel("Time from R (ms)", fontsize=8)
    ax.set_ylabel("mV", fontsize=8)
    ax.grid(alpha=0.3)

# final panel: overlay of all templates
ax = axes[7]
for i, aid in enumerate(order):
    if aid in templates and templates[aid][0] is not None:
        template, t_ms, beats, hr, nb_, inv = templates[aid]
        ax.plot(t_ms, template, color=cmap(i % 10), linewidth=1.6, label=f"{aid}")
ax.axvline(0, color="k", linewidth=0.4, linestyle="--", alpha=0.5)
ax.set_title("All outlier templates overlaid", fontsize=10)
ax.set_xlabel("Time from R (ms)", fontsize=8)
ax.set_ylabel("mV", fontsize=8)
ax.legend(fontsize=7, ncol=2, title="animal")
ax.grid(alpha=0.3)

fig.suptitle("Outlier animals through the existing pipeline — averaged beat (bold) over "
             "individual beats (grey)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIGURES_DIR / "outlier_averaged_beats.png", dpi=150)
plt.close(fig)
print(f"\nSaved figure -> {FIGURES_DIR / 'outlier_averaged_beats.png'}")

# summary
n_ok = sum(1 for r in rows if r["status"] == "OK")
n_rev = sum(1 for r in rows if r["status"] == "NEEDS_REVIEW")
n_fp = sum(1 for r in rows if r["status"] == "FAILED_PEAKS")
n_f = sum(1 for r in rows if r["status"] == "FAILED")
print(f"\nSummary of 7 outliers: OK={n_ok}  NEEDS_REVIEW={n_rev}  FAILED_PEAKS={n_fp}  FAILED={n_f}")
