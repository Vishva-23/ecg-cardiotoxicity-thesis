"""
Full-recording quality scan for the 3 unrescued outliers (102, 107, 112).

For each: slide a 30 s window across the ENTIRE recording in 5 s steps; for every
window compute HR, rr_std and 50/100/150 Hz mains-to-cardiac ratio using the
committed NB02 pipeline functions (reused verbatim). A window "passes" if
HR in [350,700], rr_std <= 120, n_beats >= 10 (same gate as the constrained audit).

Purpose: decide WITH Roisin whether 102/107/112 are recoverable-but-mislabelled
or genuinely unusable recordings. Honest: if a recording has ZERO passing windows
anywhere, that is stated plainly.

Outputs (NOT committed):
  outputs/figures/timeline_102.png, timeline_107.png, timeline_112.png
  outputs/recording_quality_scan.csv   (1 row per window, all 3 animals)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

NB = "02_beat_averaging_and_clustering.ipynb"

TARGETS = {102: "2024_11_01_102.txt", 107: "2024_11_01_107.txt", 112: "2024_11_13_112.txt"}

WINDOW_WIDTH_S  = 30
STEP_S          = 5
HR_GATE_LOW     = 350
HR_GATE_HIGH    = 700
RR_STD_CAP      = 120
NEAR_RADIUS_S   = 120     # "near the annotation" = within this of annotated start

# ---- reuse pipeline ----
nb = json.load(open(NB, encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
P = {"__name__": "__pipeline__"}
exec(compile(next(s for s in codes if s.startswith("import re")), "setup", "exec"), P)
exec(compile(next(s for s in codes if "def extract_morphology_features" in s), "helpers", "exec"), P)
FS = P["FS"]; DATA_DIR = P["DATA_DIR"]; OUTPUTS_DIR = P["OUTPUTS_DIR"]; FIGURES_DIR = P["FIGURES_DIR"]
parse_header_and_layout = P["parse_header_and_layout"]; load_ecg_file = P["load_ecg_file"]
find_baseline_window = P["find_baseline_window"]; bandpass_filter = P["bandpass_filter"]
detect_r_peaks = P["detect_r_peaks"]
WIN = int(WINDOW_WIDTH_S * FS); STEP = int(STEP_S * FS)
_integrate = getattr(np, "trapezoid", np.trapz)


def eval_window(segment):
    if segment.size < 2 * FS:
        return np.nan, np.nan, 0
    filtered = bandpass_filter(segment)
    peaks, inverted, hr, _ = detect_r_peaks(filtered)
    rr_std = float(np.std(np.diff(peaks) * (1000.0 / FS), ddof=1)) if len(peaks) > 1 else np.nan
    return (float(hr) if not np.isnan(hr) else np.nan), rr_std, int(len(peaks))


def mains_ratio(seg):
    f, pxx = welch(seg, fs=FS, nperseg=min(len(seg), 4096))
    def band(lo, hi):
        m = (f >= lo) & (f <= hi)
        return float(_integrate(pxx[m], f[m])) if m.any() else 0.0
    mains = band(49, 51) + band(99, 101) + band(149, 151)
    cardiac = band(4, 45)
    return mains / cardiac if cardiac > 0 else np.inf


all_rows = []
for aid, fname in TARGETS.items():
    path = DATA_DIR / fname
    n_header, lead_cols, ecg_col, titles = parse_header_and_layout(path)
    voltage, markers = load_ecg_file(path, n_header, ecg_col)
    n = len(voltage)
    rec_len_min = n / FS / 60.0
    a_start, a_end, rule, s_text, e_text = find_baseline_window(markers, n)
    ann_start_s = (a_start / FS) if a_start is not None else np.nan

    rows = []
    c = 0
    while c + WIN <= n:
        seg = voltage[c:c + WIN]
        hr, rr_std, nb_ = eval_window(seg)
        mr = mains_ratio(seg)
        passes = bool(nb_ >= 10 and not np.isnan(hr) and not np.isnan(rr_std)
                      and HR_GATE_LOW <= hr <= HR_GATE_HIGH and rr_std <= RR_STD_CAP)
        near = (not np.isnan(ann_start_s)) and abs(c / FS - ann_start_s) <= NEAR_RADIUS_S
        rows.append(dict(animal_id=aid, start_sample=c, start_s=round(c / FS, 2),
                         start_min=round(c / FS / 60, 3), end_s=round((c + WIN) / FS, 2),
                         hr=round(hr, 1) if not np.isnan(hr) else np.nan,
                         rr_std=round(rr_std, 1) if not np.isnan(rr_std) else np.nan,
                         n_beats=nb_, mains_ratio=round(mr, 4) if np.isfinite(mr) else np.nan,
                         passes_gate=passes, within_120s_of_annotation=bool(near)))
        c += STEP
    rdf = pd.DataFrame(rows)
    all_rows.append(rdf)

    # ---- report ----
    passing = rdf[rdf.passes_gate]
    near_pass = passing[passing.within_120s_of_annotation]
    far_pass = passing[~passing.within_120s_of_annotation]
    print("=" * 64)
    print(f"ANIMAL {aid}  ({fname})")
    print(f"  recording length     : {rec_len_min:.1f} min ({n:,} samples)")
    print(f"  annotation points at : {ann_start_s/60:.2f} min" if not np.isnan(ann_start_s)
          else "  annotation points at : (none found)")
    print(f"  windows scanned      : {len(rdf)}")
    print(f"  PASSING windows      : {len(passing)}")
    if len(passing) == 0:
        print("  >> ZERO passing windows anywhere in the recording — UNUSABLE by this gate.")
    else:
        print(f"     within +/-120s of annotation : {len(near_pass)}")
        print(f"     elsewhere in recording       : {len(far_pass)}")
        best = passing.sort_values(['rr_std', 'mains_ratio']).iloc[0]
        loc = "NEAR annotation" if best['within_120s_of_annotation'] else "FAR from annotation"
        print(f"  best window          : t={best['start_min']:.2f} min, HR={best['hr']:.0f}, "
              f"rr_std={best['rr_std']:.1f}, mains={best['mains_ratio']:.3f}  [{loc}]")
        # where are the passing windows?
        if len(near_pass) == 0:
            print("  >> NO passing window near the annotation; only usable data is elsewhere.")
        if len(far_pass) == 0:
            print("  >> all passing windows are near the annotation.")

    # ---- timeline figure: 3 stacked tracks ----
    t = rdf.start_min.values
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    gate_pass_t = rdf[rdf.passes_gate].start_min.values

    def shade(ax):
        for tp in gate_pass_t:
            ax.axvspan(tp, tp + WINDOW_WIDTH_S / 60, color="green", alpha=0.18, lw=0)
        if not np.isnan(ann_start_s):
            ax.axvline(ann_start_s / 60, color="red", linestyle="--", linewidth=1.4)

    axes[0].plot(t, rdf.hr.values, color="steelblue", linewidth=0.9)
    axes[0].axhspan(HR_GATE_LOW, HR_GATE_HIGH, color="green", alpha=0.06)
    axes[0].axhline(HR_GATE_LOW, color="green", lw=0.6); axes[0].axhline(HR_GATE_HIGH, color="green", lw=0.6)
    axes[0].set_ylabel("HR (bpm)"); axes[0].set_ylim(0, 800)
    shade(axes[0])
    axes[0].set_title(f"Animal {aid} — full-recording quality scan "
                      f"(green = window passes HR[{HR_GATE_LOW},{HR_GATE_HIGH}] & rr_std<={RR_STD_CAP}; "
                      f"red dashed = annotation)", fontsize=10)

    axes[1].plot(t, rdf.rr_std.values, color="darkorange", linewidth=0.9)
    axes[1].axhline(RR_STD_CAP, color="red", lw=0.8, linestyle=":")
    axes[1].set_ylabel("rr_std (ms)"); axes[1].set_ylim(0, min(800, np.nanmax(rdf.rr_std.values) * 1.1 + 1))
    shade(axes[1])

    axes[2].plot(t, rdf.mains_ratio.values, color="purple", linewidth=0.9)
    axes[2].set_ylabel("mains/cardiac"); axes[2].set_xlabel("recording time (min)")
    shade(axes[2])

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"timeline_{aid}.png", dpi=150)
    plt.close(fig)
    print(f"  saved figure -> figures/timeline_{aid}.png")

pd.concat(all_rows, ignore_index=True).to_csv(OUTPUTS_DIR / "recording_quality_scan.csv", index=False)
print("\nSaved per-window CSV -> outputs/recording_quality_scan.csv")
