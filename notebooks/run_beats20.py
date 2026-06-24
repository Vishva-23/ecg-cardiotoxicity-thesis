"""
TASK 2 — spread-out averaged-beat figures (~20 beats per panel) for the 7 outliers.

Uses the CLEANEST window from a full-recording scan (Roisin confirmed a different/
later window is fine; later windows under deeper anaesthesia are cleaner). Reuses
NB02 functions verbatim. One figure per animal: a vertically-offset "waterfall" of
~20 individual beats + the averaged-beat template alongside.

Deterministic: beats chosen by even spacing (np.linspace), no randomness.
Outputs (not committed): outputs/figures/beats20_<id>.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

NB = "02_beat_averaging_and_clustering.ipynb"
TARGETS = {102: "2024_11_01_102.txt", 107: "2024_11_01_107.txt", 112: "2024_11_13_112.txt",
           115: "2024_11_13_115.txt", 117: "2024_11_14_117.txt", 126: "2024_11_15_126.txt",
           235: "2024_12_18_235.txt"}
WINDOW_WIDTH_S, STEP_S = 30, 5
HR_GATE_LOW, HR_GATE_HIGH, RR_STD_CAP = 350, 700, 120
N_SHOW = 20

# reuse pipeline
nb = json.load(open(NB, encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
P = {"__name__": "__pipeline__"}
exec(compile(next(s for s in codes if s.startswith("import re")), "setup", "exec"), P)
exec(compile(next(s for s in codes if "def extract_morphology_features" in s), "helpers", "exec"), P)
FS = P["FS"]; DATA_DIR = P["DATA_DIR"]; FIGURES_DIR = P["FIGURES_DIR"]
parse_header_and_layout = P["parse_header_and_layout"]; load_ecg_file = P["load_ecg_file"]
bandpass_filter = P["bandpass_filter"]; detect_r_peaks = P["detect_r_peaks"]; average_beats = P["average_beats"]
WIN, STEP = int(WINDOW_WIDTH_S * FS), int(STEP_S * FS)
_integrate = getattr(np, "trapezoid", np.trapz)


def mains_ratio(seg):
    f, pxx = welch(seg, fs=FS, nperseg=min(len(seg), 4096))
    def band(lo, hi):
        m = (f >= lo) & (f <= hi)
        return float(_integrate(pxx[m], f[m])) if m.any() else 0.0
    c = band(4, 45)
    return (band(49, 51) + band(99, 101) + band(149, 151)) / c if c > 0 else np.inf


def best_clean_window(voltage):
    """Slide whole recording; return (start, end) of the lowest-rr_std passing window
       (tie-break: lowest mains ratio). Falls back to lowest-rr_std overall if none pass."""
    n = len(voltage); cands, fallback = [], []
    c = 0
    while c + WIN <= n:
        seg = voltage[c:c + WIN]
        filt = bandpass_filter(seg)
        peaks, inv, hr, _ = detect_r_peaks(filt)
        if len(peaks) > 1:
            rr_std = float(np.std(np.diff(peaks) * (1000.0 / FS), ddof=1))
            fallback.append((rr_std, c))
            if (len(peaks) >= 10 and not np.isnan(hr) and HR_GATE_LOW <= hr <= HR_GATE_HIGH
                    and rr_std <= RR_STD_CAP):
                cands.append((rr_std, mains_ratio(seg), c))
        c += STEP
    if cands:
        cands.sort(key=lambda t: (t[0], t[1]))
        cs = cands[0][2]; passed = True
    else:
        fallback.sort(); cs = fallback[0][1] if fallback else 0; passed = False
    return cs, cs + WIN, passed


for aid, fname in TARGETS.items():
    path = DATA_DIR / fname
    nh, lc, ec, _t = parse_header_and_layout(path)
    voltage, markers = load_ecg_file(path, nh, ec)
    cs, ce, passed = best_clean_window(voltage)
    seg = voltage[cs:ce]
    filt = bandpass_filter(seg)
    peaks, inverted, hr, _ = detect_r_peaks(filt)
    if inverted:
        filt = -filt
    template, t_ms, beats = average_beats(filt, peaks)
    win_min = cs / FS / 60.0

    if beats is None or getattr(beats, "ndim", 0) != 2 or len(beats) == 0:
        print(f"animal {aid}: no beats extracted in chosen window — skipped")
        continue

    # pick ~20 evenly spaced beats (deterministic)
    n_beats = len(beats)
    idx = np.unique(np.linspace(0, n_beats - 1, min(N_SHOW, n_beats)).astype(int))
    sel = beats[idx]

    # vertical offset so each beat is clearly separated
    ptp = float(np.median([np.ptp(b) for b in sel])) or 1.0
    step = 1.15 * ptp

    fig, (axw, axt) = plt.subplots(1, 2, figsize=(13, 9), gridspec_kw={"width_ratios": [2.4, 1]})

    for i, b in enumerate(sel):
        axw.plot(t_ms, b + i * step, color="steelblue", linewidth=0.9)
    axw.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
    axw.set_yticks([i * step for i in range(len(sel))])
    axw.set_yticklabels([f"beat {j}" for j in idx], fontsize=6)
    axw.set_xlabel("Time from R peak (ms)")
    axw.set_title(f"{len(sel)} individual beats (waterfall)")
    axw.grid(alpha=0.2, axis="x")

    axt.plot(t_ms, template, color="crimson", linewidth=2.0)
    axt.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
    axt.set_xlabel("Time from R peak (ms)"); axt.set_ylabel("mV")
    axt.set_title("averaged-beat template")
    axt.grid(alpha=0.3)

    inv_flag = "  | INVERTED" if inverted else ""
    src = "clean passing window" if passed else "best-available window (none passed gate)"
    fig.suptitle(f"Animal {aid} — {hr:.0f} bpm — window @ {win_min:.1f} min ({src}) — "
                 f"n_beats={n_beats}{inv_flag}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_DIR / f"beats20_{aid}.png", dpi=150)
    plt.close(fig)
    print(f"animal {aid}: window @ {win_min:5.1f} min, HR={hr:5.0f}, n_beats={n_beats}, "
          f"inverted={inverted}, passed_gate={passed}  -> beats20_{aid}.png")

print("\nSaved 7 figures: beats20_{102,107,112,115,117,126,235}.png")
