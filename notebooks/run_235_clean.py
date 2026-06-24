"""
Animal 235 only — find a cleaner LATER window (light early anaesthesia caused
respiratory baseline wander; signal improves later). Reuses NB02 functions.
235 is NOT inverted (positive R) — so R peaks are detected positively, no flip.
Nothing committed.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

NB = "02_beat_averaging_and_clustering.ipynb"
FNAME = "2024_12_18_235.txt"
WINDOW_WIDTH_S, STEP_S, START_AFTER_S = 30, 5, 120     # search from 120 s onward
HR_LOW, HR_HIGH, HR_HIGH_RELAX = 350, 580, 620          # HR cap 580; relaxes to 620 only if needed
RR_STD_CAP, LF_CAP, STD_CAP = 120, 0.05, 0.5            # tighter low-freq 0.05; signal std < 0.5 mV
LF_HZ = 3.0          # respiratory / baseline-wander band: power below this / total
N_SHOW = 20

# ---- reuse pipeline ----
nb = json.load(open(NB, encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
P = {"__name__": "__p__"}
exec(compile(next(s for s in codes if s.startswith("import re")), "setup", "exec"), P)
exec(compile(next(s for s in codes if "def extract_morphology_features" in s), "helpers", "exec"), P)
FS = P["FS"]; DATA_DIR = P["DATA_DIR"]; OUTPUTS_DIR = P["OUTPUTS_DIR"]; FIGURES_DIR = P["FIGURES_DIR"]
parse_header_and_layout = P["parse_header_and_layout"]; load_ecg_file = P["load_ecg_file"]
bandpass_filter = P["bandpass_filter"]            # notch(50/100/150) + bandpass 0.5-150
detect_r_peaks = P["detect_r_peaks"]              # NB02 detector (with invert retry)
average_beats = P["average_beats"]
WIN, STEP = int(WINDOW_WIDTH_S * FS), int(STEP_S * FS)
_integrate = getattr(np, "trapezoid", np.trapz)


def low_freq_ratio(filt):
    """Fraction of power below LF_HZ (respiratory wander) over total power.
       (Power spectrum is identical for filt and -filt, so polarity is irrelevant.)"""
    f, pxx = welch(filt, fs=FS, nperseg=min(len(filt), 4096))
    tot = _integrate(pxx, f)
    lo = _integrate(pxx[f < LF_HZ], f[f < LF_HZ]) if (f < LF_HZ).any() else 0.0
    return float(lo / tot) if tot > 0 else np.nan


def window_metrics(seg):
    """Reuse NB02's detector; flip so beats are upright (235's recorded R is
       negative-going), but we do NOT call this 'inversion' in the figure."""
    filt = bandpass_filter(seg)
    pk, inverted, hr, _ = detect_r_peaks(filt)
    filt_up = -filt if inverted else filt
    rr_std = float(np.std(np.diff(pk) * (1000.0 / FS), ddof=1)) if len(pk) > 1 else np.nan
    return (filt_up, pk, (float(hr) if not np.isnan(hr) else np.nan),
            rr_std, low_freq_ratio(filt), float(np.std(filt)))


# ---- load ----
path = DATA_DIR / FNAME
nh, lc, ec, _t = parse_header_and_layout(path)
voltage, markers = load_ecg_file(path, nh, ec)
n = len(voltage)
print(f"Animal 235: {n/FS/60:.1f} min recording ({n:,} samples)")

# ---- STEP 1: scan from 120 s onward, collect every window's metrics ----
recs = []
c = START_AFTER_S * FS
while c + WIN <= n:
    filt, pk, hr, rr_std, lf, sd = window_metrics(voltage[c:c + WIN])
    recs.append(dict(start=c, hr=hr, rr_std=rr_std, lf=lf, std=sd, n=len(pk)))
    c += STEP


def passes(r, hr_high):
    return (r["n"] >= 10 and not np.isnan(r["hr"]) and HR_LOW <= r["hr"] <= hr_high
            and r["rr_std"] <= RR_STD_CAP and r["lf"] < LF_CAP and r["std"] < STD_CAP)


full = [r for r in recs if passes(r, HR_HIGH)]
relaxed = [r for r in recs if passes(r, HR_HIGH_RELAX)]

print("\n--- STEP 1: scan from 120 s ---")
print(f"  criteria: HR in [{HR_LOW},{HR_HIGH}], rr_std<={RR_STD_CAP}, "
      f"low-freq<{LF_CAP}, signal std<{STD_CAP} mV")
print(f"  windows scanned: {len(recs)}   pass ALL (HR<=580): {len(full)}   "
      f"pass with HR cap relaxed to 620: {len(relaxed)}")

if full:
    best = min(full, key=lambda r: r["lf"]); mode = "all criteria met (HR cap 580)"
elif relaxed:
    best = min(relaxed, key=lambda r: r["lf"])
    mode = "HR cap RELAXED to 620 (no window passed with HR<=580)"
    print(f"  >> No window met HR<=580; relaxed HR cap to 620 as instructed.")
else:
    # nothing even with relaxed HR: report honestly, fall back to cleanest std+rr_std window
    avail = [r for r in recs if r["n"] >= 10 and not np.isnan(r["hr"])
             and HR_LOW <= r["hr"] <= HR_HIGH_RELAX and r["rr_std"] <= RR_STD_CAP and r["std"] < STD_CAP]
    if not avail:
        print("  >> No window meets even the relaxed criteria after 120 s — cannot export. Stopping.")
        raise SystemExit(0)
    best = min(avail, key=lambda r: r["lf"])
    mode = (f"BEST AVAILABLE — no window met low-freq<{LF_CAP} even with HR<=620; "
            f"chose lowest low-freq ({best['lf']:.4f}) among std/rr_std-valid windows")
    print(f"  >> {mode}")

# old window (0.6 min) metrics for comparison
old_start = int(0.6 * 60 * FS)
_, _, old_hr, old_rrstd, old_lf, old_std = window_metrics(voltage[old_start:old_start + WIN])

print(f"\n  CHOSEN WINDOW ({mode}):")
t_min = best["start"] / FS / 60.0
print(f"    start     = {t_min:.2f} min ({best['start']/FS:.0f} s)")
print(f"    HR        = {best['hr']:.0f} bpm")
print(f"    rr_std    = {best['rr_std']:.1f} ms")
print(f"    low-freq  = {best['lf']:.4f}")
print(f"    signal std= {best['std']:.4f} mV")
print(f"    n_beats   = {best['n']}")

# ---- STEP 2: export filtered clean window ----
(OUTPUTS_DIR / "clean_per_animal").mkdir(parents=True, exist_ok=True)
filt, pk, hr, rr_std, lf, sd = window_metrics(voltage[best["start"]:best["start"] + WIN])
t = np.arange(len(filt)) / FS
lines = ["time_s\tmv"]
for ti, mi in zip(t, filt):
    lines.append(f"{round(float(ti), 3)}\t{round(float(mi), 6)}")
out_txt = OUTPUTS_DIR / "clean_per_animal" / "clean_235_updated.txt"
with open(out_txt, "w", newline="") as fh:
    fh.write("\r\n".join(lines) + "\r\n")
print(f"\n--- STEP 2: exported {out_txt}  ({len(filt)} samples, tab/CRLF, time_s+mv) ---")

# ---- STEP 3: 20-beat waterfall (positive, NOT inverted) ----
template, t_ms, beats = average_beats(filt, pk)
n_beats = 0 if beats is None else len(beats)
idx = np.unique(np.linspace(0, n_beats - 1, min(N_SHOW, n_beats)).astype(int))
sel = beats[idx]
ptp = float(np.median([np.ptp(b) for b in sel])) or 1.0
step = 1.15 * ptp
fig, (axw, axt) = plt.subplots(1, 2, figsize=(13, 9), gridspec_kw={"width_ratios": [2.4, 1]})
for i, b in enumerate(sel):
    axw.plot(t_ms, b + i * step, color="steelblue", linewidth=0.9)
axw.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
axw.set_yticks([i * step for i in range(len(sel))])
axw.set_yticklabels([f"beat {j}" for j in idx], fontsize=6)
axw.set_xlabel("Time from R peak (ms)"); axw.set_title(f"{len(sel)} individual beats (waterfall)")
axw.grid(alpha=0.2, axis="x")
axt.plot(t_ms, template, color="crimson", linewidth=2.0)
axt.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
axt.set_xlabel("Time from R peak (ms)"); axt.set_ylabel("mV"); axt.set_title("averaged-beat template")
axt.grid(alpha=0.3)
fig.suptitle(f"Animal 235 — {hr:.0f} bpm — window @ {t_min:.1f} min "
             f"(respiratory artifact cleared) — n_beats={n_beats}", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGURES_DIR / "beats20_235_updated.png", dpi=150); plt.close(fig)
print(f"--- STEP 3: saved {FIGURES_DIR / 'beats20_235_updated.png'} (NOT labelled inverted) ---")

# ---- comparison ----
print("\n=== LOW-FREQUENCY (respiratory) IMPROVEMENT ===")
print(f"  OLD window @ 0.6 min : low-freq<3Hz ratio = {old_lf:.4f}  (HR {old_hr:.0f}, rr_std {old_rrstd:.1f})")
print(f"  NEW window @ {t_min:.1f} min: low-freq<3Hz ratio = {best['lf']:.4f}  (HR {best['hr']:.0f}, rr_std {best['rr_std']:.1f})")
if old_lf and not np.isnan(old_lf):
    print(f"  -> respiratory/baseline power reduced {old_lf/best['lf']:.1f}x "
          f"({old_lf*100:.1f}% -> {best['lf']*100:.1f}% of total power below 3 Hz)")
