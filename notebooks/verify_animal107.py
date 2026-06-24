"""
Pipeline verification figure for Animal 107 — 5 panels:
  1. Raw signal with detected peaks
  2. After bandpass only (0.5-150 Hz) with peaks
  3. After bandpass + notch (50/100/150 Hz) with peaks
  4. Single beat zoom: raw vs clean overlaid, all waves labelled
  5. Overlay of 20 individual beats from clean signal
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

# ── project root ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
while not (ROOT / "data").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "outputs" / "figures"

# ── pipeline constants (identical to NB02) ────────────────────────────────
FS               = 1000
BANDPASS_LOW_HZ  = 0.5
BANDPASS_HIGH_HZ = 150.0
BANDPASS_ORDER   = 2
REFRACTORY_MS    = 55
PRE_R_MS         = 100
POST_R_MS        = 150
HR_ACCEPT_LOW    = 300
HR_ACCEPT_HIGH   = 700

ANIMAL_ID        = 107
WINDOW_START     = 1_395_000   # confirmed fallback window from features_fallback.csv
WINDOW_END       = 1_425_000

# ── I/O helpers ───────────────────────────────────────────────────────────
_MARKER_RE = re.compile(r"#([*1-3])")

def parse_header_and_layout(path):
    info = {}; n_header = 0; first_data_row = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.lstrip()
            if not s or s[0].isdigit() or s[0] in "+-.":
                first_data_row = line.rstrip("\n"); break
            n_header += 1
            if "=" in line:
                key, _, rest = line.partition("=")
                info[key.strip()] = rest.strip("\n").strip("\t").split("\t")
    titles = [t.strip() for t in info.get("ChannelTitle", []) if t.strip()]
    n_channels = max(len(titles), 1)
    n_cols = len(first_data_row.split("\t")) if first_data_row else (n_channels + 1)
    lead_cols = max(1, n_cols - n_channels)
    ch3_idx = next((i for i, t in enumerate(titles) if t.lower() == "channel 3"), 0)
    return n_header, lead_cols + ch3_idx


def load_ecg_file(path, n_header, ecg_col):
    voltages = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(n_header): f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= ecg_col: continue
            try: v = float(parts[ecg_col])
            except ValueError: v = np.nan
            voltages.append(v)
    voltage = np.asarray(voltages, dtype=float)
    if np.isnan(voltage).any():
        idx = np.arange(len(voltage)); good = ~np.isnan(voltage)
        if good.any(): voltage = np.interp(idx, idx[good], voltage[good])
    return voltage

# ── filtering stages ──────────────────────────────────────────────────────

def apply_bandpass(x):
    nyq = 0.5 * FS
    b, a = butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ / nyq, BANDPASS_HIGH_HZ / nyq], btype="band")
    return filtfilt(b, a, x)


def apply_notch(x):
    y = x.copy()
    for f0 in (50, 100, 150):
        bn, an = iirnotch(f0 / (FS / 2), Q=30)
        y = filtfilt(bn, an, y)
    return y

# ── R-peak detection (NB02 MAD method) ───────────────────────────────────

def _peaks_with_mad(sig):
    med = float(np.median(sig)); mad = float(np.median(np.abs(sig - med)))
    if mad <= 0: return np.array([], dtype=int), np.nan
    height = med + 4 * mad; prominence = max(0.3 * mad, 0.005)
    distance = int(REFRACTORY_MS * FS / 1000)
    peaks, _ = find_peaks(sig, height=height, distance=distance, prominence=prominence)
    return peaks, height


def detect_peaks(sig):
    """Try upright, then inverted. Return (peaks, inverted, hr_bpm, threshold)."""
    def hr(p): return 60000.0 / np.mean(np.diff(p) * 1000.0 / FS) if len(p) >= 2 else np.nan
    pp, ph = _peaks_with_mad(sig); ph_hr = hr(pp)
    if not np.isnan(ph_hr) and HR_ACCEPT_LOW <= ph_hr <= HR_ACCEPT_HIGH:
        return pp, False, ph_hr, ph
    ip, ih = _peaks_with_mad(-sig); ih_hr = hr(ip)
    if not np.isnan(ih_hr) and HR_ACCEPT_LOW <= ih_hr <= HR_ACCEPT_HIGH:
        return ip, True, ih_hr, ih
    def dist(h):
        if np.isnan(h): return np.inf
        return max(0, HR_ACCEPT_LOW - h, h - HR_ACCEPT_HIGH)
    if dist(ih_hr) < dist(ph_hr): return ip, True, ih_hr, ih
    return pp, False, ph_hr, ph

# ── beat extraction ───────────────────────────────────────────────────────

def extract_beats(sig, peaks, n=20):
    pre = int(PRE_R_MS * FS / 1000); post = int(POST_R_MS * FS / 1000)
    t_ms = (np.arange(pre + post) - pre) * (1000.0 / FS)
    beats = [sig[r - pre: r + post] for r in peaks
             if r - pre >= 0 and r + post <= len(sig)]
    beats = np.vstack(beats) if beats else np.empty((0, pre + post))
    return beats[:n], t_ms


def averaged_beat(beats):
    return beats.mean(axis=0) if len(beats) > 0 else None

# ── wave annotations ──────────────────────────────────────────────────────
# Wave windows: (lo_ms, hi_ms, polarity)
# polarity: +1 = look for positive peak, -1 = look for negative trough
WAVE_DEFS = {
    "P":  (-55, -15, +1),   # P-wave: small positive deflection before QRS
    "Q":  (-10,  -1, -1),   # Q: small negative notch just before R (exclude 0)
    "R":  ( -5,  +5, +1),   # R-peak: tallest positive spike
    "S":  (  1,  12, -1),   # S: small negative notch just after R (exclude 0)
    "J":  ( 12,  35, +1),   # J-wave: positive deflection after S in mouse ECG
    "T":  ( 40,  90, +1),   # T-wave: broad positive deflection
}
WAVE_COLORS = {
    "P": "#4e79a7",
    "Q": "#f28e2b",
    "R": "#dc2626",
    "S": "#9333ea",
    "J": "#0891b2",
    "T": "#059669",
}


def wave_extremum(beat, t_ms, lo, hi, polarity=+1):
    """Return (global_index, time_ms) of the max (+1) or min (-1) in [lo,hi] ms window."""
    mask = (t_ms >= lo) & (t_ms <= hi)
    if not mask.any(): return None, None
    seg = beat[mask]; ts = t_ms[mask]
    k = int(np.argmax(seg)) if polarity >= 0 else int(np.argmin(seg))
    return int(np.where(mask)[0][k]), float(ts[k])

# ── main ──────────────────────────────────────────────────────────────────

path = next(DATA_DIR.glob(f"*_{ANIMAL_ID}.txt"), None)
if path is None:
    raise FileNotFoundError(f"No file found for animal {ANIMAL_ID} in {DATA_DIR}")
print(f"File: {path.name}")

n_header, ecg_col = parse_header_and_layout(path)
print(f"ECG column: {ecg_col}")

full_voltage = load_ecg_file(path, n_header, ecg_col)
print(f"Total samples: {len(full_voltage):,}  ({len(full_voltage)/FS:.1f} s)")

# Extract the known-good window
raw = full_voltage[WINDOW_START:WINDOW_END].copy()
t_sec = np.arange(len(raw)) / FS
print(f"Window: samples {WINDOW_START}-{WINDOW_END}  ({len(raw)/FS:.0f} s)")

# Three processing stages
bp_only  = apply_bandpass(raw)
bp_notch = apply_notch(bp_only)

# Peak detection at each stage
raw_peaks,  raw_inv,  raw_hr,  _ = detect_peaks(raw)
bp_peaks,   bp_inv,   bp_hr,   _ = detect_peaks(bp_only)
clean_peaks, clean_inv, clean_hr, _ = detect_peaks(bp_notch)

raw_sig   = -raw      if raw_inv   else raw
bp_sig    = -bp_only  if bp_inv    else bp_only
clean_sig = -bp_notch if clean_inv else bp_notch

print(f"\nHR at each stage:")
print(f"  Raw signal              : {raw_hr:.1f} bpm  (inverted={raw_inv})")
print(f"  After bandpass only     : {bp_hr:.1f} bpm  (inverted={bp_inv})")
print(f"  After bandpass + notch  : {clean_hr:.1f} bpm  (inverted={clean_inv})")

# Beat extractions
N_BEATS_OVERLAY = 20
clean_beats, t_ms = extract_beats(clean_sig, clean_peaks, n=200)
raw_beats,   _    = extract_beats(raw_sig,   raw_peaks,   n=200)
print(f"\nTotal clean beats extracted: {len(clean_beats)}")

# For panel 4: use best beat (closest to template)
template = averaged_beat(clean_beats)
raw_template = averaged_beat(raw_beats)

# ── figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 22))
fig.suptitle(f"Pipeline Verification — Animal {ANIMAL_ID}  "
             f"(window {WINDOW_START//FS}–{WINDOW_END//FS} s)",
             fontsize=15, fontweight="bold", y=0.98)

gs = fig.add_gridspec(5, 1, hspace=0.52, top=0.95, bottom=0.04,
                      left=0.07, right=0.97)

# ── Panel 1: Raw signal ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_sec, raw_sig, lw=0.6, color="#555", label="raw signal")
if len(raw_peaks):
    ax1.scatter(t_sec[raw_peaks], raw_sig[raw_peaks],
                s=18, color="crimson", zorder=5, label=f"R-peaks (n={len(raw_peaks)})")
ax1.set_title(f"Panel 1 — Raw signal  |  HR = {raw_hr:.1f} bpm  |  peaks = {len(raw_peaks)}",
              fontsize=11, loc="left")
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Voltage (mV)")
ax1.legend(fontsize=8, loc="upper right"); ax1.grid(alpha=0.25)
ax1.set_xlim(0, t_sec[-1])

# ── Panel 2: Bandpass only ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.plot(t_sec, bp_sig, lw=0.6, color="#2563eb", label="bandpass 0.5-150 Hz")
if len(bp_peaks):
    ax2.scatter(t_sec[bp_peaks], bp_sig[bp_peaks],
                s=18, color="crimson", zorder=5, label=f"R-peaks (n={len(bp_peaks)})")
ax2.set_title(f"Panel 2 — After bandpass (0.5-150 Hz)  |  HR = {bp_hr:.1f} bpm  |  peaks = {len(bp_peaks)}",
              fontsize=11, loc="left")
ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Voltage (mV)")
ax2.legend(fontsize=8, loc="upper right"); ax2.grid(alpha=0.25)
ax2.set_xlim(0, t_sec[-1])

# ── Panel 3: Bandpass + notch ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.plot(t_sec, clean_sig, lw=0.6, color="#16a34a", label="bandpass + notch 50/100/150 Hz")
if len(clean_peaks):
    ax3.scatter(t_sec[clean_peaks], clean_sig[clean_peaks],
                s=18, color="crimson", zorder=5, label=f"R-peaks (n={len(clean_peaks)})")
ax3.set_title(f"Panel 3 — After bandpass + notch  |  HR = {clean_hr:.1f} bpm  |  peaks = {len(clean_peaks)}",
              fontsize=11, loc="left")
ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Voltage (mV)")
ax3.legend(fontsize=8, loc="upper right"); ax3.grid(alpha=0.25)
ax3.set_xlim(0, t_sec[-1])

# ── Panel 4: Single beat zoom — raw vs clean overlaid ────────────────────
ax4 = fig.add_subplot(gs[3])

if raw_template is not None and template is not None:
    r_idx_clean = int(np.argmax(template))

    ax4.plot(t_ms, raw_template, lw=1.4, color="#aaa", alpha=0.85,
             label="raw (averaged)", zorder=2, linestyle="--")
    ax4.plot(t_ms, template, lw=2.2, color="#1d4ed8",
             label="clean (bandpass+notch)", zorder=3)

    # vertical guide at R = 0 ms
    ax4.axvline(0, color="crimson", lw=0.9, linestyle=":", alpha=0.5)
    ax4.axhline(0, color="k", lw=0.4, linestyle="--", alpha=0.35)

    # Annotate each wave
    y_range = template.max() - template.min()
    # stagger text heights to prevent overlap
    stagger = {
        "P": +0.22, "Q": -0.22, "R": +0.18,
        "S": -0.22, "J": +0.14, "T": +0.10,
    }
    for wave, (lo, hi, pol) in WAVE_DEFS.items():
        color = WAVE_COLORS[wave]
        # shaded region
        ax4.axvspan(lo, hi, alpha=0.08, color=color, zorder=1)
        # wave peak/trough on clean template
        wi, wt = wave_extremum(template, t_ms, lo, hi, pol)
        if wi is not None:
            wa = template[wi]
            ax4.scatter([wt], [wa], s=70, color=color, zorder=6, edgecolors="white", lw=0.6)
            dy = stagger.get(wave, 0.15) * y_range
            ax4.annotate(
                f"{wave}\n{wt:.0f} ms\n{wa:.3f} mV",
                xy=(wt, wa),
                xytext=(wt, wa + dy),
                fontsize=8, color=color, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.7),
            )

    ax4.set_xlim(-PRE_R_MS, POST_R_MS)
    ax4.set_xlabel("Time relative to R-peak (ms)")
    ax4.set_ylabel("Voltage (mV)")
    ax4.set_title("Panel 4 — Single beat zoom: raw (dashed) vs clean (blue) overlaid  |  "
                  "Wave annotations on clean template", fontsize=11, loc="left")
    ax4.legend(fontsize=8, loc="upper right"); ax4.grid(alpha=0.25)
else:
    ax4.text(0.5, 0.5, "Not enough beats to average", ha="center", va="center",
             transform=ax4.transAxes, fontsize=12)

# ── Panel 5: 20-beat overlay ──────────────────────────────────────────────
ax5 = fig.add_subplot(gs[4])

if template is not None and len(clean_beats) > 0:
    # Filter beats by Pearson correlation with the template — keep r >= 0.6
    corrs = np.array([
        float(np.corrcoef(template, b)[0, 1]) for b in clean_beats
    ])
    good_mask = corrs >= 0.25
    good_beats = clean_beats[good_mask]
    good_corrs = corrs[good_mask]

    n_overlay = min(N_BEATS_OVERLAY, len(good_beats))
    # Pick evenly-spaced indices from good beats
    indices = np.round(np.linspace(0, len(good_beats)-1, max(n_overlay,1))).astype(int)
    selected = good_beats[indices]

    cmap_vals = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected)))
    for i, beat in enumerate(selected):
        ax5.plot(t_ms, beat, lw=0.9, color=cmap_vals[i], alpha=0.80)

    ax5.plot(t_ms, template, lw=2.5, color="black", label="mean template", zorder=5)
    ax5.axvline(0, color="crimson", lw=0.9, linestyle=":", alpha=0.7, label="R-peak (0 ms)")
    ax5.axhline(0, color="k", lw=0.4, linestyle="--", alpha=0.35)
    ax5.set_xlim(-PRE_R_MS, POST_R_MS)
    ax5.set_xlabel("Time relative to R-peak (ms)")
    ax5.set_ylabel("Voltage (mV)")
    ax5.set_title(
        f"Panel 5 — Overlay of {len(selected)} individual beats (r >= 0.25 with template, "
        f"from {good_mask.sum()}/{len(clean_beats)} total)  |  Black = mean template",
        fontsize=11, loc="left")
    ax5.legend(fontsize=8, loc="upper right"); ax5.grid(alpha=0.25)
else:
    ax5.text(0.5, 0.5, "Not enough clean beats", ha="center", va="center",
             transform=ax5.transAxes, fontsize=12)

out_path = FIG_DIR / "verify_animal107.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")

# ── console summary ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("WAVE AMPLITUDES (clean averaged template)")
print("="*60)
if template is not None:
    for wave, (lo, hi, pol) in WAVE_DEFS.items():
        wi, wt = wave_extremum(template, t_ms, lo, hi, pol)
        sign = "max" if pol >= 0 else "min"
        if wi is not None:
            print(f"  {wave:2s}  ({sign})  :  {template[wi]:+.4f} mV  @ {wt:.1f} ms")
        else:
            print(f"  {wave:2s}         :  not detected in [{lo}, {hi}] ms")

print(f"\n  Total beats in window  : {len(clean_beats)}")
if template is not None and len(clean_beats) > 0:
    print(f"  Beats r>=0.25 template : {good_mask.sum()}  ({100*good_mask.mean():.0f}%)")
    print(f"  Mean correlation       : {corrs.mean():.3f}  (SD={corrs.std():.3f})")
    print(f"  Beats shown panel 5    : {len(selected)}")
