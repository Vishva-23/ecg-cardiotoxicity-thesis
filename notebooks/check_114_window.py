"""Scan animal 114's signal around 68-69 s to measure the clean patch length."""
import json, sys
from pathlib import Path
import numpy as np

nb_path = Path(__file__).parent / "02_beat_averaging_and_clustering.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
P = {"__name__": "__pipeline__"}
exec(compile(next(s for s in codes if s.startswith("import re")), "setup", "exec"), P)
exec(compile(next(s for s in codes if "def extract_morphology_features" in s), "helpers", "exec"), P)

DATA_DIR = P["DATA_DIR"]
parse_header_and_layout = P["parse_header_and_layout"]
load_ecg_file           = P["load_ecg_file"]
bandpass_filter         = P["bandpass_filter"]
detect_r_peaks          = P["detect_r_peaks"]
FS = int(P["FS"])

path = DATA_DIR / "2024_11_114.txt"
n_header, _, ecg_col, _ = parse_header_and_layout(path)
voltage, _ = load_ecg_file(path, n_header, ecg_col)
print(f"Animal 114 total recording: {len(voltage)/FS:.1f} s  ({len(voltage):,} samples)")
print()

STEP = FS        # 1 s steps
WIN  = 5 * FS    # 5 s window

header = f"  {'start_s':>8}  {'end_s':>6}  {'HR_bpm':>7}  {'rr_std':>8}  {'n_beats':>7}  pass?"
print(header)
print("  " + "-" * 55)

passing_starts = []
for c in range(55 * FS, min(130 * FS, len(voltage) - WIN), STEP):
    seg  = voltage[c:c + WIN]
    filt = bandpass_filter(seg)
    peaks, inv, hr, _ = detect_r_peaks(filt)
    rr     = np.diff(peaks) * 1000.0 / FS
    rr_std = float(np.std(rr, ddof=1)) if len(rr) > 1 else float("nan")
    n      = len(peaks)
    ok     = (350 <= hr <= 700 and not (rr_std != rr_std) and rr_std <= 120 and n >= 5)
    if ok:
        passing_starts.append(c)
    flag = "PASS" if ok else ""
    print(f"  {c/FS:>8.1f}  {(c+WIN)/FS:>6.1f}  {hr:>7.1f}  {rr_std:>8.1f}  {n:>7}  {flag}")

print()
if passing_starts:
    run_start = passing_starts[0] / FS
    run_end   = (passing_starts[-1] + WIN) / FS
    span      = run_end - run_start
    print(f"Passing span: {run_start:.1f}s  to  {run_end:.1f}s  =  {span:.1f}s total")
    print(f"Passing windows: {len(passing_starts)}")
    # Check whether a 20s sub-window would fit
    for w in (20, 15, 10):
        can_fit = span >= w
        print(f"  >= {w}s sub-window fits: {can_fit}")
else:
    print("NO passing windows found in 55-130 s region.")
