# The ECG pipeline — what it does, how it works, and what changed

A technical description of the analysis pipeline: what problem it solves, how each
stage works, and precisely which code changed relative to the earlier version.
Every number quoted is reproducible from the notebooks in [`../notebooks/`](../notebooks/).

---

## 1. What the pipeline does

It converts raw LabChart ECG recordings from anaesthetised mice into a table of
per-animal cardiac measurements suitable for statistical comparison between
treatment groups.

```
LabChart .txt  ->  locate baseline window  ->  filter  ->  detect R-peaks
               ->  average beats into a template  ->  extract features
               ->  quality-gate  ->  feature table  ->  multivariate analysis
```

Mouse ECG is not simply a fast human ECG, and that drives most of the design:

| property | human | mouse | consequence for the code |
|---|---|---|---|
| R-peak amplitude | 1–2 mV | ~0.2–0.7 mV | standard detectors miss beats; needs an adaptive threshold |
| Heart rate | 60–100 bpm | 300–700 bpm | beats are ~120 ms apart, so search windows must be short |
| Filter band | up to ~40 Hz | up to **150 Hz** | the R-spike is sharp; a 40 Hz cutoff would flatten it |
| ST segment | flat | **absent** | a J-wave follows R immediately; repolarisation starts at once |
| T-wave | upright | often **inverted** | T is found as a trough, not a peak |

Because of the amplitude problem, NeuroKit2's five standard R-peak algorithms all
failed on this data, which is why the pipeline uses its own detector.

---

## 2. How it works, stage by stage

### 2.1 Locate the analysis window
`parse_header_and_layout` reads the LabChart header (sampling rate, channel
layout); `load_ecg_file` loads the voltage trace and its annotation markers;
`find_baseline_window` finds the baseline segment from those markers, skipping
non-baseline protocol blocks (`jugular`, `injection`, `oc 1`…).

Windows range from **11 to 88 seconds**. Recordings with no usable markers are
excluded rather than guessed at (see §4, Animal 110).

### 2.2 Filter — `bandpass_filter`
Butterworth band-pass **0.5–150 Hz** (order 2, zero-phase `filtfilt`) plus a notch
cascade at **50 / 100 / 150 Hz**.

- The 0.5 Hz high-pass removes baseline drift.
- The 150 Hz low-pass preserves the sharp mouse R-spike.
- The notch removes mains hum, which is the dominant interference.

This was chosen over wavelet and EMD denoising (Notebook 03) because the noise is
stationary and narrow-band — exactly what a notch handles — while wavelet/EMD
target non-stationary noise that isn't present here. Verified in the frequency
domain: the notch removes **~90%** of 50 Hz power on clean recordings while leaving
the cardiac harmonic comb intact.

### 2.3 Detect beats — `detect_r_peaks`
An adaptive threshold of **median + 4 × MAD** with a **55 ms refractory period**
(the shortest physiologically plausible RR at mouse rates). MAD rather than
standard deviation makes the threshold robust to artifact spikes. Polarity is
detected automatically and normalised, so recordings with reversed electrodes are
handled rather than discarded.

### 2.4 Average beats — `average_beats`
Beats are extracted in a window of **−100 ms to +150 ms** around each R-peak and
averaged into a single template per animal. Averaging suppresses random noise,
raising signal-to-noise for the small P and T deflections. The individual beats are
retained so beat-to-beat variability (`qt_std_ms`, `qrs_std_ms`) can be reported.

### 2.5 Extract features — `extract_morphology_features`
Landmarks are located in physiologically bounded windows relative to R:

| feature | how it is measured |
|---|---|
| R amplitude | peak-to-peak across the QRS window (−15…+35 ms) |
| QRS duration | full width at half maximum of the R-spike |
| P amplitude / PR | positive peak in −70…−35 ms, above a noise floor |
| Q / S amplitude | minimum in −14…−2 ms / +8…+32 ms |
| J-wave | maximum in +2…+18 ms |
| T amplitude | trough after S (inverted in mouse) |
| QT | Q-onset to T-wave return to baseline, **bounded by the RR interval** |
| QTc | Mitchell formula: `QT / sqrt(RR/100)` |

### 2.6 Quality gating — informative missingness
The pipeline does **not** median-fill missing values, because a fabricated value is
indistinguishable from a real one downstream. Instead a value is left `NaN` and a
**cause-flag** records *why*:

| rule | trigger | what is nulled |
|---|---|---|
| Rule A | `rr_cv > 0.15` (irregular rhythm) | `rr_mean_ms`, `qt_ms`, `qtc_ms` |
| Rule B | T-wave excursion `< 0.005 mV` (flat T) | T-dependent measures |

Four cause-flags (`beats_detected`, `rhythm_regular`, `twave_isolated`,
`enough_beats_for_sd`) enter the analysis as features in their own right, so the
*pattern of missingness* is analysable rather than hidden.

### 2.7 Multivariate analysis
Continuous features are standardised; cause-flags enter unscaled. PCA reduces the
matrix, and clustering is run three ways (k-means, hierarchical Ward, Gaussian
mixture) at k = 2…6, with 3D PCA and a nonlinear t-SNE embedding as cross-checks.

---

## 3. What changed relative to the old pipeline

### 3.1 QT interval — was over-estimated by 26–53% ✅ fixed

**The bug.** QT ends where the T-wave returns to baseline. The search for that
point was unbounded. At ~500 bpm the RR interval is only ~120 ms, so the search ran
**past the end of the beat and into the next one**, latching onto the *following
beat's P-wave* — which also sits near baseline. QT was therefore measured to a
landmark belonging to the next cardiac cycle.

**The fix.** The endpoint search is bounded by the RR interval, so it cannot cross
into the next beat (`_qt_per_beat(beat, t_ms, rr_ms=None)` — "RR-bounded and
polarity-aware").

**Effect:**

| | old | new |
|---|---|---|
| median QT | 72.7 ms | **43.9 ms** |
| median QTc | 68.5 ms | **41.5 ms** |

The corrected QTc lands on the published mouse value (~41 ms). That is external
validation: nothing in the pipeline was tuned to produce it.

### 3.2 R amplitude — was under-estimated by ~31% ✅ fixed

**The bug.** R amplitude was measured from zero to the R peak, so it ignored the
S-wave trough and reported only part of the deflection.

**The fix.** Measure the full **peak-to-peak** deflection across the QRS window,
and average across individual beats rather than reading the template alone:

```python
_qrs_w = (t_ms >= -15) & (t_ms <= 35)
r_amp  = float(np.max(sig[_qrs_w]) - np.min(sig[_qrs_w]))
```

**Effect.** Mean error against the manual LabChart cursor measurements fell from
**≈32.5% to ≈13.9%**. QRS duration and QT were confirmed unchanged — the edit is
isolated to amplitude. The residual ~14% is attributed to the 150 Hz low-pass
slightly rounding the sharp R-spike, which is a deliberate filter trade-off.

### 3.3 New waveform features ➕ added
`p_wave_amplitude_mv`, `pr_interval_ms`, `q_wave_amplitude_mv`,
`s_wave_amplitude_mv` — previously not extracted at all. These cover the remaining
parameters on the study's ECG parameter list (atrial depolarisation and
atrio-ventricular conduction). They return `NaN` rather than a guess on recordings
too noisy to locate a P-wave (38 of 117).

### 3.4 New validation and analysis capability ➕ added

| addition | notebook | purpose |
|---|---|---|
| Raw → filtered → feature overlay | NB04 | visual proof that preprocessing and extraction work |
| FFT power spectrum, before vs after | NB03 | proves the filter removes mains hum and keeps cardiac harmonics |
| 3D PCA + t-SNE | NB02 | tests whether structure exists that 2D linear PCA might hide |
| Problem-animal figures + triage | NB02 | classifies the 34 flagged recordings keep / review / drop |
| 10-animal validation gallery | NB02 | the primary evidence that extraction is correct |

### 3.5 Reproducibility fixes 🔧
- **NB03** used `METHOD_NAMES` in one cell before defining it in the next, so a
  clean *Restart & Run All* failed. The constant is now defined where it is first
  used (identical value; no behavioural change).
- **NB01** reads a CSV that it writes in a later cell, so it only ran if the file
  already existed. Documented; not reordered, because that would change the
  author's intended narrative order.

### 3.6 What did **not** change
The filter design, the MAD-adaptive detector, the beat-averaging window, the
informative-missingness rules and the cause-flags are unchanged. The corrections
above affect *measurement*, not the pipeline's architecture or its quality logic.

---

## 4. The pipeline as it stands now

Processing 118 recordings:

| outcome | n |
|---|---|
| `OK` | 111 |
| `NEEDS_REVIEW` | 6 |
| `FAILED_ANNOTATION` | **1** (Animal 110 — no baseline markers in the file) |
| **entering analysis** | **117** |

Of those 117: **83 clean**, **34 flagged** (all by rhythm irregularity; Rule B fires
on zero animals).

Current cohort values:

| measure | value |
|---|---|
| heart rate | median 520 bpm (range 93–686) |
| QT | median **43.9 ms** |
| QTc | median **41.5 ms** |
| QRS (R-FWHM) | median 6 ms |
| rr_cv, clean animals | median 0.025 |

**Validation gallery — the ten cleanest recordings:**
**QTc 41.6 ± 0.9 ms** and **QT 44.2 ± 1.0 ms** across ten independent animals,
against a mouse literature QTc of ~41 ms. Ten separate animals agreeing to under
1 ms, with nothing in the pipeline calibrated to that target, is the strongest
single check that extraction is correct.

---

## 5. Open items (decisions, not defects)

1. **`qrs_duration_ms` measures the R-spike width (FWHM), not clinical QRS
   duration** (Q-onset → S-offset). It reads ~6 ms where a true QRS is nearer
   10–29 ms. Because anthracycline cardiotoxicity *widens* the QRS, the definition
   matters — and QRS offset is genuinely ambiguous in mouse ECG, where the J-wave
   runs on with no clean isoelectric point. **A definition is needed from Roisin.**
2. **R-amplitude convention.** The pipeline measures peak-to-peak, which matched
   the manual cursor measurements far better than baseline-to-peak (32.5% → 13.9%
   error). Worth confirming this is the intended convention.
3. **Animal 110** can be recovered if baseline annotation markers are added in
   LabChart.
4. **The 6 REVIEW animals** have irregular rhythm but physiologically plausible
   heart rates, so they may represent **real rhythm variability rather than
   artifact** — a physiological judgement.
5. **Treatment-group metadata** is still withheld, so no supervised comparison
   between treatment arms has been run. See
   [why_supervised_is_required.md](why_supervised_is_required.md).
