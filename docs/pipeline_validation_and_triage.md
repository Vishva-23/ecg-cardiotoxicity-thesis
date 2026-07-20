# ECG pipeline — validation, problem-animal triage, and open questions

**Status report for the supervisor meeting.** Consolidates the checks run on the
pipeline: confirmation that it works on the clean animals, the filter/FFT
verification, a triage of the 34 flagged animals, a recovery analysis, and
point-by-point answers to the open questions. Every number below is reproducible
from the notebooks and the scripts in `code/`.

---

## Executive summary

- **The pipeline works on the clean animals.** 83 of 117 recordings are clean;
  heart rate, rhythm regularity, and QTc are all physiological. Beat detection is
  confirmed correct.
- **The filter is verified.** It removes the 50 Hz mains oscillation (Luke's
  "oscillation across the trace") and out-of-band noise while preserving the
  heartbeat — shown in both the time domain and the FFT power spectrum.
- **The 34 "problem" animals are a gradient, not a wall:** ~6 are essentially
  clean, 1–2 are recoverable by relocating the analysis window, ~16 are irregular
  *throughout* the recording (a physiological question, not a data problem), and
  ~10 are genuinely too noisy to use.
- **Two items are for the supervisors, not for data processing:** whether the
  irregular-throughout animals reflect real arrhythmia (Roisin), and the scope
  question of re-examining every recording from scratch (Luke/Roisin).

---

## 1. The pipeline works on the clean animals (the headline)

The proper test of the pipeline is the clean animals, not the known problem cases.

| metric | result | verdict |
|---|---|---|
| clean animals | **83 of 117** | the majority |
| heart rate | 385–626 bpm (median **530**) | all inside the 300–700 physiological gate |
| rhythm (rr_cv) | median **0.025** | very regular — locking onto real beats |
| QTc | median **41.5 ms** | matches mouse literature (~41 ms) |

Beat detection on a clean animal is shown in the validation overlay
(`overlay_validation_A201.png`): every beat is detected, and the averaged-beat
template shows clean P / QRS / J-wave / T morphology.

## 2. Filter verification (FFT power spectrum)

Two weeks ago Luke saw a visible oscillation running through the whole trace.
That oscillation is **50 Hz mains hum**. Re-running the FFT with the current
settings confirms it is removed:

- **50 Hz mains: ~90% removed** on clean animals, up to 98.6% on noisy ones.
- Everything above **150 Hz** is cut by the bandpass.
- The evenly-spaced comb of peaks that remains is **not noise** — it is the
  heartbeat's harmonics (HR fundamental ~8.3 Hz = 499 bpm, plus overtones). A
  periodic heartbeat must produce that comb.

Overlaying raw vs filtered in the time domain confirms the same thing: on clean
animals the two track each other closely (correlation 0.93–0.97) and only
~26–39% of the signal is removed (noise + drift), while the heartbeat shape is
preserved.

## 3. The small bumps on clean animals are real, not noise

Between the R-peak and baseline on a clean beat there are small deflections.
These are **genuine cardiac features**, not residual noise: the **P wave**, the
**J-wave** (prominent in mice, immediately after R), and the **inverted T-wave**.
Mice have no flat ST segment, so repolarisation begins almost immediately. These
should be explained, not filtered away.

## 4. Measurement note — the QRS-duration field

The current `qrs_duration_ms` measures the **FWHM of the R-spike** (~6 ms), not
the clinical QRS duration (Q-onset → S-offset, ~10–29 ms). It systematically
under-reads the true QRS width. Because anthracycline cardiotoxicity *widens* the
QRS, the definition matters — **this is flagged for Roisin to decide the QRS-offset
definition** (genuinely ambiguous in mouse ECG, where the J-wave runs straight on
with no clean isoelectric point). Interim: rename the field to `r_width_fwhm_ms`
so it is not mistaken for QRS duration.

## 5. Triage of the 34 flagged animals

The 34 flagged recordings span a gradient, measured by raw-vs-filter correlation
and the fraction of the signal that is noise:

| group | count | criterion | action |
|---|---|---|---|
| 🟢 barely-flagged (essentially clean) | 6 | corr 0.95–0.99, 17–32% removed | keep via a two-tier gate |
| ✅ recoverable by window relocation | 1–2 | a cleaner window exists in the recording | move to clean set after review |
| 🟡 irregular throughout | ~16 | irregularity spread across the whole recording | **physiological question — Roisin's call** |
| 🔴 severe / mostly noise | ~10 | corr < 0.75 and/or > 80% of signal is noise | drop |

**Barely-flagged (6):** 117, 129, 131, 134, 139, 245. Filter behaves perfectly;
they only tripped the `rr_cv > 0.15` gate on mild irregularity.

**Severe (≈10):** 102, 103, 107, 109, 113, 115, 120, 125, 127, 240. Example —
Animal 120: raw and filtered barely track (corr 0.39), 97% of the trace is noise,
HR 93 bpm (impossible for an anaesthetised mouse), polarity inverted. Unusable.

## 6. Recovery analysis (what "recovery" can and cannot do)

Each LabChart file is ~29 minutes but only a short window is measured. If that
window landed on a noisy stretch, relocating to the cleanest window in the same
recording (`quality_window_search`) can recover usable beats. Result across the
18 middle-group animals:

- **1 clean recovery: Animal 123** — rr_cv 0.220 → 0.089, HR 503 → 502 bpm
  (consistent), now below the 0.15 gate.
- **1 borderline: Animal 126** — a clean window exists (rr_cv 0.585 → 0.047) but
  HR shifts 452 → 559 bpm, so it needs manual review.
- **16 of 18: irregular throughout.** Relocating to the cleanest window does
  **not** fix the irregularity — meaning it is intrinsic to the animal's rhythm,
  not a windowing artifact.

**This is the key finding:** when relocation cannot fix the irregularity, the
cause is either **real physiology** (sinus arrhythmia / tachycardia / bradycardia
— a finding, and Roisin's call) or a **genuinely broken recording**. It cannot be
recovered by data processing, and that is the correct, defensible conclusion.

**How each recovery is checked against the original:**
1. **Visual overlay** — detected R-peaks sit on real raw beats (real beats?).
2. **Physiological sanity** — HR 300–700 bpm, rr_cv < 0.15, QTc ~41 ms.
3. **HR stability** — recovered HR matches the original HR (same animal, not a
   spurious signal). This is the decisive check.
4. **Manual cross-check** — against LabChart cursor values where available.

## 7. Point-by-point answers to the open questions

| # | question | answer |
|---|---|---|
| 1 | Do the FFT spectra look better with the new settings? | **Yes** — 50 Hz mains removed (~90%), out-of-band noise cut, cardiac harmonics preserved. |
| 2 | Problem animals 111/113 — placement? baseline? re-examine from scratch? | 113/120 flagged polarity-inverted (reversed electrodes plausible). Baseline drift handled by a bounded window. "Re-examine everything from scratch" is a **scope change — a question for the supervisors, not a plan.** |
| 3 | Are the small bumps on clean animals noise? | **No** — they are P / J-wave / T waves (real physiology). |
| 4 | Do the clean animals pass, with correct beat detection? | **Yes** — 83 animals, HR physiological, rr_cv 0.025, QTc 41.5 ms. |
| 5 | Did any treatment cause real tachy/bradycardia? | **Roisin's physiological call** — cannot be settled from traces alone, and needs the treatment labels. Some "fast clustered" traces may be real tachycardia rather than detection failure. |
| 6 | Clear, interpretable plots for clean and problem animals? | Delivered — overlay, FFT, and per-animal raw-vs-filtered figures. |

## 8. Open questions for the supervisors

- **QRS-offset definition** (Roisin) — the current field measures R-spike width,
  not clinical QRS duration; the correct definition is ambiguous in mouse ECG.
- **Physiological call on irregular-throughout animals** (Roisin) — real
  arrhythmia vs broken recording; needs the treatment metadata.
- **Scope of re-examination** (Luke/Roisin) — whether to re-check every recording
  from scratch rather than relying on the manual LabChart comments.

---

## Appendix — figures

- `overlay_validation_A201.png`, `overlay_validation_A111.png` — raw → filtered → features
- `fft_overlay_A201.png` — FFT power spectrum, before vs after (overlaid)
- `clean_beat_qrs_A201.png` — labelled clean beat + QRS FWHM vs true QRS
- `clean_rawvsfilt_A133.png` (and 122, 116) — clean-animal raw vs filtered
- `problem_animals/` — raw vs filtered for all 34 flagged animals
- `recovered/recovered_A123.png`, `recovered_A126.png` — recovery checks
