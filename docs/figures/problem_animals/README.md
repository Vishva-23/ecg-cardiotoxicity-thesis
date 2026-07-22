# Problem animals — raw vs filtered (all 34 flagged)

One figure per flagged recording (`problem_A<id>.png`): raw and filtered overlaid
with detected R-peaks (top), and what the filter removed (bottom). **The plots show
a 1-second slice for legibility; the table below is computed over the full baseline
window.**

Regenerate with [`code/problem_animal_triage.py`](../../../code/problem_animal_triage.py).

> **Correction notice.** An earlier version of this page grouped these animals by
> raw-vs-filtered correlation and "% signal removed". Those numbers were computed
> over only the **first 1 second** of baseline windows that are 11–88 s long, so
> they were unrepresentative, and the grouping derived from them was wrong. On full
> windows those metrics separate clean from flagged only weakly (clean median 43%
> removed vs flagged 72%, with heavy overlap — 61 of 83 clean animals exceed the
> best flagged animal). **They are not a reliable triage criterion and are no longer
> used here.** The triage below is keyed on `rr_cv`, `status`, and heart rate — the
> pipeline's own metrics, all computed over the full window.

## Triage criteria

| tier | rule |
|---|---|
| **KEEP** | `status == OK`, HR within 300–700 bpm, `rr_cv ≤ 0.25` — mildly over the 0.15 gate only |
| **REVIEW** | `status == OK`, HR in range, `0.25 < rr_cv ≤ 0.60` — needs a judgement call |
| **DROP** | `status != OK`, **or** HR outside 300–700, **or** `rr_cv > 0.60` |

## 🟢 KEEP (13) — mildly irregular, otherwise sound

Candidates to retain under a two-tier gate.

| animal | HR | rr_cv |
|---|---|---|
| 125 | 500 | 0.151 |
| 129 | 525 | 0.156 |
| 134 | 598 | 0.169 |
| 127 | 560 | 0.185 |
| 245 | 459 | 0.193 |
| 252 | 560 | 0.206 |
| 225 | 465 | 0.211 |
| 106 | 465 | 0.216 |
| 259 | 499 | 0.217 |
| 123 | 503 | 0.221 |
| 160 | 542 | 0.222 |
| 119 | 578 | 0.226 |
| 240 | 544 | 0.234 |

## 🟡 REVIEW (6) — moderately irregular, HR still physiological

Rhythm is irregular but heart rate is plausible, so this may be **real rhythm
variability rather than artifact** — a physiological judgement (see the open
question for Roisin in the main report).

| animal | HR | rr_cv |
|---|---|---|
| 234 | 573 | 0.264 |
| 139 | 686 | 0.304 |
| 130 | 517 | 0.310 |
| 105 | 451 | 0.445 |
| 131 | 462 | 0.479 |
| 126 | 452 | 0.587 |

## 🔴 DROP (15) — unusable

Failed status, non-physiological heart rate, or extreme irregularity.

| animal | status | HR | rr_cv | why |
|---|---|---|---|---|
| 113 | NEEDS_REVIEW | 213 | 0.647 | HR below range |
| 101 | OK | 400 | 0.650 | rr_cv > 0.6 |
| 121 | NEEDS_REVIEW | 125 | 0.674 | HR far below range |
| 151 | OK | 344 | 0.825 | rr_cv > 0.6 |
| 109 | NEEDS_REVIEW | 171 | 0.934 | HR below range |
| 232 | OK | 337 | 0.937 | rr_cv > 0.6 |
| 114 | NEEDS_REVIEW | 205 | 0.955 | HR below range |
| 117 | OK | 336 | 0.960 | rr_cv > 0.6 |
| 120 | NEEDS_REVIEW | **93** | 1.015 | HR impossible for a mouse |
| 112 | OK | 314 | 1.385 | extreme irregularity |
| 103 | OK | 340 | 1.406 | extreme irregularity |
| 102 | OK | 367 | 1.473 | extreme irregularity |
| 111 | NEEDS_REVIEW | 284 | 1.527 | HR below range + extreme |
| 107 | OK | 303 | 1.997 | extreme irregularity |
| 115 | OK | 455 | **5.434** | no coherent rhythm |

## Recovery (window relocation)

Relocating to the cleanest window in the same recording was attempted. Only
**123** improved to below the 0.15 gate (rr_cv 0.220 → 0.089, HR 503 → 502 —
consistent, so it is the same animal's rhythm in a cleaner stretch). **126** found
a clean window (0.587 → 0.047) but its HR shifted 452 → 559, so it needs manual
review. For the rest, relocation did **not** reduce `rr_cv` — the irregularity is
spread across the whole recording, meaning it is intrinsic to the animal's rhythm
rather than a bad measurement window.

See [pipeline_validation_and_triage.md](../../pipeline_validation_and_triage.md)
for the full analysis.
