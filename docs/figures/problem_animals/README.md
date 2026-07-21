# Problem animals — raw vs filtered (all 34 flagged)

One figure per flagged recording (`problem_A<id>.png`). Each shows the raw signal
and the filtered signal overlaid with detected R-peaks (top), and what the filter
removed (bottom).

**How to read the numbers.** `corr` is the correlation between the raw and the
filter-only output — high means the filter preserved the shape. `% removed` is the
fraction of the signal the filter stripped as noise/drift. For reference, **clean
animals sit at corr 0.93–0.97 with only 26–39% removed**. Values above 100% occur
when the raw is dominated by noise that is anti-correlated with the recovered
signal.

Sorted by severity (least to most affected).

## 🟢 Barely flagged — essentially clean (6)

Filter behaves perfectly; these only tripped the `rr_cv > 0.15` rhythm gate.
Candidates to **keep** under a two-tier gate.

| animal | corr | % removed | HR | beats |
|---|---|---|---|---|
| 117 | 0.99 | 17 | 336 | 114 |
| 129 | 0.98 | 19 | 525 | 127 |
| 245 | 0.98 | 23 | 459 | 230 |
| 131 | 0.97 | 24 | 462 | 230 |
| 139 | 0.97 | 25 | 686 | 368 |
| 134 | 0.95 | 32 | 598 | 873 |

## 🟡 Noisy but signal present (18)

Real beats visible. Window-relocation recovery was attempted on this group —
only **123** fully recovered (rr_cv 0.220 → 0.089) and **126** was borderline.
The rest are irregular *throughout* the recording, which points to real
physiology or a genuinely poor recording rather than a bad measurement window.

| animal | corr | % removed | HR | beats | note |
|---|---|---|---|---|---|
| 225 | 0.93 | 43 | 465 | 144 | |
| 111 | 0.92 | 43 | 284 | 72 | erratic bursts + gaps |
| 130 | 0.91 | 44 | 517 | 565 | |
| 126 | 0.93 | 45 | 452 | 225 | **borderline recovery** |
| 119 | 0.92 | 45 | 578 | 248 | no clean window found |
| 106 | 0.89 | 51 | 465 | 232 | |
| 121 | 0.89 | 52 | 125 | 26 | HR below physiological range |
| 123 | 0.86 | 52 | 503 | 305 | ✅ **recovered** |
| 151 | 0.93 | 55 | 344 | 125 | |
| 105 | 0.84 | 56 | 451 | 291 | |
| 232 | 0.84 | 56 | 337 | 132 | |
| 112 | 0.87 | 57 | 314 | 87 | polarity inverted |
| 160 | 0.84 | 61 | 542 | 428 | |
| 234 | 0.87 | 62 | 573 | 192 | |
| 252 | 0.80 | 62 | 560 | 184 | |
| 259 | 0.89 | 64 | 499 | 192 | |
| 101 | 0.75 | 70 | 400 | 196 | |
| 114 | 0.76 | 70 | 205 | 56 | polarity inverted, slow |

## 🔴 Severe — mostly noise (10)

corr < 0.75 and/or > 80% of the trace is noise. Recommended to **drop**.

| animal | corr | % removed | HR | beats | note |
|---|---|---|---|---|---|
| 113 | 0.62 | 83 | 213 | 46 | polarity inverted |
| 102 | 0.51 | 88 | 367 | 180 | |
| 103 | 0.43 | 93 | 340 | 157 | |
| 109 | 0.41 | 94 | 171 | 171 | HR below range |
| 120 | 0.39 | 97 | **93** | 27 | inverted; HR impossible for a mouse |
| 240 | 0.78 | 99 | 544 | 174 | |
| 107 | 0.21 | 102 | 303 | 150 | |
| 115 | 0.74 | 103 | 455 | 408 | |
| 127 | 0.74 | 105 | 560 | 115 | |
| 125 | 0.41 | 131 | 500 | 94 | rr_cv sits exactly on the 0.15 threshold |

---

Reproduced by the pipeline functions in
[`02_beat_averaging_and_clustering.ipynb`](../../../notebooks/02_beat_averaging_and_clustering.ipynb).
See [pipeline_validation_and_triage.md](../../pipeline_validation_and_triage.md)
for the full triage and recovery analysis.
