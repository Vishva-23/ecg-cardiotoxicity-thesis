# ECG Cardiotoxicity Analysis

Reproducible Python pipeline for preprocessing and multivariate analysis of preclinical mouse ECG data to assess cardiac safety.

## Notebooks

Run in order:

| Notebook | What it does |
|----------|-------------|
| `01_data_exploration.ipynb` | Load raw recordings, apply bandpass filter, detect R peaks, extract per-beat features (RR, QRS, QT, QTc) |
| `02_beat_averaging_and_clustering.ipynb` | Average beats per animal, build multivariate feature matrix, run PCA and clustering |
| `03_denoising_comparison.ipynb` | Compare denoising strategies; quantify noise residuals and spectral quality |
| `04_healthy_template.ipynb` | Construct a healthy-animal beat template; compute distance-to-healthy metric for each recording |

## How the pipeline works

- [The ECG pipeline — what it does, how it works, and what changed](docs/pipeline_explained.md)
  — a stage-by-stage description of the processing chain, why mouse ECG needs a
  custom detector and a 150 Hz band, and a precise before/after changelog of the
  QT and R-amplitude corrections.

## Findings

- [Why the ECG features do not cluster into treatment groups](docs/clustering_negative_result.md)
  — a documented negative result. Unsupervised clustering recovers only a
  **data-quality** split (clean vs irregular), not the 5 treatment arms. The
  treatment question requires **supervised** analysis against the withheld
  metadata; this note explains why, what was tried, and the solution.
- [Why supervised analysis is required — an evidence log](docs/why_supervised_is_required.md)
  — the reproducible battery behind that conclusion: 3 clustering algorithms ×
  5 cluster counts, outlier-removal re-analysis, and a PCA variance check, all
  showing only a data-quality split (k=2) and no treatment structure (k=5).
- [Corrections and methods](docs/corrections_and_methods.md) — what the QT and
  R-amplitude fixes changed (QT median 72.7 → 43.9 ms; QTc → 41.2 ms, matching
  mouse literature), why the PCA moved after correcting the values, and the
  denoising comparison behind the Butterworth + notch choice.
- [Pipeline validation and problem-animal triage](docs/pipeline_validation_and_triage.md)
  ([Word version](docs/ECG_pipeline_validation_report.docx)) — confirmation the
  pipeline works on the 83 clean animals, FFT filter verification, a triage of the
  34 flagged animals (keep / recover / physiology / drop), the window-relocation
  recovery analysis, and point-by-point answers to the supervisor's questions.

## How to run

```bash
pip install -r requirements.txt
jupyter lab
```

Open each notebook in order and run **Kernel → Restart Kernel and Run All Cells**.

Place raw recordings in `data/` before running Notebook 01.

Raw ECG recordings are not included in this repository (data governance).
