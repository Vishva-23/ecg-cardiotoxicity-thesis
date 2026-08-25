# ECG Cardiotoxicity Analysis

Reproducible Python pipeline for preprocessing and multivariate analysis of preclinical
mouse ECG data, assessing the cardiac safety of ethanolamine against a doxorubicin
positive control.

---

## Consolidated workflow (start here)

The analysis is organised into **three sequential notebooks** in [`notebooks/`](notebooks/).
Each notebook is self-contained (the former standalone `.py` scripts now live inside the
notebook cells); run them in order.

| Step | Notebook | Needs raw data? | Produces |
|------|----------|-----------------|----------|
| 1 | [`NB1_pipeline_and_features.ipynb`](notebooks/NB1_pipeline_and_features.ipynb) | **Yes** (`data/*.txt`) | `_metadata_merged.csv` (master table), `rich_features.csv`, `detector_robustness.csv` |
| 2 | [`NB2_analysis_and_results.ipynb`](notebooks/NB2_analysis_and_results.ipynb) | **No** — reads `_metadata_merged.csv` | all result CSVs + single-graph figures (dose-response, effect sizes, Bayesian, PERMANOVA/TOST/leave-one-mouse-out) |
| 3 | [`NB3_flagged_recovery.ipynb`](notebooks/NB3_flagged_recovery.ipynb) | **Yes** (`data/*.txt`) | recovery tables + figures (23 flagged recordings recovered) |

**NB1 → produces the master table → NB2 turns it into every result and figure → NB3 is an
independent side-branch** that re-windows the flagged recordings.

- **Fastest check (no raw data):** open **NB2** and *Run All* — it reproduces every result
  table and figure from the committed `_metadata_merged.csv`.
- Full run order, the paths list, and troubleshooting: [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md).
- What each notebook contains and how the stages connect: [`docs/NOTEBOOKS_README.md`](docs/NOTEBOOKS_README.md).

### What the notebooks import
The notebooks are self-contained apart from two co-located helpers in `notebooks/`:

- `qt_cohort_audit.py` — loads the validated pipeline functions from
  `02_beat_averaging_and_clustering.ipynb` (imported by NB1 and NB3).
- `02_beat_averaging_and_clustering.ipynb` — the validated beat-averaging / QT reference
  notebook that `qt_cohort_audit.py` reads its functions from.

NB2's figure helper is embedded as its own cell — no separate file.

---

## How to run

```bash
pip install -r requirements.txt
jupyter lab
```

Place the raw recordings in `data/` (raw ECG `.txt` files are **not** included, per data
governance), then open the notebooks from `notebooks/` and run
**Kernel → Restart Kernel and Run All Cells** in the NB1 → NB2 → NB3 order.

Paths are auto-detected — NB1 walks up from the working directory until it finds `data/`,
and all outputs are written under `outputs/` (and `outputs/figures/`). You do not edit any
path by hand. See [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md) for the full paths list.

---

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

---

## Exploratory notebooks (historical)

The original exploratory notebooks (`01_data_exploration`, `03_denoising_comparison`,
`04_healthy_template`, `04_beat_overlay`, `05_detect_diagnosis`) remain in `notebooks/` as
the development record. The three consolidated notebooks above supersede them for
reproducing the thesis results.
