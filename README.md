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

## Findings

- [Why the ECG features do not cluster into treatment groups](docs/clustering_negative_result.md)
  — a documented negative result. Unsupervised clustering recovers only a
  **data-quality** split (clean vs irregular), not the 5 treatment arms. The
  treatment question requires **supervised** analysis against the withheld
  metadata; this note explains why, what was tried, and the solution.

## How to run

```bash
pip install -r requirements.txt
jupyter lab
```

Open each notebook in order and run **Kernel → Restart Kernel and Run All Cells**.

Place raw recordings in `data/` before running Notebook 01.

Raw ECG recordings are not included in this repository (data governance).
