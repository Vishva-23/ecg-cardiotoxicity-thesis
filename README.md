# ECG Cardiotoxicity Thesis
## Reproducible Signal Processing and Multivariate Analysis of Preclinical Mouse ECG Data

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![JupyterLab](https://img.shields.io/badge/JupyterLab-4.x-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Objectives](#research-objectives)
3. [Repository Structure](#repository-structure)
4. [Data Description](#data-description)
5. [Pipeline Summary](#pipeline-summary)
6. [Key Findings So Far](#key-findings-so-far)
7. [Installation and Setup](#installation-and-setup)
8. [How to Run](#how-to-run)
9. [Results](#results)
10. [Reference Papers](#reference-papers)
11. [Acknowledgements](#acknowledgements)

---

## Project Overview

This repository contains the code, notebooks, and results for my Masters thesis at University College Cork (UCC). The project develops a **reproducible, automated Python pipeline** for preprocessing and analysing electrocardiogram (ECG) signals recorded from mice in preclinical cardiac safety studies.

Conventional ECG analysis relies on manual signal summarisation and univariate statistical testing, which limits the ability to capture multivariate structure and assess robustness under real experimental conditions. This project addresses that gap by building methods that:

- Extract informative features from noisy, small-sample biomedical data
- Explicitly account for experimental design limitations
- Are fully reproducible — same code, same result, every time
- Are stress-tested under realistic preclinical conditions

**Context:** Drugs used in cancer chemotherapy (e.g. doxorubicin) can cause serious heart damage — a condition called cardiotoxicity. Before any drug reaches human trials, it is tested in mice. The heart electrical activity of these mice is recorded as an ECG. Better computational methods for analysing these recordings can lead to earlier, more reliable detection of cardiac harm.

---

## Research Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Build a reproducible ECG preprocessing and feature extraction pipeline in Python | ✅ In progress |
| 2 | Apply multivariate methods (PCA, clustering) to find structure not visible in univariate analysis | ⬜ Upcoming |
| 3 | Quantify the impact of experimental design constraints on analytical conclusions | ⬜ Upcoming |
| 4 | Optional: Bayesian hierarchical modelling for uncertainty quantification | ⬜ Later stage |

**Research Questions:**
- How can noisy mouse ECG time-series data be processed reproducibly?
- Do multivariate feature representations reveal structure not apparent in traditional univariate summaries?
- How sensitive are inferred patterns to experimental design deviations such as blocking, timing delays, and small sample sizes?
- To what extent does explicit modelling of experimental structure alter analytical conclusions?

---

## Repository Structure

```
ecg-cardiotoxicity-thesis/
│
├── README.md                          ← This file
│
├── notebooks/
│   └── 01_data_exploration.ipynb      ← Pipeline: load, filter, detect, extract
│
├── outputs/
│   ├── animal_201_baseline_features.csv   ← Feature table (Animal 201)
│   └── figures/
│       ├── 01_full_recording.png          ← Full 16-minute ECG overview
│       ├── 02_filter_comparison.png       ← Raw vs bandpass filtered signal
│       ├── 03_r_peaks_5sec.png            ← R peak detection (5 seconds)
│       ├── 04_r_peaks_1sec_zoom.png       ← R peak verification (1 second)
│       ├── 05_single_beat.png             ← Single beat morphology
│       └── 06_rr_intervals.png            ← RR interval variability
│
├── .gitignore
└── requirements.txt
```

> **Note:** Raw data files (.txt) are not included in this repository for data management and privacy reasons. Data is provided by Dr Roisin Kelly-Laubscher (UCC Pharmacology). Please contact the supervisors for access.

---

## Data Description

### Source
- **Equipment:** LabChart (ADInstruments)
- **Export format:** Tab-separated .txt files
- **Recording type:** Surface ECG, Lead II configuration
- **Animal model:** C57Bl/6J mice under isoflurane anaesthesia
- **File naming:** `Year_Month_Day_AnimalNumber.txt`

### File Received (Animal 201)

| Property | Value |
|----------|-------|
| Filename | `2024_12_14_201.txt` |
| Animal | Mouse 201 |
| Date | 14 December 2024 |
| Total samples | 1,009,300 |
| Sampling rate | 1000 Hz |
| Duration | 1009.3 seconds (~16.8 minutes) |
| NaN values | 229 (annotation lines) |
| Signal range | ±10 mV |

### Experimental Annotations

Two markers were embedded in the recording by Dr Kelly-Laubscher:

| Annotation | Sample | Time | Meaning |
|------------|--------|------|---------|
| `ecg baseline` | 94,599 | 94.6s | Start of resting reference period |
| `ecg baseline end` | 119,599 | 119.6s | End of resting reference period |

The 25-second baseline window (94.6s → 119.6s) is the cleanest segment of the recording and serves as the reference/control period for each animal.

### Signal Quality by Segment

| Segment | Duration | Std Dev | Quality |
|---------|----------|---------|---------|
| Pre-baseline | 94.6s | 0.16 mV | Good |
| Baseline | 25.0s | 0.10 mV | Excellent |
| Post-baseline | 889.7s | 0.97 mV | Noisy (~10x) |

---

## Pipeline Summary

### Notebook 01 — Data Exploration and Feature Extraction

The pipeline processes one animal at a time through the following steps:

```
Raw .txt file
     ↓
1. Load and parse
   Skip 6 header lines, extract voltage column (mV)
   Handle NaN values from annotation lines
     ↓
2. Map annotations
   Identify baseline window from embedded markers
   Split signal into pre-baseline / baseline / post-baseline
     ↓
3. Bandpass filter
   Butterworth 0.5–40 Hz (zero-phase, filtfilt)
   Removes baseline wander and high-frequency noise
     ↓
4. R peak detection
   scipy.find_peaks with manual threshold parameters
   (height=0.08, distance=100, prominence=0.05)
     ↓
5. RR interval extraction
   np.diff(r_peaks) / sampling_rate * 1000
     ↓
6. QRS duration
   Full Width at Half Maximum of R peak
     ↓
7. QT and QTc intervals
   Threshold-crossing method after R peak
   QTc = QT / sqrt(RR/100)  [Mitchell et al., 1998]
     ↓
8. Feature table
   Saved as CSV — one row per animal per segment
```

### Key Methodological Decision: R Peak Detection

Standard automated algorithms were systematically evaluated and all failed:

| Algorithm | Peaks Found | Heart Rate | Status |
|-----------|------------|------------|--------|
| Pan-Tompkins (1985) | 16 | 192 bpm | ❌ Failed |
| Hamilton (2002) | 13 | 156 bpm | ❌ Failed |
| Elgendi (2010) | 4 | 48 bpm | ❌ Failed |
| Engzee (2012) | 0 | 0 bpm | ❌ Failed |
| Rodrigues (2021) | 18 | 216 bpm | ❌ Failed |
| **Manual (scipy)** | **41** | **492 bpm** | **✅ Correct** |

Expected: ~38 peaks in 5 seconds = ~456 bpm (normal mouse range: 400–500 bpm)

**Root cause:** R peak amplitude is ~0.2 mV in this recording, compared to ~1–2 mV in human ECG datasets these algorithms were designed and calibrated for. The mouse-specific J wave also confuses detectors expecting human waveform morphology.

This failure of standard pipelines on preclinical mouse data is a documented methodological finding and a direct contribution of this thesis.

---

## Key Findings So Far

### Animal 201 — Baseline Segment Results

| Parameter | Measured Value | Literature Reference | Status |
|-----------|---------------|---------------------|--------|
| Heart rate | 492.5 bpm | 400–500 bpm | ✅ Normal |
| RR interval (mean) | 121.8 ms | 120–150 ms | ✅ Normal |
| RR interval (std) | 0.9 ms | — | ✅ Very stable |
| QRS duration | FWHM method | 10–15 ms | ⚠️ Under review |
| QT interval | 70.5 ms | 40–50 ms | ⚠️ Elevated |
| QTc | 63.9 ms | 34–43 ms | ⚠️ Elevated |

> **Note on QTc:** The elevated QTc relative to literature values is an open question to be discussed with supervisors. Possible explanations include: signal amplitude scaling differences between LabChart export settings and the equipment used in reference studies; differences in T wave endpoint detection method (automated threshold crossing vs manual specialist verification); or genuine individual animal variation. This will be resolved before further analysis proceeds.

### Beat Morphology

Single beat analysis confirms textbook mouse Lead II ECG characteristics:
- No prominent Q wave (smooth upstroke to R peak)
- Clear J wave at approximately +15 samples after R peak
- T wave hump at approximately +60 samples after R peak
- No prominent S wave (signal does not go negative)
- Consistent with Merentie et al. (2015) and Boukens et al. (2014)

---

## Installation and Setup

### Requirements

```
Python 3.12
numpy==1.26.4      (must be <2.0 for compatibility)
pandas>=2.0
scipy>=1.10
matplotlib>=3.5
neurokit2==0.2.13
jupyterlab
```

### Install

```bash
# Clone the repository
git clone https://github.com/Vishva-23/ecg-cardiotoxicity-thesis.git
cd ecg-cardiotoxicity-thesis

# Install dependencies
pip install "numpy<2" pandas scipy matplotlib neurokit2 jupyterlab

# Launch JupyterLab
jupyter lab
```

> **Important:** NumPy must be version < 2.0. NumPy 2.x causes compatibility conflicts with pandas and matplotlib in the current Anaconda environment.

---

## How to Run

1. Place the data file in the `data/` folder:
   ```
   data/2024_12_14_201.txt
   ```

2. Open JupyterLab:
   ```bash
   jupyter lab
   ```

3. Open `notebooks/01_data_exploration.ipynb`

4. Update the file path in Section 2:
   ```python
   DATA_PATH = r'path\to\your\data\2024_12_14_201.txt'
   ```

5. Run all cells: **Kernel → Restart Kernel and Run All Cells**

---

## Results

### Output Files

| File | Description |
|------|-------------|
| `outputs/animal_201_baseline_features.csv` | Feature table for Animal 201 baseline |
| `outputs/figures/01_full_recording.png` | Full 16-minute ECG with annotated baseline window |
| `outputs/figures/02_filter_comparison.png` | Before and after bandpass filtering |
| `outputs/figures/03_r_peaks_5sec.png` | R peak detection over 5 seconds |
| `outputs/figures/04_r_peaks_1sec_zoom.png` | 1-second zoom verifying individual peak detection |
| `outputs/figures/05_single_beat.png` | Single beat morphology with J wave and T wave labelled |
| `outputs/figures/06_rr_intervals.png` | Beat-to-beat RR interval variability |

### Feature Table Columns

| Column | Description | Unit |
|--------|-------------|------|
| `animal_id` | Animal identifier | — |
| `date` | Recording date | YYYY-MM-DD |
| `recording_type` | Anaesthesia type | — |
| `segment` | Experimental period | — |
| `n_beats_detected` | Number of R peaks found | count |
| `heart_rate_bpm` | Mean heart rate | bpm |
| `rr_mean_ms` | Mean RR interval | ms |
| `rr_std_ms` | RR interval standard deviation | ms |
| `qrs_mean_ms` | Mean QRS duration (FWHM) | ms |
| `qt_mean_ms` | Mean QT interval | ms |
| `qtc_ms` | QTc interval | ms |

---

## Reference Papers

The following papers directly informed the methods and biological context of this project:

---

### Primary Tool Reference

**Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., Schölzel, C., & Chen, S. H. A. (2021).**
NeuroKit2: A Python toolbox for neurophysiological signal processing.
*Behavior Research Methods, 53*, 1689–1696.
https://doi.org/10.3758/s13428-020-01516-y

> Used for: ECG signal cleaning and peak detection. Primary tool for the preprocessing pipeline. The paper's emphasis on reproducibility and open-source methodology directly aligns with Thesis Objective 1.

---

### Mouse ECG Algorithm and Reference Values

**Merentie, M., Lipponen, J. A., Hedman, M., Hedman, A., Hartikainen, J., Huusko, J., Lottonen-Raikaslehto, L., Parviainen, V., Laidinen, S., Karjalainen, P. A., & Ylä-Herttuala, S. (2015).**
Mouse ECG findings in aging, with conduction system affecting drugs and in cardiac pathologies: Development and validation of ECG analysis algorithm in mice.
*Physiological Reports, 3*(12), e12639.
https://doi.org/10.14814/phy2.12639

> Used for: Normal reference values for mouse ECG parameters (heart rate, RR, QRS, QTc). Mouse-specific QTc formula (Mitchell et al., 1998 modification). Confirmation that mouse ECG has a J wave, no isoelectric ST segment, and a merged T wave — all of which affect algorithm design decisions in this pipeline.

---

### Cardiotoxicity ECG and Anesthesia Effects

**Warhol, A., George, S. A., Obaid, S. N., Efimova, T., & Efimov, I. R. (2021).**
Differential cardiotoxic electrocardiographic response to doxorubicin treatment in conscious versus anesthetized mice.
*Physiological Reports, 9*, e14987.
https://doi.org/10.14814/phy2.14987

> Used for: Confirmation that isoflurane anaesthesia is a systematic confound that slows heart rate and prolongs RR intervals — relevant because all recordings in this project are from anaesthetised mice. Validation that 1 kHz sampling rate and .txt export format are appropriate for mouse ECG analysis. QTc formula confirmation. Heart rate variability (HRV) metrics as additional cardiotoxicity features.

---

### ECG Preprocessing Review

**Safdar, M. F., Nowak, R. M., & Pałka, P. (2024).**
Pre-Processing techniques and artificial intelligence algorithms for electrocardiogram (ECG) signals analysis: A comprehensive review.
*Computers in Biology and Medicine, 170*, 107908.
https://doi.org/10.1016/j.compbiomed.2023.107908

> Used for: Justification of bandpass filtering (0.5–40 Hz) as the standard preprocessing approach, validated across 200+ studies. QRS detection methodology (Pan-Tompkins and threshold-based approaches). Identification of the small-dataset limitation in current ECG AI methods — the gap this thesis directly addresses.

---

### Mouse ECG Morphology

**Boukens, B. J., Rivaud, M. R., Rentschler, S., & Coronel, R. (2014).**
Misinterpretation of the mouse ECG: 'musing the waves of Mus musculus'.
*The Journal of Physiology, 592*(21), 4613–4626.
https://doi.org/10.1113/jphysiol.2014.279380

> Used for: Detailed explanation of why mouse ECG is fundamentally different from human ECG, and why human ECG analysis algorithms require adaptation for preclinical research. Cited in Merentie et al. (2015) and Warhol et al. (2021) as the definitive reference on mouse ECG morphology.

---

### QTc Formula for Mice

**Mitchell, G. F., Jeron, A., & Koren, G. (1998).**
Measurement of heart rate and Q-T interval in the conscious mouse.
*American Journal of Physiology, 274*(3), H747–H751.
https://doi.org/10.1152/ajpheart.1998.274.3.H747

> Used for: The mouse-specific heart rate correction formula for QT interval:
> **QTc = QT / √(RR / 100)**
> where QT and RR are in milliseconds. This formula is used in both Merentie et al. (2015) and Warhol et al. (2021) and is implemented throughout this pipeline.

---

## Acknowledgements

- **Dr Roisin Kelly-Laubscher** (UCC Pharmacology & Therapeutics) — for providing the experimental ECG data and biological expertise
- **Dr Luke Kelly** (UCC Statistics) — for statistical and methodological guidance
- **NeuroKit2 development team** — for providing an open-source, well-documented biosignal processing library

---

## Project Status

**Current stage:** Notebook 01 complete — baseline pipeline built and validated for Animal 201.

**Next steps:**
- Receive remaining data files from Dr Kelly-Laubscher
- Apply pipeline to all animals and all experimental segments
- Build complete multivariate feature matrix
- Resolve QTc measurement discrepancy with supervisors
- Begin PCA and multivariate analysis (Notebook 02)

---

*Masters Thesis — University College Cork — 2024/2025*
