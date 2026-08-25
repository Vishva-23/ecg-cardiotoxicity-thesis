# Consolidated Analysis Notebooks

The analysis is organised into **three sequential notebooks** that replace the scattered
`.py` scripts. Run them in order; each stage's output feeds the next.

```
raw .txt ECG (data/)
      │
      ▼
┌─────────────────────────────────────────────┐
│ NB1_pipeline_and_features.ipynb              │  ← run first (slow: processes all recordings)
│  • two-stage filter → R-peak detection →     │
│    beat averaging → feature extraction       │
│  • merge with animal_codes_final.xlsx        │
│  • filter/detector validation, rich features │
└─────────────────────────────────────────────┘
      │  writes: outputs/_metadata_merged.csv   (master table, verified vs frozen baseline)
      │          outputs/rich_features.csv
      │          outputs/detector_robustness.csv
      ▼
┌─────────────────────────────────────────────┐
│ NB2_analysis_and_results.ipynb               │  ← run second (the results engine)
│  • descriptives, grouping (unsup/sup/PCA)    │
│  • dose-response tests, effect sizes         │
│  • Bayesian hierarchical model               │
│  • consensus phenotyping                     │
│  • advanced robustness (PERMANOVA/TOST/      │
│    robust reg/leave-one-mouse-out/ROPE)      │
└─────────────────────────────────────────────┘
      │  writes: all_tests_results.csv, effect-size / dose-response / Bayesian figures, …
      ▼
┌─────────────────────────────────────────────┐
│ NB3_flagged_recovery.ipynb   (optional)      │  ← run third, independent side-branch
│  • re-window flagged recordings              │
│  • recover clean segments (23 recovered)     │
│  • recovery / robustness figures             │
└─────────────────────────────────────────────┘
      │  writes: flagged_recovery.csv, flagged_rewindowed.csv,
      │          recovered_windows.csv, recovered23_animals_features.csv
```

## Why three notebooks (not one)

The split lands on the two points where data is persisted to disk (`_metadata_merged.csv`
and the recovery tables). This means:

- **NB1 is the only slow stage** (it processes every raw recording). Once it has written
  `_metadata_merged.csv`, you can re-run NB2 as often as you like without re-paying that cost.
- **NB2 is fast and is where all the thesis numbers come from** — re-run it freely while iterating.
- **NB3 is an optional branch** — the main results in NB2 do not depend on it.

## Design notes

- The **pipeline functions are loaded once** at the top of NB1 (from the original
  `02_beat_averaging_and_clustering.ipynb` cells) and reused, instead of the previous pattern
  where 15 separate scripts each re-loaded them.
- NB1 **verifies** that the regenerated `_metadata_merged.csv` reproduces the frozen baseline
  on every analysis-critical column (status, group, study, sex, all features, `rr_cv`).
- Each NB2 / NB3 section is self-contained (re-reads its inputs), so sections can be
  re-run independently.

## To execute non-interactively

```bash
python -c "import nbformat; from nbclient import NotebookClient; \
nb=nbformat.read('NB1_pipeline_and_features.ipynb',as_version=4); \
NotebookClient(nb,timeout=1800,resources={'metadata':{'path':'.'}}).execute(); \
nbformat.write(nb,'NB1_pipeline_and_features.ipynb')"
```
(repeat for NB2, NB3), or open in Jupyter and *Run All*.
