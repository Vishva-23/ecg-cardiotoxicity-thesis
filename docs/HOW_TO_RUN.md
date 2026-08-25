# How to Run — step-by-step for a new user

This package is **portable**: paths are worked out automatically, as long as you keep the
folder layout below and **run the notebooks from the `code/` folder**.

---

## 1. Folder layout (keep this exactly)

```
FINAL_notebooks_and_results/          <-- project root
├── code/                             <-- OPEN AND RUN THE NOTEBOOKS FROM HERE
│   ├── NB1_pipeline_and_features.ipynb
│   ├── NB2_analysis_and_results.ipynb
│   ├── NB3_flagged_recovery.ipynb
│   ├── qt_cohort_audit.py            (pipeline loader)
│   ├── figutil.py                    (saves each figure as a single graph)
│   └── 02_beat_averaging_and_clustering.ipynb   (legacy validation notebook)
├── data/                             <-- PUT THE RAW ECG .txt FILES HERE
├── outputs/                          <-- RESULTS ARE WRITTEN HERE (auto-created)
│   └── figures/                      <-- figures written here
├── animal_codes_final.xlsx           <-- animal metadata (already included)
└── README.md
```

> **The one rule that makes paths work:** run each notebook with the working directory set to
> `code/` (open the `.ipynb` from inside `code/`, or `cd code` first). Everything else is relative
> to that.

---

## 2. Where the dataset goes

Put the **118 raw ECG text files** (the LabChart exports, e.g. `2024_12_14_201.txt`) into the
**`data/`** folder. That's the only thing you need to add — the metadata spreadsheet is already in
the package.

---

## 3. How the paths resolve (the "paths list")

| What | Path used by the code | Resolves to |
|---|---|---|
| **Raw ECG input** | auto-detected: walks up from `code/` until it finds a `data/` folder | `…/data/` |
| **Animal metadata** | `../animal_codes_final.xlsx` (relative to `code/`) | project-root xlsx |
| **Pipeline functions** | `02_beat_averaging_and_clustering.ipynb` (in the current folder) | `code/02_…ipynb` |
| **Master feature table** | `../outputs/_metadata_merged.csv` | `…/outputs/_metadata_merged.csv` |
| **Result tables (CSV)** | `../outputs/…​.csv` | `…/outputs/` |
| **Figures** | `../outputs/figures/…​.png` | `…/outputs/figures/` |

So: **inputs come from `data/` + `outputs/`, outputs go to `outputs/` (+ `outputs/figures/`)** — all
relative to `code/`. You do **not** edit any path by hand.

---

## 4. Requirements (one-time)

Python 3.11 (Anaconda recommended) with:

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib openpyxl neurokit2 nbformat nbclient ipykernel
```

Anaconda already has most of these. (No pymc / arviz / skbio needed.)

---

## 5. Run order

Open each notebook **from `code/`** and **Run All**, in this order:

| Step | Notebook | Needs raw data? | Produces |
|---|---|---|---|
| 1 | **NB1** — pipeline → features | **Yes** (`data/*.txt`) | `_metadata_merged.csv`, `rich_features.csv`, `detector_robustness.csv` |
| 2 | **NB2** — analysis & results | **No** — reads the shipped `_metadata_merged.csv` | result CSVs + single-graph figures |
| 3 | **NB3** — flagged recovery | **Yes** (`data/*.txt`) | recovery CSVs + figures |

**Fastest check (no raw data needed):** open `code/NB2_analysis_and_results.ipynb` and Run All —
it reproduces every result table and figure from the included `_metadata_merged.csv`.

### Optional — validation figures
The beat-structure / per-beat / filter validation figures come from the **legacy**
`02_beat_averaging_and_clustering.ipynb` (also run from `code/`, needs `data/*.txt`). Running it
now writes each panel as its own single graph.

---

## 6. Run without opening Jupyter (from `code/`)

```bash
cd code
python -c "import nbformat; from nbclient import NotebookClient; \
nb=nbformat.read('NB2_analysis_and_results.ipynb',as_version=4); \
NotebookClient(nb,timeout=1800,resources={'metadata':{'path':'.'}}).execute(); \
nbformat.write(nb,'NB2_analysis_and_results.ipynb')"
```
(swap in NB1 / NB3 the same way).

---

## 7. If something goes wrong

- **`FileNotFoundError: .../data`** → you didn't run from `code/`, or `data/` is empty. Put the
  `.txt` files in `data/` and run from `code/`.
- **`FileNotFoundError: _metadata_merged.csv`** (in NB2) → run NB1 first, or use the shipped copy in
  `outputs/`.
- **Figures look like one big multi-panel image** → make sure `figutil.py` is in `code/`; the
  notebooks import it to save each panel as a single graph.
- **Different machine, hardcoded path error** → already fixed; paths are relative/auto-detected.
