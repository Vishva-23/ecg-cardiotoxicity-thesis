"""Split outputs/outlier_clean_signals.csv into one file per animal.

For each animal_id, writes:
  - outputs/clean_per_animal/clean_<id>.csv  (comma-delimited, columns: time_s, mv)
  - outputs/clean_per_animal/clean_<id>.txt  (tab-delimited, LabChart style, same columns)

Run from the project root:  python notebooks/split_clean_per_animal.py
"""

from pathlib import Path
import pandas as pd

# Dynamic path detection — works regardless of where the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "outputs" / "outlier_clean_signals.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "clean_per_animal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC)

for animal_id, group in df.groupby("animal_id", sort=True):
    out = group[["time_s", "mv"]]

    csv_path = OUT_DIR / f"clean_{animal_id}.csv"
    txt_path = OUT_DIR / f"clean_{animal_id}.txt"

    # Comma-delimited CSV
    out.to_csv(csv_path, index=False)

    # Tab-delimited LabChart-style text: column names only, no extra header.
    out.to_csv(txt_path, sep="\t", index=False)

    print(f"animal {animal_id}: {len(out)} samples -> {csv_path.name}, {txt_path.name}")

print(f"\nWrote {df['animal_id'].nunique()} animals to {OUT_DIR}")
