"""
Missing-animal report from per_file_audit.csv.

Protocol: 120 animals = acute (100s) 60 + chronic (200s) 60.
Expected full set = {101..160} (acute) U {201..260} (chronic).
Anything in that expected set with NO file = MISSING.

Output: outputs/missing_animals_report.csv (animal_id, status).
Nothing committed.
"""
import pandas as pd
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "data").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
OUT = ROOT / "outputs"

# expected full protocol set (60 acute + 60 chronic)
ACUTE = list(range(101, 161))      # 101..160
CHRONIC = list(range(201, 261))    # 201..260
EXPECTED = ACUTE + CHRONIC

audit = pd.read_csv(OUT / "per_file_audit.csv")
found = sorted(int(a) for a in audit["animal_id"].dropna().unique())
status_map = {int(r.animal_id): str(r.status) for _, r in audit.iterrows()
              if pd.notna(r.animal_id)}

# 1) all IDs found
print("=" * 64)
print(f"1) ANIMAL IDs FOUND IN FILES: {len(found)} files")
print("=" * 64)
print("  acute (100s):  ", [a for a in found if 100 <= a < 200])
print("  chronic (200s):", [a for a in found if 200 <= a < 300])
other = [a for a in found if not (100 <= a < 300)]
if other:
    print("  outside 100s/200s:", other)

# 2) gaps = expected but not found
missing = sorted(set(EXPECTED) - set(found))
extra = sorted(set(found) - set(EXPECTED))
print("\n" + "=" * 64)
print(f"2) MISSING ANIMALS (no file at all): {len(missing)}")
print("=" * 64)
print(f"  >>> MISSING IDs: {missing}")
print(f"      acute missing  : {[a for a in missing if a < 200]}")
print(f"      chronic missing: {[a for a in missing if a >= 200]}")
if extra:
    print(f"  NOTE: files present but OUTSIDE expected 101-160/201-260: {extra}")
print(f"  (expected {len(EXPECTED)} = 60 acute + 60 chronic; found {len(found)}; "
      f"missing {len(missing)})")

# 3) FAILED_PEAKS and NEEDS_REVIEW from the audit status
fp = sorted(a for a, s in status_map.items() if s == "FAILED_PEAKS")
nr = sorted(a for a, s in status_map.items() if s == "NEEDS_REVIEW")
print("\n" + "=" * 64)
print("3) PROCESSING-FLAG ANIMALS (file present, but flagged)")
print("=" * 64)
print(f"  FAILED_PEAKS ({len(fp)}): {fp}")
print(f"  NEEDS_REVIEW ({len(nr)}): {nr}")

# 4) clean summary table: every expected animal -> status
rows = []
for aid in sorted(EXPECTED + extra):
    if aid in missing:
        st = "MISSING"
    else:
        st = status_map.get(aid, "UNKNOWN")
    rows.append({"animal_id": aid, "status": st})
report = pd.DataFrame(rows).sort_values("animal_id").reset_index(drop=True)
report.to_csv(OUT / "missing_animals_report.csv", index=False)

print("\n" + "=" * 64)
print("4) SAVED outputs/missing_animals_report.csv")
print("=" * 64)
print("  status counts:")
for st, n in report["status"].value_counts().items():
    print(f"    {st:<14}: {n}")

print("\n" + "*" * 64)
print(f"*  MISSING ANIMALS FOR ROISIN: {missing}")
print("*" * 64)
