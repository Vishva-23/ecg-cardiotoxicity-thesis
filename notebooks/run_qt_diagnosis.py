"""
QT measurement DIAGNOSIS (no pipeline changes). Reuses NB02 functions.

Computes, for every OK animal (annotated window, HR 300-700, >=10 beats), BOTH:
  qt_template  = _measure_one() on the AVERAGED TEMPLATE  (the old method)
  qt_per_beat  = mean of _qt_per_beat() over individual beats (NB01 method, current)
to show exactly where the 40 ms floor comes from and whether per-beat fixes it.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB = "02_beat_averaging_and_clustering.ipynb"
PUB_QTC_LOW, PUB_QTC_HIGH = 34, 60     # PART 3 pass band

# ---- reuse pipeline ----
nb = json.load(open(NB, encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
helper_src = next(s for s in codes if "def extract_morphology_features" in s)
P = {"__name__": "__p__"}
exec(compile(next(s for s in codes if s.startswith("import re")), "setup", "exec"), P)
exec(compile(helper_src, "helpers", "exec"), P)
bc = next(s for s in codes if "def mitchell_qtc" in s)
a = bc.index("QTC_FORMULA_NAME ="); b = bc.index("per_animal", a)
exec(compile(bc[a:b], "mitchell", "exec"), P)
FS = P["FS"]; DATA_DIR = P["DATA_DIR"]; OUTPUTS_DIR = P["OUTPUTS_DIR"]; FIGURES_DIR = P["FIGURES_DIR"]
parse_filename = P["parse_filename"]; parse_header_and_layout = P["parse_header_and_layout"]
load_ecg_file = P["load_ecg_file"]; find_baseline_window = P["find_baseline_window"]
bandpass_filter = P["bandpass_filter"]; detect_r_peaks = P["detect_r_peaks"]
average_beats = P["average_beats"]; _measure_one = P["_measure_one"]; _qt_per_beat = P["_qt_per_beat"]
mitchell_qtc = P["mitchell_qtc"]


def per_beat_qt(beats, t_ms):
    vals = [_qt_per_beat(b, t_ms) for b in beats]
    vals = [v for v in vals if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else np.nan,
            float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            len(vals))


rows, templates = [], {}
for path in sorted(DATA_DIR.glob("*.txt")):
    aid, _ = parse_filename(path)
    if aid is None:
        continue
    try:
        nh, lc, ec, _t = parse_header_and_layout(path)
        v, mk = load_ecg_file(path, nh, ec)
        if v.size == 0 or float(np.median(np.abs(v))) > 50:
            continue
        s, e, *_ = find_baseline_window(mk, len(v))
        if s is None:
            continue
        seg = v[s:e]
        if seg.size < 2 * FS:
            continue
        filt = bandpass_filter(seg)
        pk, inv, hr, _ = detect_r_peaks(filt)
        if inv:
            filt = -filt
        if len(pk) < 10 or not (300 <= hr <= 700):
            continue
        rr_mean = float(np.mean(np.diff(pk) * (1000.0 / FS)))
        template, t_ms, beats = average_beats(filt, pk)
        qt_template = _measure_one(template, t_ms)[5]                 # OLD: template method
        qt_pb, qt_pb_std, n_valid = per_beat_qt(beats, t_ms)         # NB01 per-beat
        rows.append(dict(animal_id=aid, hr=round(hr, 1), rr_mean_ms=round(rr_mean, 1),
                         qt_template_ms=round(qt_template, 1) if not np.isnan(qt_template) else np.nan,
                         qt_per_beat_ms=round(qt_pb, 1) if not np.isnan(qt_pb) else np.nan,
                         qt_per_beat_std=round(qt_pb_std, 1) if not np.isnan(qt_pb_std) else np.nan,
                         n_perbeat_valid=n_valid, n_beats=len(pk)))
        templates[aid] = (template, t_ms, qt_template, qt_pb)
    except Exception:
        continue

df = pd.DataFrame(rows).sort_values("animal_id").reset_index(drop=True)
N = len(df)

# =====================================================================
print("=" * 72)
print("PART 1 — THE 40 ms FLOOR")
print("=" * 72)
qt_t = df["qt_template_ms"]
floor_exact = int((qt_t == 40).sum())
floor_1ms = int(((qt_t - 40).abs() <= 1).sum())
print(f"n animals (OK set)                          : {N}")
print(f"\n[TEMPLATE method qt_template_ms]")
print(f"  exactly 40 ms                             : {floor_exact}")
print(f"  within 1 ms of 40                         : {floor_1ms}")
print(f"  range  min/max/mean/std                   : "
      f"{qt_t.min():.1f} / {qt_t.max():.1f} / {qt_t.mean():.1f} / {qt_t.std():.1f}")
qt_p = df["qt_per_beat_ms"].dropna()
print(f"\n[PER-BEAT method qt_per_beat_ms  (= current NB02 qt_ms)]")
print(f"  within 1 ms of 40                         : {int(((qt_p-40).abs()<=1).sum())}")
print(f"  range  min/max/mean/std                   : "
      f"{qt_p.min():.1f} / {qt_p.max():.1f} / {qt_p.mean():.1f} / {qt_p.std():.1f}")

# histogram
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
axes[0].hist(qt_t.dropna(), bins=np.arange(30, 105, 2.5), color="indianred", edgecolor="black")
axes[0].axvline(40, color="black", lw=2, ls="--", label="40 ms floor")
axes[0].set_title(f"TEMPLATE method (old): {floor_1ms}/{N} pinned at ~40 ms")
axes[0].set_xlabel("qt_template_ms"); axes[0].set_ylabel("animals"); axes[0].legend()
axes[1].hist(qt_p, bins=np.arange(30, 105, 2.5), color="steelblue", edgecolor="black")
axes[1].axvline(40, color="black", lw=2, ls="--", label="40 ms")
axes[1].axvspan(60, 80, color="green", alpha=0.12, label="C57BL/6J raw 60-80 ms")
axes[1].set_title(f"PER-BEAT method (current): mean {qt_p.mean():.0f} ms, none floored")
axes[1].set_xlabel("qt_per_beat_ms"); axes[1].legend()
fig.suptitle("QT distribution — template method floors at 40 ms; per-beat does not", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIGURES_DIR / "qt_histogram_diagnosis.png", dpi=150)
plt.close(fig)
print("\nSaved figures/qt_histogram_diagnosis.png")

# print the exact QT code
def extract_block(src, start_key, end_key):
    i = src.index(start_key); j = src.index(end_key, i)
    return src[i:j].rstrip()
print("\n--- NB02 _measure_one() QT lines (TEMPLATE method) ---")
print(extract_block(helper_src, "    baseline = float(np.median(sig[t_ms < -70])",
                     "    return r_amp"))
print("\n--- NB02 _qt_per_beat() (PER-BEAT method, NB01-derived) ---")
print(extract_block(helper_src, "def _qt_per_beat", "def extract_morphology_features"))
print("\n--- how extract_morphology_features picks qt_ms ---")
emf = extract_block(helper_src, "    qt_ms = qt_template", "        qt_std_ms          =")
print(emf)

# floored animals (template) for the annotated figure
floored = df[(df.qt_template_ms - 40).abs() <= 1].sort_values("qt_per_beat_ms", ascending=False)
pick = floored["animal_id"].head(3).tolist()
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, aid in zip(axes, pick):
    template, t_ms, qt_t1, qt_p1 = templates[aid]
    baseline = float(np.median(template[t_ms < -70])) if (t_ms < -70).any() else 0.0
    r_amp = float(template[int(np.argmax(template))]); tol = 0.05 * r_amp
    ax.plot(t_ms, template, color="steelblue", lw=1.4)
    ax.axhline(baseline, color="gray", ls=":", lw=1)
    ax.axhspan(baseline - tol, baseline + tol, color="orange", alpha=0.15, label="template tol (+/-5% R)")
    ax.axvline(0, color="red", ls="--", lw=1, label="R peak (t=0)")
    ax.axvline(qt_t1, color="black", lw=2, label=f"template T-end={qt_t1:.0f} ms (FLOOR)")
    if not np.isnan(qt_p1):
        ax.axvline(qt_p1, color="green", lw=2, ls="-.", label=f"per-beat QT={qt_p1:.0f} ms")
    ax.axvspan(40 - 0.6, 40 + 0.6, color="black", alpha=0.08)
    ax.set_title(f"Animal {aid}"); ax.set_xlabel("ms from R"); ax.set_ylabel("mV")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.suptitle("Why the template floors at 40 ms: the averaged T-wave is already within "
             "+/-5% of baseline at t=40 ms (search starts at 40)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIGURES_DIR / "qt_floor_diagnosis.png", dpi=150)
plt.close(fig)
print("Saved figures/qt_floor_diagnosis.png  (animals " + str(pick) + ")")

# =====================================================================
print("\n" + "=" * 72)
print("PART 2 — PER-BEAT METHOD VERIFICATION")
print("=" * 72)
def report(aid):
    r = df[df.animal_id == aid]
    if r.empty:
        print(f"  animal {aid}: not in OK set"); return
    r = r.iloc[0]
    qtc_pb = mitchell_qtc(r.qt_per_beat_ms, r.rr_mean_ms)
    print(f"  animal {aid}: per-beat QT={r.qt_per_beat_ms:.1f}+/-{r.qt_per_beat_std:.1f} ms "
          f"(n={r.n_perbeat_valid} beats), RR={r.rr_mean_ms:.0f} ms -> QTc={qtc_pb:.1f} ms  "
          f"[template QT was {r.qt_template_ms:.0f}]")
print("Animal 201 (NB01 prototype):")
report(201)
print("\n3 template-floored animals via per-beat:")
for aid in pick:
    report(aid)
print(f"\nC57BL/6J published: raw QT ~60-80 ms, QTc ~34-43 ms.")

# =====================================================================
print("\n" + "=" * 72)
print("PART 3 — SIDE-BY-SIDE TABLE")
print("=" * 72)
extreme_floored = floored["animal_id"].head(5).tolist()
not_floored = df[(df.qt_template_ms - 40).abs() > 5].sort_values("animal_id")["animal_id"].head(5).tolist()
sel = [201] + extreme_floored + not_floored
out = []
for aid in sel:
    r = df[df.animal_id == aid]
    if r.empty:
        continue
    r = r.iloc[0]
    qtc_t = mitchell_qtc(r.qt_template_ms, r.rr_mean_ms)
    qtc_p = mitchell_qtc(r.qt_per_beat_ms, r.rr_mean_ms)
    out.append(dict(animal_id=aid, qt_template_ms=r.qt_template_ms, qt_per_beat_ms=r.qt_per_beat_ms,
                    qtc_template=round(qtc_t, 1) if not np.isnan(qtc_t) else np.nan,
                    qtc_per_beat=round(qtc_p, 1) if not np.isnan(qtc_p) else np.nan,
                    published_range_pass=bool(not np.isnan(qtc_p) and PUB_QTC_LOW <= qtc_p <= PUB_QTC_HIGH),
                    group=("prototype" if aid == 201 else
                           "template_floored" if aid in extreme_floored else "template_ok")))
tab = pd.DataFrame(out)
tab.to_csv(OUTPUTS_DIR / "qt_diagnosis_table.csv", index=False)
print(tab.to_string(index=False))
print("\nSaved qt_diagnosis_table.csv")
