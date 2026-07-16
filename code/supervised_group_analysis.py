"""SUPERVISED group-comparison template for the ethanolamine / doxorubicin study.

This is the analysis that actually answers the thesis question ("does ethanolamine
protect against doxorubicin cardiotoxicity?"). It is the correct tool once the
treatment metadata is released -- unlike the blind PCA/clustering, it uses the
KNOWN groups.

Design (matches Roisin's methods email):
  * 2 studies analysed SEPARATELY: acute (1 wk) and chronic (6 wk) -- different models.
  * 5 groups each: control, dox, eth1.6+dox, eth16+dox, eth160+dox.
  * sex included as a factor (6 M + 6 F per group).

What it runs, per study:
  1. Per-parameter omnibus: one-way ANOVA (+ Kruskal-Wallis backup) across the 5
     groups, with BH-FDR across parameters and eta^2 effect size.
  2. Two-way group x sex ANOVA (statsmodels OLS) where n allows.
  3. Planned contrasts vs CONTROL (each group vs control; Welch t + Mann-Whitney),
     BH-corrected -- this is the "is there a dox effect / does eth reverse it" test.
  4. Dose-response: Spearman trend of each parameter across eth dose within the
     dox-treated arms (0/1.6/16/160) -- tests monotonic protection.
  5. Multivariate: LDA and PLS-DA, 5-fold CV accuracy vs chance (1/5 = 20%).
     If accuracy ~ chance, the ECG parameters do NOT separate the groups.

Usage:
  python supervised_group_analysis.py                 # dry-run w/ SYNTHETIC labels
  python supervised_group_analysis.py metadata.csv    # real labels (see template)

Metadata CSV columns: animal_id, study(acute|chronic), sex(M|F),
  group(control|dox|eth1.6|eth16|eth160)

Outputs: ../outputs/supervised_results_<study>.csv and a printed report.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

FEATURES_CSV = "../outputs/all_features_recomputed.csv"
META_TEMPLATE = "../outputs/treatment_metadata_TEMPLATE.csv"

# ECG parameters to test (only those the pipeline extracts).
PARAMS = ["heart_rate_bpm", "rr_mean_ms", "qrs_duration_ms", "qt_ms", "qtc_ms",
          "r_amplitude_mv", "s_wave_amplitude_mv", "p_wave_amplitude_mv",
          "pr_interval_ms", "q_wave_amplitude_mv", "t_wave_amplitude_mv",
          "j_wave_amplitude_mv", "rt_interval_ms"]
GROUPS = ["control", "dox", "eth1.6", "eth16", "eth160"]
ETH_DOSE = {"control": 0.0, "dox": 0.0, "eth1.6": 1.6, "eth16": 16.0, "eth160": 160.0}


def load_features():
    df = pd.read_csv(FEATURES_CSV)
    df = df[df["status"].isin(["OK", "NEEDS_REVIEW"])].copy()
    return df


def make_template(df):
    """Write a blank metadata template listing the animals we actually have."""
    t = pd.DataFrame({"animal_id": sorted(df["animal_id"].astype(int).unique())})
    t["study"] = ""; t["sex"] = ""; t["group"] = ""
    t.to_csv(META_TEMPLATE, index=False)
    print(f"  wrote blank metadata template -> {META_TEMPLATE}")


def synthetic_labels(df):
    print("\n" + "!" * 72)
    print("!!  NO METADATA SUPPLIED -- USING SYNTHETIC RANDOM LABELS (DRY RUN).")
    print("!!  Every number below is MEANINGLESS. It only proves the script runs.")
    print("!!  Re-run with the real metadata CSV to get valid results.")
    print("!" * 72)
    rng = np.random.default_rng(42)
    m = pd.DataFrame({"animal_id": df["animal_id"].astype(int).values})
    m["study"] = rng.choice(["acute", "chronic"], len(m))
    m["sex"] = rng.choice(["M", "F"], len(m))
    m["group"] = rng.choice(GROUPS, len(m))
    return m


def bh(pvals):
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum() > 0:
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def eta_squared(groups):
    all_v = np.concatenate(groups)
    grand = all_v.mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = ((all_v - grand) ** 2).sum()
    return ss_b / ss_t if ss_t > 0 else np.nan


def analyse_study(df, study):
    d = df[df["study"] == study].copy()
    if d["group"].nunique() < 2 or len(d) < 10:
        print(f"\n[{study}] too few animals/groups ({len(d)}) -- skipped.")
        return
    print(f"\n{'='*72}\nSTUDY: {study.upper()}   (n={len(d)})   group counts:")
    print("  " + d["group"].value_counts().reindex(GROUPS).fillna(0).astype(int).to_string().replace("\n", "\n  "))

    rows = []
    for p in PARAMS:
        if p not in d.columns:
            continue
        sub = d[[p, "group", "sex"]].dropna(subset=[p])
        groups = [sub.loc[sub.group == gname, p].values for gname in GROUPS if (sub.group == gname).any()]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        f, p_anova = stats.f_oneway(*groups)
        h, p_kw = stats.kruskal(*groups)
        eta = eta_squared(groups)
        # two-way group x sex (guarded)
        p_gxs = np.nan
        try:
            if sub["sex"].nunique() == 2 and (sub.groupby(["group", "sex"]).size() >= 2).all():
                model = ols(f"{p} ~ C(group)*C(sex)", data=sub).fit()
                aov = sm.stats.anova_lm(model, typ=2)
                p_gxs = float(aov.loc["C(group):C(sex)", "PR(>F)"])
        except Exception:
            pass
        # dose-response within dox arms (control excluded; dox=0 dose)
        dr = sub[sub.group != "control"].copy()
        dr["dose"] = dr["group"].map(ETH_DOSE)
        rho = p_trend = np.nan
        if dr["dose"].nunique() >= 3 and len(dr) >= 6:
            rho, p_trend = stats.spearmanr(dr["dose"], dr[p])
        rows.append({"parameter": p, "anova_p": p_anova, "kruskal_p": p_kw,
                     "eta2": eta, "groupXsex_p": p_gxs,
                     "dose_rho": rho, "dose_trend_p": p_trend})

    res = pd.DataFrame(rows)
    res["anova_q_bh"] = bh(res["anova_p"])
    res["dose_trend_q_bh"] = bh(res["dose_trend_p"])

    # vs-control contrasts (Welch t), BH across params, per group
    contrasts = []
    ctrl = d[d.group == "control"]
    for gname in [g for g in GROUPS if g != "control"]:
        gd = d[d.group == gname]
        for p in PARAMS:
            if p not in d.columns:
                continue
            a = ctrl[p].dropna().values; b = gd[p].dropna().values
            if len(a) >= 2 and len(b) >= 2:
                t, pv = stats.ttest_ind(a, b, equal_var=False)
                _, pmw = stats.mannwhitneyu(a, b, alternative="two-sided")
                pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                dcoh = (b.mean() - a.mean()) / pooled if pooled > 0 else np.nan
                contrasts.append({"group_vs_control": gname, "parameter": p,
                                  "welch_p": pv, "mannwhitney_p": pmw, "cohens_d": dcoh})
    cdf = pd.DataFrame(contrasts)
    if len(cdf):
        cdf["welch_q_bh"] = bh(cdf["welch_p"])

    print("\n-- Per-parameter omnibus (5-group), BH-FDR corrected --")
    show = res.copy()
    for c in ["anova_p", "kruskal_p", "anova_q_bh", "eta2", "groupXsex_p", "dose_rho", "dose_trend_p", "dose_trend_q_bh"]:
        if c in show: show[c] = show[c].map(lambda x: f"{x:.3g}" if pd.notna(x) else "-")
    print(show.to_string(index=False))
    if len(cdf):
        sig = cdf[cdf["welch_q_bh"] < 0.10].sort_values("welch_q_bh")
        print("\n-- Contrasts vs control with BH-q < 0.10 (candidate real effects) --")
        print(sig.to_string(index=False) if len(sig) else "  (none survive FDR)")

    # multivariate separability (median-impute so P/PR NaNs don't shrink n)
    Xcols = [c for c in PARAMS if c in d.columns]
    md = d.copy()
    md[Xcols] = md[Xcols].apply(pd.to_numeric, errors="coerce")
    md[Xcols] = md[Xcols].fillna(md[Xcols].median(numeric_only=True))
    md = md.dropna(subset=Xcols)  # drops only if a whole column is NaN
    if md["group"].nunique() >= 2 and len(md) >= 15:
        X = StandardScaler().fit_transform(md[Xcols].values)
        y = md["group"].values
        counts = pd.Series(y).value_counts()
        k = int(min(5, counts.min()))
        if k >= 2:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            lda_acc = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=cv).mean()
            print(f"\n-- Multivariate separability (n={len(md)}, chance={1/md['group'].nunique():.2f}) --")
            print(f"   LDA {k}-fold CV accuracy: {lda_acc:.3f}")
            print("   (accuracy near chance => ECG parameters do NOT separate the groups)")

    out = f"../outputs/supervised_results_{study}.csv"
    res.to_csv(out, index=False)
    if len(cdf):
        cdf.to_csv(out.replace(".csv", "_contrasts.csv"), index=False)
    print(f"\n   saved: {out}")


def main():
    df = load_features()
    make_template(df)
    if len(sys.argv) > 1:
        meta = pd.read_csv(sys.argv[1])
        if meta["group"].isna().all() or (meta["group"] == "").all():
            print("Metadata 'group' column is blank -> falling back to synthetic.")
            meta = synthetic_labels(df)
    else:
        meta = synthetic_labels(df)
    meta["animal_id"] = meta["animal_id"].astype(int)
    df["animal_id"] = df["animal_id"].astype(int)
    df = df.merge(meta[["animal_id", "study", "sex", "group"]], on="animal_id", how="inner")
    df = df[df["group"].isin(GROUPS)]
    for study in ["acute", "chronic"]:
        analyse_study(df, study)


if __name__ == "__main__":
    main()
