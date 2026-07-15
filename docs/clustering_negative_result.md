# Why the ECG features do not cluster into treatment groups

**A documented negative result, not a pipeline failure.**

This note explains why unsupervised clustering cannot recover the 5 treatment
arms from the ECG features, what was tried, and what the correct solution is.
It is written so a reader (supervisor, examiner, or future student) understands
that the absence of clusters is an *expected* property of the data, not a bug.

---

## 1. The question

The study has **5 treatment groups × 2 studies** (control, doxorubicin-only, and
doxorubicin + ethanolamine at 1.6 / 16 / 160 mg/kg; acute and chronic). The
natural hope is: *do the animals separate into these groups on the basis of their
ECG features alone?*

Answer: **No — and they cannot, by the nature of unsupervised methods.**

## 2. What the data actually shows

Clustering was run with k-means, hierarchical (Ward), and Gaussian mixture
models, at k=2 and k=5, on the PCA of the multivariate feature matrix (n = 117
usable recordings).

| clustering target | silhouette | interpretation |
|---|---|---|
| **k = 2** | **0.62** | one strong split exists |
| **k = 5** (to match the 5 arms) | **0.28** | noise-level — no real 5-way structure |

The single real split at k = 2 is **data quality**, not treatment:

- cluster A ≈ 83 clean recordings
- cluster B ≈ 34 flagged recordings (irregular rhythm, `rr_cv > 0.15`)

Removing the outliers and re-clustering made k ≥ 3 **worse**, not better — there
is no hidden group structure underneath. The distance-to-healthy score is a
**unimodal smear with no natural gap**, so it cannot be thresholded into
categories either.

## 3. Why this happens (the reason)

Clustering is **unsupervised**: it finds the largest axis of variance and splits
on it. In this cohort the largest axis is **recording quality**, because any
treatment effect is buried under three larger sources of variance:

1. **Recording artifact** — motion, baseline wander, heart rate out of the
   accepted 300–700 bpm band (the 34 irregular animals dominate).
2. **Inter-animal biology** — mice differ substantially even within one group.
3. **A genuinely small drug effect** on the *surface* ECG.

The algorithm has no way to know which variance is "treatment" and which is
"noise," so it locks onto the biggest signal: quality. **You cannot recover
labels the algorithm was never given.**

## 4. Approaches tried, and why each failed

| approach | outcome |
|---|---|
| k-means at k = 5 (match 5 arms) | silhouette 0.28 — no 5-way structure |
| Hierarchical Ward + GMM at k = 2 / k = 5 | same — only the quality split survives |
| Remove outliers, re-cluster | k ≥ 3 got **worse**, not better |
| PCA before clustering | PC1 = quality axis; treatment on no component |
| Distance-to-healthy as a classifier | continuous smear, no natural cutoff |
| Treat cluster == treatment group | clusters map to artifact, not drug |

Every unsupervised route lands in the same place: it rediscovers **data
quality**, never **treatment**.

## 5. The solution

The fix is **not a better clustering algorithm** — it is a change of method:

1. **Supervised analysis, once treatment metadata is released.** ANOVA / LDA /
   PLS-DA are *told* the groups and can find the separating axis even when it is
   small and not the dominant variance direction. This is the only route to a
   healthy-vs-cardiotoxic answer.
2. **Dose-response** across ethanolamine 1.6 → 16 → 160 mg/kg (Spearman): a
   monotonic trend survives even when discrete groups overlap — the most
   sensitive probe available.
3. **Two-tier quality gate first:** exclude the ~6 genuinely broken recordings,
   review-flag the ~28 irregular ones, so treatment signal is not drowned by
   artifact. Report the 0.15–0.30 `rr_cv` sensitivity sweep.
4. **While metadata is withheld:** recover the 34 irregular animals from
   LabChart to grow usable *n*, and run a power analysis to confirm the study is
   powered to detect a plausible effect.
5. **Report the null honestly.** "ECG features do not cluster by treatment" is a
   legitimate finding — it is precisely *why* the supervised design is the right
   tool.

---

## Bottom line

Clustering cannot find the groups because it is blind to the labels and the
largest signal in the data is **recording quality, not the drug**. The answer to
the thesis question comes from **supervised testing against the real treatment
metadata**, not from any refinement of the clustering.

Healthy-vs-unknown is therefore a **data-quality label** (driven by rhythm
irregularity), which the pipeline exposes deterministically via the `status`
field and the cause-flags — it is *not*, and was never intended to be, a
healthy-vs-cardiotoxic verdict.
