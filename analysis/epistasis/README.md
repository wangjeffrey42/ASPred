# Pairwise epistasis analysis

Moved here from a session scratchpad (tmp dirs get cleared). All scripts resolve
paths relative to their own location, so they run in place:

```
cd analysis/epistasis/scripts && python3 s1_features.py
```

`cache/` holds intermediate pickles and is regenerable by rerunning s1 → s21 in order.
`results/` holds the reported tables.

Scripts need the `aspred` conda env (`conda activate aspred`).

## The question

Does the paired-encoder ESM-2 affinity model encode interactions between RBD mutations,
and does it add anything over a simple linear model? Short answers: the interactions are
real and the model recovers their *ordering* but almost none of their *magnitude*; and no,
it does not beat an additive baseline on held-out data.

## Headline result: a 47-parameter linear model matches the trained model

`s20` scores both on the same held-out test rows (17,130 per antibody). The additive Tobit
is fit on train genotypes only, so neither model has seen the test split. Model numbers are
the run5 checkpoint, consistent with `results/paired_affinity_metrics.csv` (`test_full`).

| group | additive ρ | model ρ | diff |
|---|---|---|---|
| 002-S21F2 | 0.647 | 0.655 | −0.008 |
| AZD1061 | 0.516 | 0.504 | +0.012 |
| AZD8895 | 0.662 | 0.646 | +0.017 |
| LY-CoV1404 | 0.667 | 0.644 | +0.023 |
| REGN10987 | 0.575 | 0.581 | −0.005 |
| S309 | 0.621 | 0.576 | **+0.045** |
| mean | **0.615** | **0.601** | **+0.014** |

47 parameters per group (intercept + 46 allele indicators). The additive model is ahead in
4 of 6 groups and behind by at most 0.008 in the other two. Whatever the trained model has
learned beyond per-allele additivity does not show up as held-out rank accuracy.

This is the context for everything below: the interaction structure the model *does* carry
is real but too compressed to buy it anything.

## Pipeline

| script | what it does |
|---|---|
| `s1_features.py` | allele indicators from train genotypes (>=1% freq), censoring bounds per antibody |
| `s2_additive.py` | additive Tobit fit per antibody |
| `s3_diag.py` | design diagnostics; shows the negative interior R2 is a metric artifact, not a bug |
| `s4_cells.py` | alias groups, four-state cell counts per pair, three-way feasibility |
| `s5`–`s6` | first and second pairwise scans (superseded by s11) |
| `s9_diag_reps.py` | interior identifiability on the alias-collapsed design |
| `s10_selection.py` | selection-on-outcome severity: allele depletion and effect attenuation |
| `s11_scan3.py` | **canonical scan**: screen/confirm on disjoint genotype halves, alias-collapsed base |
| `s12_report.py` | per-group report, overlaps, artifact flags |
| `s13_tables.py` | CSV exports |
| `s14_crossgroup.py` | cross-group sharing with uniform and position-matched permutation nulls |
| `s15_model_epistasis.py` | model-implied interaction per pair via ASPred-Affinity forward passes |
| `s16_model_vs_fitted.py` | correlates model-implied against fitted coefficients |
| `s17_model_main_effects.py` | model-implied **main** effects, same backgrounds |
| `s18_main_vs_interaction_slope.py` | main-effect vs interaction attenuation side by side |
| `s19_excess_vs_censoring.py` | tests whether excess interaction compression tracks censoring |
| `s20_additive_vs_model.py` | held-out additive-vs-trained-model comparison (headline table) |
| `s21_replicate_noise.py` | measurement-noise floor from the MABAa/MABAb replicate pair |

## Decisions that are easy to get wrong

**Three allele pairs are collinear at |r| > 0.99 and are not separable.** One
representative each, used everywhere including inside interaction terms:

| representative | also stands for |
|---|---|
| `373P>S` | `375F>S` |
| `376T>A` | `405D>N` |
| `477N>S` | `478K>T` |

46 indicators collapse to 43. A pair drawn from within one group is not estimable and
is excluded. 903 representative pairs → **867** after requiring all four cells >= 50 in
both genotype halves.

**Collapse the aliases before judging interior identifiability.** An earlier pass
flagged AZD1061 and S309 as rank-deficient on interior rows (cond 4e14, 1e17). That was
the aliasing, not censoring — the interior design still held all 46 indicators. After
collapsing, all six interior designs are full rank at cond 67–75.

**S309 and AZD8895 are dropped from the interior-only comparison**, on selection
grounds rather than rank. For their top alleles the main effect *is* pushing the
response past the bound, so conditioning on interior rows selects on the outcome and
removes the effect being estimated. 486P>F loses 83% of its mass from the interior
subset and 356T>K 80% (P(censored | 356T>K) = 95%); nothing in the other four groups
exceeds 50%. S309's additive effects attenuate to 0.16x of their Tobit values, and
373P>S, 332I>V and 460N>K attenuate to ~0. The interior-only control is unavailable
precisely where censoring is strongest.

**Screening and inference use disjoint genotype halves**, so the reported p-values are
not selection-contaminated.

## Canonical checkpoint

`models/paired_affinity/run5_rbd_split_continued` (see its `README_STATUS.txt`); run6
is a plateau-confirmation run and is not canonical.

## Model-implied vs fitted interactions

`s15` scores the four allele corners of each of the 867 pairs on 16 backgrounds drawn
from held-out test genotypes (14,593 unique RBD sequences after dedup, 55,488 corner
evaluations per antibody) and forms `d = y11 - y10 - y01 + y00`. Both `d` and the Tobit
interaction coefficients are in affinity units, so the magnitude ratio is interpretable.

The model ranks interactions better than chance in most groups but understates them by
4–20x: RMS model-implied / RMS fitted runs 0.045 (S309) to 0.250 (AZD1061). Rank
agreement is good for REGN10987 (Spearman 0.686, 0.800 on FDR-significant terms) and
AZD1061 (0.745 / 0.833), absent for AZD8895 (0.325, n.s.), and for S309 the Pearson is
0.050 while the Spearman is 0.455 — it gets some of the ordering, almost none of the
scale. This matches `results/epistasis_residual_test.png`, where the model's
non-additive component spans ~7x less than the measured one.

## Is the attenuation specific to interactions?

No. `s17`/`s18` repeat the comparison for main effects (43 representatives, same 16
backgrounds, `d = y(alt) - y(ref)`). Main-effect slopes are 0.12–0.33, median 0.26 —
nowhere near 1. The model understates main effects by roughly the same 3–4x it
understates interactions, so the shrinkage is a global property of the fit, not
something that happens only to interaction structure.

Interactions are nonetheless attenuated *further*: median RMS ratio 0.288 for main
effects vs 0.184 for interactions, a 1.57x additional shrinkage, rising to 2.35x for
AZD8895 and 3.34x for S309. Direction is recovered far better for main effects
(Pearson 0.82–0.95 in every group) than for interactions (0.05–0.78).

Read together with the slopes: the model learns which direction each allele pushes and
roughly orders them, but compresses every effect toward zero, and compresses
interactions hardest.

## Is the excess compression censoring-driven?

No. `s19` regresses the excess ratio (main-effect RMS ratio / interaction RMS ratio) on
censoring across the six groups. Censoring gives Pearson 0.613 (p = 0.20, permutation
p = 0.199 over all 720 orderings), and the apparent trend is carried entirely by S309:
among the other four groups, which span 55.0-60.7% censoring, the correlation is +0.089
(p = 0.91). AZD8895 settles it directly - it is the *least* censored group at 49.2% and
has the second-highest excess compression at 2.35x.

What does predict excess compression is the size of the effects themselves: RMS fitted
main effect gives Pearson 0.907 (p = 0.013) and RMS fitted interaction 0.836 (p = 0.038).
Groups with larger true effects are compressed proportionally harder. That is the
signature of a shrinkage/regression-to-the-mean property of the trained model, i.e.
architectural (or a training-objective effect), not a consequence of censored labels.

Caveat on power: n = 6 groups. These correlations rank the explanations but cannot
settle them; the AZD8895 counterexample is the sturdier part of the argument.

## Why Tobit and not OLS

Every group has a hard point mass on one bound — 49% to 76% of rows (`group_summary.csv`),
so OLS is fitting a response that cannot move in one direction over half the time. All
reported coefficients are Tobit, with the bound taken from the empirical point mass
(`s1_features.py`).

OLS was run in an earlier session and gave slightly *lower* model-vs-fitted correlations
throughout. The ordering of the conclusions did not change. That comparison was not kept as
a result file, so **numbers in any pre-Tobit notes will not match the tables here** — the
CSVs in `results/` are authoritative.

## The six groups sit on two incompatible scales

This is the most load-bearing unresolved issue in the analysis.

`RBD Data/File_Details.csv` (**parent-repo-relative** — see below) labels all six antibodies
identically: Library M, `Enrichment at 100nM`, column `ph_mean`. The metadata draws no
distinction between them. The values do:

| set | antibodies | range | point mass |
|---|---|---|---|
| floor | 002-S21F2, AZD8895, S309 | `[0.00, 3.3–5.2]` | 49–76% exactly at 0 |
| ceiling | AZD1061, LY-CoV1404, REGN10987 | `(−2.8, 1.00]` | 55–61% exactly at 1 |

The upstream `README.txt` gives the mechanism: an enrichment score depends on which
subpopulation was sorted. An evading-population sort yields `s` (escape; 0 = none, may
exceed 1); a binding-population sort yields `1 − s` (1 = full binding, may go negative).
The 3/3 split matches that exactly, which means **the sign of the phenotype is inverted
between the two sets** — a positive coefficient reads as "more escape" for the floor set and
"more binding" for the ceiling set.

This is inferred from the bound structure. Nothing upstream records which sort was used.

**What it affects.** `s14_crossgroup.py:43` compares `np.sign(coef[a][t]) == np.sign(coef[b][t])`
with no harmonization, so sign agreement across the two sets is computed on opposed scales:

- **Unaffected:** every per-group quantity — all attenuation slopes, RMS ratios, the excess
  compression analysis, the headline additive comparison. These are ratios or correlations
  within one group.
- **Unaffected:** the strongest sharing result. AZD1061/REGN10987 (overlap 15, sign
  agreement 14/15, p = 0.0) is *within* the ceiling set.
- **Affected:** the four cross-set pairs with nonzero overlap — 002-S21F2/AZD1061 (3),
  002-S21F2/LY-CoV1404 (1), AZD1061/AZD8895 (1), LY-CoV1404/S309 (1). Their
  `sign_agree_frac` in `crossgroup_overlap.csv` should be read as uninterpretable rather
  than as evidence either way. All have overlap ≤ 3, so nothing substantive rests on them.

**What it affects outside this directory.** Traced; the numbers were sound throughout, but
several labels were not. The three figure scripts are now fixed; the labeled data splits
are not.

`src/phenotype_convention.py` is now the single source of truth for which antibody sits on
which scale. Scripts import it rather than restating the sets, and call `pc.verify(df)`,
which asserts the declared point mass is really there — so a reprocessed or rescaled
upstream file fails loudly instead of silently flipping a label.

- `src/plot_decile_calibration.py` — **fixed.** It computed every decile, spread and
  ranking accuracy inside `groupby("antibody_name")`, so **the numbers were always valid**
  and are unchanged by the fix; only the labels were wrong. The x-axis read "Predicted
  binding rank (low to high)" and the y-axis "Mean measured affinity" for all six panels,
  but a rising curve means *more escape* for the floor trio. The six-panel figure is now
  laid out one convention per row (ceiling top, floor bottom), with axis wording from
  `pc.measured_quantity()` and panels colour-coded by convention. The **overlay figure**
  min-max normalized all six onto one panel "so the shapes are comparable", presenting two
  opposed phenotypes as a single trend; it is now **split into one panel per convention**,
  since normalizing is only meaningful among curves that rise in the same direction — though
  that split fixes the sign error **only**, and the figure still must not be read as a shape
  comparison between antibodies (see the caveat below). Three of the four printed quantities
  — `spread`, `monotonic` and `rank acc` — are convention-**invariant**: a flip reverses the
  decile order and negates every mean, and the two cancel exactly (verified bit-identical).
  Only the decile means are convention-dependent.

  **Invariant is not the same as comparable.** An earlier draft of this section called
  `rank acc` "the one quantity comparable across all six"; that was wrong. It is computed
  only on pairs whose measured values differ, and that share tracks censoring — 42.6% of
  pairs for S309 (75.8% censored) against 76.1% for AZD8895 (48.7% censored),
  Pearson(rank acc, censoring) = +0.68 over the six. Restricted to interior-vs-interior
  pairs the ordering **reverses**: S309 goes from best (84.2%) to worst (55.3%), REGN10987
  from fifth to first. So the 84.1% annotated on S309's panel is mostly a statement about
  its point mass, and it sits awkwardly beside S309's lowest model ρ (0.5765) and worst
  additive-vs-model gap (+0.045) reported above. Use the per-group Spearman for model
  quality, not this number.
  *Caveat on the overlay.* Within one panel the normalized curve is very nearly a rescaling
  of the per-decile uncensored fraction (|Pearson| = 0.96–1.00 in every antibody), so what
  the eye reads as "shape" is largely the censoring mechanism rather than graded
  calibration. In the floor panel the three censoring fractions are 48.7 / 56.9 / 75.8%, so
  S309's late rise against AZD8895's early rise is a point-mass artifact. The SEM bars
  collapse near the bound and understate uncertainty in exactly the deciles that drive the
  shape.
- `src/plot_binds_escaped_split.py` — **fixed.** It labelled `label == 1` as **"Binds"**
  for all six groups; for the floor trio `affinity > median` means more escape, so the
  legend was inverted for three of six antibodies. Each bar now carries its own class from
  `pc.positive_label()` ("Binds" / "Escaped"), bars are grouped and bracketed by convention
  so the two scales never interleave, and the **pooled total was removed** — it summed
  "Binds" counts and "Escaped" counts into one number.
- `src/plot_measured_vs_predicted_heatmap.py` — **fixed.** It sorted its shared row order by
  "mean MEASURED binding across antibodies" and took one color scale from all antibodies
  pooled, averaging across opposed signs and across ranges `[0, 5.2]` vs `(−2.8, 1]`. It
  now renders one block per convention, each with its own row order, colour scale and
  colorbar label.
- `data/paired/{group,variant}_holdout_labeled/` inherit the same convention: `label == 1`
  means "escapes" for the floor trio and "binds" for the ceiling trio. Per-group classifier
  metrics in `results/paired_classifier_test_metrics.csv` are valid **within** a group (each
  antibody is in training, so the model learns its convention), but the **`pooled` row mixes
  the two conventions** and is not a single binds-vs-escapes number. Two further cautions on
  that CSV, neither of which harmonizing the label would fix:
  - *The pooled row is also inflated by between-group structure*, independently of the
    convention problem. 83.3% of pooled pairs are cross-antibody, per-group prevalence spans
    0.242–0.606, and the model sees the antibody sequence, so base-rate separation alone
    lifts it: pooled AUROC 0.8620 against a per-group mean of 0.8460, and pooled F1 0.8027
    against 0.7911 (exceeding 4 of the 6 per-group F1 values).
  - *Per-group values are still not a league table.* F1 is not invariant to which class is
    positive, so S309's 0.693 (prevalence 24%, positive = "any escape") and LY-CoV1404's
    0.863 (prevalence 61%, positive = "full binding") are not on a common footing.

  `group_holdout_labeled/` is the live hazard — see the warning below.
- **Unaffected:** `s15`–`s19`. Each loops `for ab in ABS` and computes Pearson, Spearman,
  slope and RMS within one group, where a sign flip negates the fitted coefficient and the
  model-implied value together — all four statistics are algebraically invariant to it. The
  `s19` effect-size correlation uses RMS magnitudes only, so the r = 0.907 shrinkage result
  does not depend on sign convention either.

### ⚠️ `data/paired/group_holdout_labeled/` transfers an inverted label — do not train on it

This split asks a model to generalize to an **unseen antibody**, and the convention is not
held constant across the split. `make_group_cutoff_labels.py` reuses the antibody
assignment from `make_paired_splits.py` (test = S309, val = AZD8895, train = the other
four), which lands this way:

| split | antibodies | convention | rows | `label == 1` means | % positive |
|---|---|---|---|---|---|
| train | AZD1061, LY-CoV1404, REGN10987 | ceiling | 513,681 | **Binds** | 55.0–60.7% |
| train | 002-S21F2 | floor | 171,227 | **Escaped** | 42.6% |
| val | AZD8895 | floor | 171,227 | **Escaped** | 50.0% |
| test | S309 | floor | 171,227 | **Escaped** | 23.9% |

**75% of training rows define `label == 1` as more binding; 100% of validation and test
rows define it as more escape.** A classifier trained here learns the majority training
sense of the positive class and is then selected and scored on antibodies where that class
is the opposite phenotype. The failure is quiet: no column *states* which sense a row
carries (`cutoff` and `antibody_name` do distinguish the sets, but nothing asserts the
meaning and no consumer checks it), and the run will report a plausible-looking accuracy.

**What to expect, and what it will not tell you.** An earlier draft of this section
predicted anti-correlation — "near or below chance". That is not supported, and the repo's
own evidence contradicts it. The same antibody assignment has been run twice in regression
form: `ABANDONED_run1_antibody_holdout_identity_shortcut` gave val Spearman
0.0095 / −0.0598 / 0.0357 / 0.0077 / −0.0348 / −0.0300, and run7 (warm-started) gave val
AZD8895 **−0.0525** and test S309 **+0.0263** — two floor antibodies with opposite signs,
both ≈ 0. That is near-chance, not a systematic inversion. Note also that "below chance" is
undefined for accuracy here: test prevalence is 23.9%, so the majority-class baseline is
76.1%, not 50%. AUROC is the metric that is exactly antisymmetric under a label flip
(AUROC → 1 − AUROC), so state results that way if inversion is what you want to detect.

Where the inversion *does* show up in those runs is the mean level, not the ranking: run7
predicts a mean of 0.904 for AZD8895 against a true mean of 0.109 — a ceiling-scale
prediction for a floor-scale antibody.

Crucially, **a poor result here is uninterpretable rather than diagnostic.** Label
inversion and the already-documented identity shortcut predict the same observable on this
split, so neither can be attributed. That is the reason not to use it, independent of how
the number comes out.

Two further traps in the same split:

- **002-S21F2 is the only floor antibody in train**, contributing exactly a quarter of the
  rows against three ceiling antibodies. Note the risk is *not* that the model fails to
  represent per-antibody differences — run7's diagnostics record an antibody-vs-RBD
  prediction-variance ratio of 41.5 (run1: 48.8), i.e. predictions are dominated by
  antibody identity. It is that the model has no basis for inferring the sense of an
  antibody it has never seen, and one in-train example of the minority sense is a thin
  basis for learning that the sense varies at all.
- **Val and test are both floor**, so validation will not catch the inversion — it is
  inverted in exactly the same direction as test. Early stopping and model selection will
  look consistent while both are measuring the flipped quantity.

#### The label generators are the proper fix, and they are still wrong

`src/make_group_cutoff_labels.py` and `src/make_variant_holdout_labels.py` both restate
`CEILING_GROUPS = {"AZD1061", "LY-CoV1404", "REGN10987"}` by hand instead of importing
`src/phenotype_convention.py`. More seriously, `make_group_cutoff_labels.py`
**describes the ceiling/floor difference as nothing more than tie-handling at the median**,
and `make_variant_holdout_labels.py` adopts that account by reference ("see
make_group_cutoff_labels.py for the cutoff/tie-handling/AZD1061 rationale"), so the same
reading governs both:

> This is the same median split in both cases -- only the tie-handling at the cutoff value
> flips, because the mass of ties sits on the opposite side of the distribution for ceiling
> groups vs. floor groups

**That description is wrong, and it is the reason the bug propagated.** Two separate things
are going on and the docstring collapses them into one:

1. *Tie-handling* — `>` vs `>=` at the cutoff. The `>=` tie-break is necessary: with >50%
   of rows exactly on the bound, `> cutoff` would be degenerate for ceiling groups. But
   calling the result a "median split" of a graded phenotype overstates what comes out
   of it; see the point about saturation below.
2. *Sign inversion* — the underlying phenotype runs in opposite directions between the two
   sets. The cutoff rule neither causes this nor corrects it. It is still there after the
   labels are assigned, which is exactly why `label == 1` means "binds" in one set and
   "escapes" in the other.

**The cutoff is the saturation bound, so `label` is essentially the censoring indicator.**
Verified on the data: for all three ceiling groups the median is exactly 1.0 and
`label == 1 ⟺ affinity == 1.0`; for 002-S21F2 and S309 the median is exactly 0.0 and
`label == 1 ⟺ affinity > 0`. Only AZD8895 has an interior cutoff (0.017354). Three
consequences worth stating plainly:

- The "% positive" column in the table above is, for five of six antibodies, just the
  complement of the censoring fraction — S309's 23.9% positive is 100 − 76.1% censored.
- The task these splits define is **"predict saturation"**, not graded binding. What the
  `pooled` row really mixes is "is saturated" for three groups against "is *not* saturated"
  for three.
- It makes the prescribed fix provably well-defined rather than an arbitrary convention
  choice: flipping the floor set maps both sets onto `label == 1 ⟺ s == 0` (no escape),
  which is exactly the ceiling definition.

Reading (1) as the whole story makes the convention look like a cosmetic detail of the
median split. That framing plausibly propagated into `src/plot_binds_escaped_split.py`,
which is the one figure script that re-implements the median-cutoff rule itself. The other
two mislabelled figures are an independent error: `plot_decile_calibration.py` and
`plot_measured_vs_predicted_heatmap.py` read only
`results/rbd_split_test_predictions.csv` and never touch labels or cutoffs — they simply
called raw affinity "binding".

**The proper fix is to correct these generators and regenerate the labeled splits**, not to
patch consumers downstream. Concretely: import `phenotype_convention` rather than restating
the set, decide one phenotypic direction for `label == 1`, flip the floor set (or
equivalently the ceiling set) so the positive class names one phenotype throughout, record
the chosen direction in a column so no future reader has to infer it, and rewrite
`data/paired/{group,variant}_holdout_labeled/`. Until that is done, every consumer of these
splits has to re-derive the convention by hand, and `group_holdout_labeled/` stays unusable
for its stated purpose.

**This is deliberately not done yet.** Choosing the direction is a content decision, and
regenerating overwrites labeled data that existing results were computed from — see the
caveat on `results/paired_classifier_test_metrics.csv` above. Do it as its own change, with
the metrics regenerated alongside.

`variant_holdout_labeled/` is safer by comparison — every antibody appears in train, val and
test (verified: all 6 in each of 821,868 / 102,714 / 102,780 rows), so each one's convention
is learned rather than transferred — but its pooled metrics still mix the two senses.

Resolving the underlying question means establishing the sort direction per antibody from
the Library M raw processing code (`RBD Data/LibraryM_raw_data_processing/`) and, if
confirmed, flipping the floor set before any cross-set comparison. The figure labels above
are fixed; the data splits and `make_*_labels.py` generators are not.

## Datasets

Three, and only one is used here. Full map in the parent repo's `CLAUDE.md`.

- `data/paired/` — model-ready `(antibody_sequence, rbd_sequence, affinity)` pairs from
  Library M. `rbd_split_{train,val,test}.csv`, 821,868 / 102,714 / 102,780 rows, split by
  RBD genotype with all six antibodies in every split. **This is what the analysis uses.**
- `data/rbd_dms/` — the eight unzipped `MABAa_*` Library M phenotype CSVs these were built
  from.
- `RBD Data/` — **in the parent repo, one level above this one**, not tracked here. The full
  upstream release: Libraries B, C and M zipped, plus `File_Details.csv` and the raw
  processing folders. Any path beginning `RBD Data/` in this README is parent-relative.

## Caveats

- **n = 6 groups.** Every cross-group correlation — including the r = 0.907 effect-size
  result that carries the shrinkage conclusion — has six points. These rank explanations;
  they do not establish them. The AZD8895 counterexample (least censored, second-most
  compressed) is the sturdier half of the censoring argument because it is a direct
  contradiction rather than a trend.
- **The two-scale problem above** is unresolved and bounds what cross-group claims mean.
- **Measurement noise: the "682 repeated genotypes" figure does not reconstruct. Do not
  cite it.** No scope over the paired data yields 682. Repeated genotypes number 181
  (train), 29 (test), 20 (val), 210 (train+test), 230 (all splits); the source
  `antibody_rbd_affinity_pairs.csv` has 229 genotypes measured twice and 1 measured three
  times — 230 in total, 1,380 duplicate `(antibody, genotype)` keys. The figure appears to
  be an artifact of an earlier session's scope choice. It was also the wrong basis: the
  paired dataset is built from **replicate `a` only** (`build_paired_dataset.py`), so
  within-dataset duplicates are incidental re-measurements, not the designed replicate.

  `s21` gives the defensible version from the actual technical replicate, MABAa vs MABAb,
  on **126,062 shared genotypes per antibody** (`results/replicate_noise.csv`).
  Per-measurement sd is 0.055–0.080 overall and **0.082–0.121 interior (median 0.102)**,
  with replicate Pearson 0.867–0.898.

  This noise floor is the *same order* as the RMS fitted interaction coefficients
  (0.054–0.163), which sounds fatal and is not: each coefficient is estimated from ~10⁵
  rows, so the reported Tobit SEs (~0.006–0.008) sit well below the effect sizes and the
  FDR-significant interactions remain identified. The noise floor bounds **single-variant
  prediction**, not the aggregate coefficient estimates. Note MABAb covers a different
  locus set than MABAa, so this is a genotype-overlap comparison, not a matched re-assay.
- **Model-implied effects use 16 backgrounds**, drawn from held-out test genotypes. Both `d`
  and the Tobit coefficients are in affinity units, so ratios are interpretable, but the
  background sample is small and `model_interaction_sd` in
  `results/model_implied_interactions.csv` is often the same order as the mean.
- **Two groups have no interior-only control.** S309 and AZD8895 are excluded from that
  comparison on selection grounds (see above), which is precisely where censoring is
  worst — so the control is missing where it would matter most.

## Open questions

1. **Does Library C have complete sublandscapes?** It is 7 loci on 7 backgrounds (16,800
   variants) at only 2–21% censoring, against Library M's 49–76%. If the sublandscapes are
   complete it supports interaction estimates that do not depend on Tobit extrapolation at
   all. Note that Library C's antibody files are *fluorescence* (`Fluor at 1uM`,
   `Fluor at 10nM`, two replicates each) and not KD — only its ACE2 file is KD — so this is
   not a drop-in replacement for the Library M target.
2. **Does the shrinkage conclusion survive on better-conditioned data?** The whole
   censoring-vs-shrinkage argument rests on six heavily censored groups. Library C is the
   natural test: if attenuation slopes stay near 0.26 where censoring is 2–21%, that closes
   the question that n = 6 leaves open.
3. **Does a Walsh-Hadamard decomposition need attenuation correction first?** Given global
   slopes of ~0.26 on main effects and a further 1.57× on interactions, a variance spectrum
   computed from model predictions would understate every order, and not uniformly —
   higher orders are compressed harder. The spectrum is probably not interpretable until
   corrected, and what the correct correction is is itself open.

