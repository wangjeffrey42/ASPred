# Pairwise epistasis analysis

Moved here from a session scratchpad (tmp dirs get cleared). All scripts resolve
paths relative to their own location, so they run in place:

```
cd analysis/epistasis/scripts && python3 s1_features.py
```

`cache/` holds intermediate pickles and is regenerable by rerunning s1 → s20 in order.
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
| floor | 002-S21F2, AZD8895, S309 | `[0.00, 3.3–4.3]` | 49–76% exactly at 0 |
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

Resolving this means establishing the sort direction per antibody from the Library M raw
processing code (`RBD Data/LibraryM_raw_data_processing/`) and, if confirmed, flipping the
floor set before any cross-set comparison.

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
- **Measurement-noise scope.** The noise floor used when judging whether interaction
  magnitudes are recoverable came from 682 repeated genotypes, estimated over **train and
  test rows together**, not test alone. It is therefore not a clean held-out noise estimate,
  and a test-only recomputation would be the honest version. This figure comes from session
  notes; no script in this directory reproduces it.
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

