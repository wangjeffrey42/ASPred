# Pairwise epistasis analysis

Moved here from a session scratchpad (tmp dirs get cleared). All scripts resolve
paths relative to their own location, so they run in place:

```
cd analysis/epistasis/scripts && python3 s1_features.py
```

`cache/` holds intermediate pickles and is regenerable by rerunning s1 → s16 in order.
`results/` holds the reported tables.

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
