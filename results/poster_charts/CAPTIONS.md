# Poster figures

Regenerate with `python3 src/make_poster_charts.py` from the repo root.
Each figure is a 300 dpi `.png` and a vector `.pdf` — **use the PDF for print**.
All numbers are computed from repo files at render time; nothing is hand-entered.

The charts carry only the marks and the values. Everything else — n, split, method,
caveats — is caption text, below. **Set these captions under the figures**; several of
the figures are misread without them.

| # | File | Caption |
|---|------|---------|
| 1 | `fig1_aspred_roc_curves` | ASPred distinguishes antigen-specific antibodies from heavy-chain sequence alone. ROC on held-out test sets, AUC in legend; ESM-2 650M + LoRA, one model per antigen (test n = 645 / 336 / 2,857, class-balanced). |
| 2 | `fig2_aspred_test_metrics` | F1 and ROC-AUC on the held-out test sets, best-loss checkpoint. Accuracy is within 0.001 of F1 on these balanced sets and is omitted. |
| 3 | `fig3_train_test_gap` | F1 on the held-out test set (filled) against the training set (hollow). Gaps of 0.06–0.13 at these dataset sizes. |
| 4 | `fig4_cross_antigen_calls` | The three antigen models are trained independently and do not make mutually exclusive calls. Influenza test set (n = 645) scored by all three at threshold 0.5; 425 of 645 are called positive by at least one. |
| 5 | `fig5_per_antibody_spearman` | Spearman ρ between predicted and measured affinity, held-out RBD-genotype test set (n = 102,780 pairs; 17,130 per antibody). The pooled bar is inflated by between-antibody separation — the per-antibody bars are the measure of RBD-mutation sensitivity. |
| 6 | `fig6_predicted_vs_measured_pooled` | All 102,780 held-out pairs; darker = more pairs (log scale), dashed = perfect prediction. The two density bands are antibody identity, not RBD mutation — that separation is what inflates the pooled ρ. |
| 7 | `fig7_per_antibody_scatter_grid` | The same data by antibody. Within each one the model tracks the measured ranking (ρ = 0.50–0.66), but predictions stay inside a narrow band. |
| 8 | `fig8_training_trajectory` | Pooled validation ρ across the training arc (runs resumed in sequence; a run invalidated by a frozen-adapter resume bug is excluded). The open circle is the chosen checkpoint, at the start of the plateau. |
| 9 | `fig9_paired_classifier_metrics` | Classification on the same held-out split, per-antibody median-cutoff labels. **The positive class is not the same phenotype in every bar:** for AZD1061, LY-CoV1404 and REGN10987 it is *binding*; for 002-S21F2, AZD8895 and S309 the phenotype is inverted and it is *escape* (each bar is labeled with its own). Metrics are computed per antibody, so each is valid on its own scale; the pooled row is excluded for that reason. S309's low F1 beside the highest AUROC is a threshold effect at a 24% escape-positive rate, not weak discrimination. |
| 10 | `fig10_data_scale` | Labeled examples: per-antigen specificity sets (train + test) vs. the paired antibody × RBD-variant table (Library M, enrichment at 100 nM). Log scale. |

## Suggested layout

- **Data:** fig 10
- **Specificity:** fig 1 large, fig 2 beside it, fig 3 small; fig 4 if space allows
- **Paired affinity:** fig 5 large, then fig 7 (or fig 6 if space is tight — same story,
  less room), fig 9 for the classification variant
- **Rigor sidebar:** fig 8

## Design notes

- Antigen colour is fixed by entity everywhere: Influenza blue, HIV orange, SARS-CoV-2 aqua.
  Single-series charts use one blue.
- Value axes are dropped wherever every bar is labeled — the numbers are on the marks.
- Palette validated for colour-vision deficiency and surface contrast; every value is also
  printed as text, so nothing depends on colour and the figures survive greyscale printing.
- Type is sized for arm's-length viewing. If you scale a figure below ~60% of its rendered
  size, raise the font sizes in `src/make_poster_charts.py` rather than shrinking the PDF.
