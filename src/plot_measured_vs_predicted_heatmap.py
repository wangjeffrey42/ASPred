#!/usr/bin/env python3
"""Measured vs. predicted binding heatmaps, RBD-genotype-holdout test set.

Predictions from models/paired_affinity/run5_rbd_split_continued/, written to
results/rbd_split_test_predictions.csv. Run from the repo root.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phenotype_convention as pc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "poster_charts")
os.makedirs(OUT, exist_ok=True)

N_VARIANTS = 250
SEED = 42
CMAP = "viridis"                     # perceptually uniform
INK, SURFACE = "#1a1a19", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 15,
    "axes.edgecolor": "#4a4a48", "axes.linewidth": 0.9,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "savefig.bbox": "tight",
})

pt = pd.read_csv(os.path.join(ROOT, "results", "rbd_split_test_predictions.csv"),
                 usecols=["antibody_name", "rbd_sequence", "affinity",
                          "predicted_affinity"])
pc.verify(pt)   # fail loudly if declared conventions no longer match the data

# 29 genotypes carry replicate rows per antibody (same amino-acid sequence measured
# twice). Replicates are averaged rather than dropped, so one row == one genotype.
n_rows, n_geno = len(pt), pt["rbd_sequence"].nunique()
measured = pt.pivot_table(index="rbd_sequence", columns="antibody_name",
                          values="affinity", aggfunc="mean")
predicted = pt.pivot_table(index="rbd_sequence", columns="antibody_name",
                           values="predicted_affinity", aggfunc="mean")
print(f"{n_rows:,} test rows -> {n_geno:,} distinct genotypes (replicates averaged)")

# Keep only variants scored against every antibody, then sample.
complete = measured.dropna().index.intersection(predicted.dropna().index)
rng = np.random.default_rng(SEED)
sample = rng.choice(complete.to_numpy(), size=N_VARIANTS, replace=False)

measured = measured.loc[sample]
predicted = predicted.loc[sample]

# One row per convention. Each block is ordered, scaled and labelled using only
# its own antibodies: averaging or colour-scaling across the two would mix a
# [0, ~4] escape score with a (~-3, 1] binding score and call the result one
# quantity. See src/phenotype_convention.py.
print(f"variants sampled:      {len(measured):,} of {len(complete):,} complete "
      f"test variants ({len(pt):,} rows total)")

blocks = []
for kind in ("ceiling", "floor"):
    cols = [c for c in measured.columns if pc.convention(c) == kind]
    if not cols:
        continue
    m = measured[cols]
    p = predicted[cols]
    # Row and column order from this convention's columns only.
    row_order = m.mean(axis=1).sort_values().index
    col_order = m.mean(axis=0).sort_values().index
    m = m.loc[row_order, col_order]
    p = p.loc[row_order, col_order]
    # Colour scale from this convention's cells only, 1st-99th percentile so a
    # few outliers do not compress everything into one colour.
    pooled = np.concatenate([m.to_numpy().ravel(), p.to_numpy().ravel()])
    vmin, vmax = (float(v) for v in np.percentile(pooled, [1, 99]))
    clipped = float(np.mean((pooled < vmin) | (pooled > vmax)))
    quantity = pc.measured_quantity(col_order[0]).capitalize()
    blocks.append(dict(kind=kind, m=m, p=p, vmin=vmin, vmax=vmax,
                       quantity=quantity, cols=list(col_order)))
    print(f"\n{kind} block ({quantity.lower()} scale, higher = "
          f"{pc.rising_means(col_order[0])}):")
    print(f"  antibodies:          {list(col_order)}")
    print(f"  colour scale:        vmin={vmin:.4f}  vmax={vmax:.4f}  "
          f"({clipped:.2%} of cells saturate)")
    print(f"  raw range:           [{pooled.min():.4f}, {pooled.max():.4f}]")
    print(f"  row order identical: {m.index.equals(p.index)}")

fig, axes = plt.subplots(len(blocks), 2, figsize=(9.0, 5.4 * len(blocks)),
                         squeeze=False)
for r, blk in enumerate(blocks):
    for c, (mat, label) in enumerate(((blk["m"], "Measured"),
                                      (blk["p"], "Predicted"))):
        ax = axes[r][c]
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap=CMAP,
                       vmin=blk["vmin"], vmax=blk["vmax"],
                       interpolation="nearest")
        if r == 0:
            ax.set_title(label, fontsize=19, fontweight="semibold", color=INK,
                         pad=12)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=13.5)
        ax.set_yticks([])
        ax.tick_params(length=0)
    axes[r][0].set_ylabel(f"RBD variants\n({blk['kind']} set)", labelpad=12,
                          fontsize=15)
    cbar = fig.colorbar(im, ax=axes[r], fraction=0.055, pad=0.04, extend="both")
    cbar.set_label(f"{blk['quantity']}  (higher = "
                   f"{pc.rising_means(blk['cols'][0])})",
                   fontsize=14, color=INK, labelpad=10)
    cbar.outline.set_linewidth(0.9)
    cbar.outline.set_edgecolor("#4a4a48")
    cbar.ax.tick_params(labelsize=13, color=INK)

for ext in ("png", "svg"):
    fig.savefig(os.path.join(OUT, f"measured_vs_predicted_heatmap.{ext}"), dpi=300)
plt.close(fig)
print("\nwrote", os.path.join(OUT, "measured_vs_predicted_heatmap.{png,svg}"))
