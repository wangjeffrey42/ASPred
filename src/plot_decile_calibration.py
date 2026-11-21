#!/usr/bin/env python3
"""Within-antibody ranking: mean measured phenotype by predicted decile.

Predictions from models/paired_affinity/run5_rbd_split_continued/, written to
results/rbd_split_test_predictions.csv. Run from the repo root.

**The six antibodies do not share a phenotype direction.** They sit on two
opposed conventions (see src/phenotype_convention.py): for the ceiling trio a
higher value means MORE BINDING, for the floor trio it means MORE ESCAPE. Every
number here is computed inside `groupby("antibody_name")`, so the deciles,
spreads and ranking accuracies were always valid -- but this figure labelled all
six axes "binding", which inverted the reading for three of them.

Both figures are therefore laid out one convention per row/panel, and the axis
wording comes from `phenotype_convention.measured_quantity()` rather than being
assumed. The overlay figure in particular used to min-max normalise all six onto
a single panel "so the shapes are comparable", which drew two opposed phenotypes
as one trend; it is now split by convention, since a rising normalised curve
means the opposite thing in each half.

What survives a sign flip, and what does not: flipping one antibody's phenotype
(`y -> c - y` on measured and predicted together) reverses the decile order AND
negates every mean, and the two cancel. `spread` and the monotonicity dip count
come out bit-identical, and `ranking_accuracy` is unchanged because `d_true` and
`d_pred` both negate while the tie set is untouched. Only the decile MEANS are
convention-dependent. Three of the four printed quantities are therefore
convention-invariant -- which is NOT the same as being comparable across
antibodies.

Ranking accuracy especially is not a cross-antibody quality measure, and an
earlier version of this file said it was. It is computed only on pairs whose
measured values differ, and that share tracks the censoring fraction: 42.6% of
pairs for S309 (75.8% censored) against 76.1% for AZD8895 (48.7% censored), with
Pearson(rank acc, censored fraction) = +0.68 over the six. Restricted to
interior-vs-interior pairs the ordering reverses -- S309 goes from best (84.2%)
to worst (55.3%) and REGN10987 from fifth to first -- so the headline number
largely reports how much of each antibody sits on its bound, not how well the
model ranks. For model quality use the per-group Spearman in
`results/paired_affinity_metrics.csv`.
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

# Convention-coded, matching plot_binds_escaped_split.py: cool = binding scale,
# warm = escape scale.
BINDS = "#5b7fa6"          # muted slate blue — ceiling trio, higher = more binding
ESCAPES = "#c2794f"        # burnt orange — floor trio, higher = more escape
CONV_COLOR = {"ceiling": BINDS, "floor": ESCAPES}
INK, SURFACE = "#1a1a19", "#ffffff"
N_DECILES = 10
N_PAIRS = 4_000_000          # for the pairwise ranking accuracy annotation
SEED = 42

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 14,
    "axes.edgecolor": "#4a4a48", "axes.linewidth": 0.9,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15,
    "axes.grid": False, "savefig.bbox": "tight",
})

pt = pd.read_csv(os.path.join(ROOT, "results", "rbd_split_test_predictions.csv"),
                 usecols=["antibody_name", "affinity", "predicted_affinity"])
pc.verify(pt)   # fail loudly if declared conventions no longer match the data

rng = np.random.default_rng(SEED)


def ranking_accuracy(true, pred, n_pairs):
    n = true.size
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d_true = np.sign(true[i] - true[j])
    d_pred = np.sign(pred[i] - pred[j])
    rankable = d_true != 0
    return float((d_true[rankable] == d_pred[rankable]).mean())


# `rng` is shared across antibodies, so this loop order fixes the pair draws.
# Keep it as a plain groupby (alphabetical); display order is applied below.
# Consequence worth knowing before editing: each antibody's accuracy depends on
# its POSITION in this loop, so adding a seventh antibody, filtering a group, or
# passing sort=False silently changes every downstream accuracy. A per-antibody
# generator -- np.random.default_rng([SEED, name]) -- would remove the coupling,
# at the cost of changing all six published numbers. Monte-Carlo noise is
# negligible either way (SE ~= 0.02-0.03 pp at N_PAIRS).
stats = {}
for name, g in pt.groupby("antibody_name"):
    order = np.argsort(g["predicted_affinity"].to_numpy(), kind="stable")
    true_sorted = g["affinity"].to_numpy()[order]

    groups = np.array_split(true_sorted, N_DECILES)
    means = np.array([b.mean() for b in groups])
    sems = np.array([b.std(ddof=1) / np.sqrt(b.size) for b in groups])

    stats[name] = dict(
        means=means, sems=sems, sizes=[b.size for b in groups],
        accuracy=ranking_accuracy(g["affinity"].to_numpy(),
                                  g["predicted_affinity"].to_numpy(), N_PAIRS),
        spread=float(means[-1] - means[0]),
        n_breaks=int(np.sum(np.diff(means) < 0)),
        convention=pc.convention(name),
    )

# One row per convention present, ordered by accuracy within the row.
CONVENTIONS = [k for k in ("ceiling", "floor")
               if any(s["convention"] == k for s in stats.values())]
rows = {k: sorted((n for n, s in stats.items() if s["convention"] == k),
                  key=lambda n: -stats[n]["accuracy"])
        for k in CONVENTIONS}

print("Mean measured phenotype by predicted decile (1 = lowest predicted)")
print("Convention-invariant: spread, monotonic, rank acc. Convention-dependent: "
      "the decile means.")
print("None of them is comparable ACROSS antibodies -- rank acc tracks the "
      "censoring fraction (see docstring).\n")
header = "antibody    " + "".join(f"{d:>7}" for d in range(1, N_DECILES + 1))
print(header + f"{'spread':>9}{'monotonic':>11}{'rank acc':>10}")
for kind in CONVENTIONS:
    sample = rows[kind][0]
    print(f"\n  {kind} — higher = {pc.rising_means(sample)}")
    for name in rows[kind]:
        s = stats[name]
        row = "".join(f"{m:>7.3f}" for m in s["means"])
        mono = "yes" if s["n_breaks"] == 0 else f"no ({s['n_breaks']} dips)"
        print(f"{name:<12}{row}{s['spread']:>9.3f}{mono:>11}{s['accuracy']:>10.1%}")
print(f"\ndecile sizes: {stats[rows[CONVENTIONS[0]][0]]['sizes']}")

ncols = max(len(v) for v in rows.values())
fig, axes = plt.subplots(len(CONVENTIONS), ncols, figsize=(9.6, 3.5 * len(CONVENTIONS)),
                         sharex=False, sharey=False, squeeze=False)
x = np.arange(1, N_DECILES + 1)
for r, kind in enumerate(CONVENTIONS):
    color = CONV_COLOR[kind]
    for c in range(ncols):
        ax = axes[r][c]
        if c >= len(rows[kind]):
            ax.set_visible(False)
            continue
        name = rows[kind][c]
        s = stats[name]
        ax.errorbar(x, s["means"], yerr=s["sems"], color=color, lw=2.0, marker="o",
                    markersize=6, ecolor=color, elinewidth=1.4, capsize=3, capthick=1.4,
                    zorder=3)
        ax.set_xlim(0.4, 10.6)
        # Each panel scaled to its own range; curves all run low-left to
        # high-right, so the top-left corner is free in every panel.
        span = s["means"].max() - s["means"].min()
        ax.set_ylim(s["means"].min() - 0.12 * span, s["means"].max() + 0.34 * span)
        ax.set_xticks([1, 5, 10])
        ax.set_title(name, fontsize=15, fontweight="semibold", color=color, pad=8)
        ax.text(0.045, 0.955, f"{s['accuracy'] * 100:.1f}% pairs ranked correctly",
                transform=ax.transAxes, fontsize=11, color=INK, ha="left", va="top")
    # Axis wording per row: "binding" and "escape" are not interchangeable here.
    # One x-label per row, on the centre panel, so it reads as a row-wide axis
    # without three overlapping copies.
    quantity = pc.measured_quantity(rows[kind][0])
    axes[r][0].set_ylabel(f"Mean measured {quantity}", labelpad=10)
    axes[r][len(rows[kind]) // 2].set_xlabel(
        f"Predicted {quantity} rank (low to high)", labelpad=8)

fig.tight_layout(h_pad=2.6, w_pad=1.6)
for ext in ("png", "svg"):
    fig.savefig(os.path.join(OUT, f"decile_calibration.{ext}"), dpi=300)
plt.close(fig)
print("\nwrote", os.path.join(OUT, "decile_calibration.{png,svg}"))

# ---------------------------------------------------------------------------
# Companion figure: curves min-max normalised to their own range, SPLIT BY
# CONVENTION, one panel each. Normalising all six onto one panel (the previous
# version) drew a rising curve for "more escape" and a rising curve for "more
# binding" as if they were one trend.
#
# The split fixes the SIGN error only. It does NOT make the normalised shapes
# comparable between antibodies, and this figure should not be read that way:
# the normalised curve is very nearly a rescaling of the per-decile uncensored
# fraction (|Pearson| = 0.96-1.00 within every antibody), so the "shape" is
# mostly the censoring mechanism, not graded model calibration. Inside the floor
# panel the censoring fractions are 48.7 / 56.9 / 75.8%, so S309's late rise
# against AZD8895's early rise is a point-mass artifact. The error bars are SEMs
# of censored means and collapse near the bound, understating uncertainty in
# exactly the deciles that drive the shape.
# ---------------------------------------------------------------------------
# Fixed colour per antibody so hues do not shift with panel order; the three
# within each panel are spaced far enough apart to stay distinguishable where
# curves overlap (REGN10987/AZD1061 share the ceiling panel). These per-antibody
# hues are deliberately NOT the per-convention colours used for the six-panel
# figure's titles, which encode only ceiling-vs-floor.
SERIES = {
    "AZD1061":    "#5b7fa6",   # slate blue
    "LY-CoV1404": "#5e9e7a",   # sage green
    "REGN10987":  "#7b6aa8",   # muted purple
    "AZD8895":    "#c2794f",   # burnt orange
    "002-S21F2":  "#b86a8c",   # dusty rose
    "S309":       "#8f9a52",   # olive
}

print("\nMin-max normalised decile means (0 = that antibody's lowest decile, "
      "1 = its highest)")
print("Normalisation is within an antibody. Shapes are NOT comparable between "
      "antibodies: they track each one's censoring fraction.\n")
print("antibody    " + "".join(f"{d:>7}" for d in range(1, N_DECILES + 1)))

fig, axes = plt.subplots(1, len(CONVENTIONS), figsize=(6.2 * len(CONVENTIONS), 5.0),
                         squeeze=False)
for c, kind in enumerate(CONVENTIONS):
    ax = axes[0][c]
    print(f"\n  {kind} — higher = {pc.rising_means(rows[kind][0])}")
    for name in rows[kind]:
        s = stats[name]
        rng_ = s["means"].max() - s["means"].min()
        norm = (s["means"] - s["means"].min()) / rng_
        print(f"{name:<12}" + "".join(f"{v:>7.3f}" for v in norm))
        color = SERIES[name]
        ax.errorbar(x, norm, yerr=s["sems"] / rng_, color=color, lw=2.2, marker="o",
                    markersize=6, ecolor=color, elinewidth=1.2, capsize=2.5,
                    capthick=1.2, label=name, zorder=3)

    quantity = pc.measured_quantity(rows[kind][0])
    ax.set_xlim(0.4, 10.6); ax.set_ylim(-0.06, 1.12)
    ax.set_xticks(range(1, N_DECILES + 1))
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel(f"Predicted {quantity} rank (low to high)", labelpad=10)
    ax.set_ylabel(f"Measured {quantity} (normalized)", labelpad=10)
    ax.set_title(f"{kind} set — higher = {pc.rising_means(rows[kind][0])}",
                 fontsize=15, fontweight="semibold", color=CONV_COLOR[kind], pad=10)
    ax.legend(loc="lower right", frameon=False, handlelength=1.4,
              fontsize=12.5, labelspacing=0.5)

fig.tight_layout(w_pad=2.4)
for ext in ("png", "svg"):
    fig.savefig(os.path.join(OUT, f"decile_calibration_overlay.{ext}"), dpi=300)
plt.close(fig)
print("\nwrote", os.path.join(OUT, "decile_calibration_overlay.{png,svg}"))
