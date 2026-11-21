#!/usr/bin/env python3
"""Poster figures for ASPred / paired affinity model. Run from the repo root."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phenotype_convention as pc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALT = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "results", "poster_charts")
os.makedirs(OUT, exist_ok=True)

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE = "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.edgecolor": "#cbcac5", "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
    "axes.labelsize": 12, "legend.frameon": False, "legend.fontsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.dpi": 110, "savefig.bbox": "tight",
})


def finish(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", name)


def title(ax, text):
    ax.text(0, 1.04, text, transform=ax.transAxes, color=INK,
            fontsize=14, fontweight="semibold", va="bottom")


def no_value_axis(ax, axis="x"):
    """Drop the axis that the printed values already carry."""
    if axis == "x":
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)


def mpath(*parts):
    p = os.path.join(ROOT, *parts)
    return p if os.path.exists(p) else os.path.join(ALT, *parts)


ANTIGENS = [("FLU", "Influenza HA"), ("HIV", "HIV gp120"), ("SARS", "SARS-CoV-2 RBD")]
AG_COLOR = {"FLU": S1, "HIV": S2, "SARS": S3}

# =====================================================================
# 1 - ROC curves
# =====================================================================
fig, ax = plt.subplots(figsize=(5.0, 4.6))
ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color="#d4d3ce", lw=1.0, zorder=1)
for key, label in ANTIGENS:
    d = pd.read_csv(mpath("models", f"650M_heavy_{key}_loss", "test_evaluation",
                          "test_predictions.csv"))
    fpr, tpr, _ = roc_curve(d["label"], d["prob"])
    ax.plot(fpr, tpr, color=AG_COLOR[key], lw=2.4, zorder=3,
            label=f"{label}   {roc_auc_score(d['label'], d['prob']):.2f}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.004)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(loc="lower right", handlelength=1.4, labelspacing=0.7)
title(ax, "ROC by antigen")
finish(fig, "fig1_aspred_roc_curves")

# =====================================================================
# 2 - test metrics
# =====================================================================
rows = []
for key, label in ANTIGENS:
    te = pd.read_csv(mpath("models", f"650M_heavy_{key}_loss", "test_evaluation",
                           "test_metrics.csv"), index_col=0)["0"]
    tr = pd.read_csv(mpath("models", f"650M_heavy_{key}_loss", "training_output",
                           "train_metrics.csv"), index_col=0)["0"]
    rows.append(dict(key=key, label=label, test_f1=te["f1"], test_roc_auc=te["roc_auc"],
                     train_f1=tr["f1"]))
p1 = pd.DataFrame(rows)

metrics = [("test_f1", "F1"), ("test_roc_auc", "ROC-AUC")]
fig, ax = plt.subplots(figsize=(5.4, 3.4))
x = np.arange(len(metrics)); w = 0.26
for i, (key, label) in enumerate(ANTIGENS):
    vals = [p1.loc[p1.key == key, m].iloc[0] for m, _ in metrics]
    bars = ax.bar(x + (i - 1) * (w + 0.02), vals, width=w, color=AG_COLOR[key], label=label)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([n for _, n in metrics])
ax.set_ylim(0, 1.1)
no_value_axis(ax, "y")
ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.28), handlelength=1.0,
          columnspacing=1.4)
title(ax, "Held-out test scores")
finish(fig, "fig2_aspred_test_metrics")

# =====================================================================
# 3 - train vs test F1
# =====================================================================
fig, ax = plt.subplots(figsize=(5.6, 2.6))
ypos = np.arange(len(ANTIGENS))[::-1]
for y, (key, label) in zip(ypos, ANTIGENS):
    r = p1.loc[p1.key == key].iloc[0]
    ax.plot([r.test_f1, r.train_f1], [y, y], color="#dcdbd6", lw=3,
            solid_capstyle="round", zorder=2)
    ax.scatter([r.train_f1], [y], s=90, color=SURFACE, edgecolor="#b0afaa", lw=1.8, zorder=3)
    ax.scatter([r.test_f1], [y], s=105, color=AG_COLOR[key], zorder=4)
    ax.text(r.test_f1 - 0.01, y, f"{r.test_f1:.2f}", fontsize=11.5, color=INK,
            va="center", ha="right", fontweight="semibold")
    ax.text(r.train_f1 + 0.01, y, f"{r.train_f1:.2f}", fontsize=11.5, color=MUTED,
            va="center", ha="left")
ax.set_yticks(ypos); ax.set_yticklabels([l for _, l in ANTIGENS])
ax.set_xlim(0.76, 1.05); ax.set_ylim(-0.5, len(ANTIGENS) - 0.4)
no_value_axis(ax, "x")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
title(ax, "F1: test → train")
finish(fig, "fig3_train_test_gap")

# =====================================================================
# 4 - cross-antigen calls
# =====================================================================
mt = pd.read_csv(os.path.join(ROOT, "results", "multi_target", "multi_target_predictions.csv"))
counts = mt["n_targets_positive"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(4.6, 2.9))
bars = ax.bar(counts.index.astype(str), counts.values, width=0.6, color=S1)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v}", ha="center", va="bottom",
            fontsize=11.5, color=INK2)
ax.set_ylim(0, counts.max() * 1.2)
no_value_axis(ax, "y")
ax.set_xlabel("Antigens called positive")
title(ax, "Calls per sequence")
finish(fig, "fig4_cross_antigen_calls")

# =====================================================================
# Phase 2 - held-out predictions
# =====================================================================
pt = pd.read_csv(os.path.join(ROOT, "results", "rbd_split_test_predictions.csv"),
                 usecols=["antibody_name", "affinity", "predicted_affinity"])
agg_rho = spearmanr(pt["affinity"], pt["predicted_affinity"]).statistic
per = (pt.groupby("antibody_name")
         .apply(lambda g: pd.Series({
             "spearman": spearmanr(g["affinity"], g["predicted_affinity"]).statistic}),
                include_groups=False)
         .sort_values("spearman"))

# ---- 5: per-antibody Spearman ----
fig, ax = plt.subplots(figsize=(5.4, 3.4))
y = np.arange(len(per))
bars = ax.barh(y, per["spearman"], height=0.62, color=S1)
for b, v in zip(bars, per["spearman"]):
    ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.2f}",
            va="center", fontsize=11.5, color=INK2)
py = len(per) + 0.85
ax.barh([py], [agg_rho], height=0.62, color="#c6d5e7")
ax.text(agg_rho + 0.015, py, f"{agg_rho:.2f}", va="center", fontsize=11.5, color=MUTED)
# The pooled bar does not just mix antibodies, it mixes two opposed phenotype
# scales (src/phenotype_convention.py): rho within ceiling is 0.585 and within
# floor 0.648, against 0.900 pooled. Name the scale on every row.
ax.set_yticks(list(y) + [py])
ax.set_yticklabels([f"{a}  ({pc.measured_quantity(a)})" for a in per.index]
                   + ["pooled (mixes both scales)"])
ax.get_yticklabels()[-1].set_color(MUTED)
ax.set_xlim(0, 1.05); ax.set_ylim(-0.6, py + 0.5)
no_value_axis(ax, "x")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
title(ax, "Spearman ρ per antibody")
finish(fig, "fig5_per_antibody_spearman")

# ---- 6: pooled predicted vs measured ----
lo = min(np.percentile(pt["affinity"], 0.5), np.percentile(pt["predicted_affinity"], 0.5))
hi = max(np.percentile(pt["affinity"], 99.5), np.percentile(pt["predicted_affinity"], 99.5))
lim = (lo, hi)
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.hexbin(pt["affinity"], pt["predicted_affinity"], gridsize=55, cmap=BLUES,
          norm=LogNorm(vmin=1), extent=(lim[0], lim[1], lim[0], lim[1]),
          linewidths=0, mincnt=1, zorder=2)
ax.plot(lim, lim, ls=(0, (4, 4)), color="#8d8c87", lw=1.0, zorder=3)
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xlabel("Measured"); ax.set_ylabel("Predicted")
# The two bands are not just "antibody identity": they are the two phenotype
# conventions, separated perfectly (100% of ceiling rows predict >0.5, 0% of
# floor rows). Say so on the figure.
ax.text(0.97, 0.955, "ceiling trio — higher = more binding", transform=ax.transAxes,
        ha="right", va="top", fontsize=9.5, color=INK2)
ax.text(0.97, 0.055, "floor trio — higher = more escape", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9.5, color=INK2)
title(ax, "Predicted vs. measured (two scales)")
finish(fig, "fig6_predicted_vs_measured_pooled")

# ---- 7: small multiples ----
order = per.sort_values("spearman", ascending=False).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.9), sharex=True, sharey=True)
for axx, name in zip(axes.ravel(), order):
    g = pt[pt.antibody_name == name]
    axx.hexbin(g["affinity"], g["predicted_affinity"], gridsize=38, cmap=BLUES,
               norm=LogNorm(vmin=1), extent=(lim[0], lim[1], lim[0], lim[1]),
               linewidths=0, mincnt=1, zorder=2)
    axx.plot(lim, lim, ls=(0, (4, 4)), color="#8d8c87", lw=1.0, zorder=3)
    axx.set_xlim(*lim); axx.set_ylim(*lim)
    axx.set_xticks([0, 1]); axx.set_yticks([0, 1])
    axx.text(0, 1.02, name, transform=axx.transAxes, fontsize=12, color=INK,
             fontweight="semibold", va="bottom")
    axx.text(1.0, 1.03, f"ρ = {per.loc[name, 'spearman']:.2f}", transform=axx.transAxes,
             fontsize=11, color=MUTED, ha="right", va="bottom")
for axx in axes[1]:
    axx.set_xlabel("Measured")
for axx in axes[:, 0]:
    axx.set_ylabel("Predicted")
fig.tight_layout(h_pad=2.4)
finish(fig, "fig7_per_antibody_scatter_grid")

# =====================================================================
# 8 - training trajectory
# =====================================================================
m = pd.read_csv(os.path.join(ROOT, "results", "paired_affinity_metrics.csv"))
tr = m[(m["scope"] == "aggregate") & (m["status"] == "valid") & (m["split"] == "val")
       & m["run"].isin(["run2", "run4", "run5", "run6"])].copy()
tr["step"] = tr["step"].astype(int)
offset, xs, ys, marks = 0, [], [], {}
for r in ["run2", "run4", "run5", "run6"]:
    g = tr[tr.run == r].sort_values("step")
    xs += (g["step"] + offset).tolist(); ys += g["spearman"].tolist()
    offset += int(g["step"].max()); marks[r] = (offset, g["spearman"].iloc[-1])

fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.plot(xs, ys, color=S1, lw=2.4, zorder=3)
bx, best = marks["run5"]
ax.scatter([bx], [best], s=80, color=SURFACE, edgecolor=S1, lw=2.2, zorder=4)
ax.set_ylim(0.79, 0.915); ax.set_xlim(0, xs[-1] * 1.02)
ax.set_yticks([0.80, 0.90]); ax.set_xticks([0, 5000, 10000])
ax.set_xlabel("Training steps"); ax.set_ylabel("Validation ρ")
title(ax, "Training curve")
finish(fig, "fig8_training_trajectory")

# =====================================================================
# 9 - paired classifier
# =====================================================================
cl = pd.read_csv(os.path.join(ROOT, "results", "paired_classifier_test_metrics.csv"))
cg = cl[cl.group != "pooled"].sort_values("auroc")
fig, ax = plt.subplots(figsize=(5.6, 3.5))
y = np.arange(len(cg)); h = 0.33
b1 = ax.barh(y + h / 2 + 0.015, cg["auroc"], height=h, color=S1, label="AUROC")
b2 = ax.barh(y - h / 2 - 0.015, cg["f1"], height=h, color=S2, label="F1")
for bars, col in ((b1, "auroc"), (b2, "f1")):
    for b, v in zip(bars, cg[col]):
        ax.text(v + 0.013, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=11, color=INK2)
# The positive class is NOT one phenotype across these six bars: the median
# cutoff labels "more binding" for the ceiling trio and "more escape" for the
# floor trio (src/phenotype_convention.py). Name it per bar, or the panel reads
# as six measurements of one quantity.
ax.set_yticks(y)
ax.set_yticklabels([f"{g}  ({pc.measured_quantity(g)})" for g in cg.group])
ax.set_xlim(0, 1.05); ax.set_ylim(-0.6, len(cg) - 0.35)
no_value_axis(ax, "x")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.16), handlelength=1.0,
          columnspacing=1.6)
title(ax, "Above-median call: binding (3) / escape (3)")
finish(fig, "fig9_paired_classifier_metrics")

# =====================================================================
# 10 - data scale
# =====================================================================
ds = []
for key, label in ANTIGENS:
    n = sum(len(pd.read_csv(os.path.join(ROOT, "datasets", f"{key}_heavy", f"{s}.csv")))
            for s in ("train", "test"))
    ds.append((label, n))
pairs = pd.read_csv(os.path.join(ROOT, "data", "paired", "antibody_rbd_affinity_pairs.csv"),
                    usecols=["rbd_sequence"])
ds.append(("Antibody × RBD", len(pairs)))
vals = [b for _, b in ds]
fig, ax = plt.subplots(figsize=(6.6, 3.0))
bars = ax.bar([a for a, _ in ds], vals, width=0.58, color=[S1, S1, S1, S2])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.4, f"{v:,}", ha="center",
            fontsize=11.5, color=INK2)
ax.set_yscale("log"); ax.set_ylim(800, max(vals) * 5)
ax.set_yticks([1e3, 1e6])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
title(ax, "Labeled examples (log scale)")
finish(fig, "fig10_data_scale")

print("\nAll figures in:", OUT)
