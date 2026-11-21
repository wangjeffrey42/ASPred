#!/usr/bin/env python3
"""Share of variants above each antibody's median cutoff, by phenotype convention.

Uses the make_variant_holdout_labels.py cutoff rule:

    label = 1  if affinity >  per-antibody median cutoff
               or affinity >= cutoff for ceiling groups, where >= 50% of rows
               sit exactly at the ceiling and "> cutoff" is degenerate.

**`label == 1` does not mean the same thing in every group.** The six antibodies
sit on two opposed phenotype conventions (see src/phenotype_convention.py): for
the ceiling trio a 1 means MORE BINDING, for the floor trio it means MORE ESCAPE.
This figure previously called the segment "Binds" for all six, which inverted the
meaning for three of them. Each bar is now labelled with its own class.

Run from the repo root.
"""
import os
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

# Single source of truth; do not restate these sets here.
CEILING_GROUPS = set(pc.CEILING)

BINDS = "#5b7fa6"          # muted slate blue — ceiling trio, 1 = more binding
ESCAPES = "#c2794f"        # burnt orange — floor trio, 1 = more escape
INK, SURFACE = "#1a1a19", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 16,
    "axes.edgecolor": "#b9b8b3", "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 15, "ytick.labelsize": 16, "axes.labelsize": 17,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False, "legend.fontsize": 16,
    "savefig.bbox": "tight",
})

df = pd.read_csv(os.path.join(ROOT, "data", "paired", "antibody_rbd_affinity_pairs.csv"),
                 usecols=["antibody_name", "affinity"])
pc.verify(df)   # fail loudly if the declared conventions no longer match the data

cutoffs = df.groupby("antibody_name")["affinity"].median()


def label_group(g):
    cutoff = cutoffs[g.name]
    if g.name in CEILING_GROUPS:
        return (g["affinity"] >= cutoff).astype(int)
    return (g["affinity"] > cutoff).astype(int)


df["label"] = df.groupby("antibody_name", group_keys=False).apply(
    label_group, include_groups=False)

stats = (df.groupby("antibody_name")["label"]
           .agg(n="count", n_pos="sum")
           .assign(n_neg=lambda x: x["n"] - x["n_pos"],
                   frac_pos=lambda x: x["n_pos"] / x["n"],
                   cutoff=cutoffs,
                   rule=lambda x: [">=" if i in CEILING_GROUPS else ">" for i in x.index],
                   convention=lambda x: [pc.convention(i) for i in x.index],
                   pos_class=lambda x: [pc.positive_label(i) for i in x.index]))

# Group by convention, then by share within it, so the two scales never interleave.
stats = stats.sort_values(["convention", "frac_pos"])

print(f"{'antibody':<12}{'conv':>9}{'n':>10}{'label=1':>10}{'label=0':>10}"
      f"{'%label=1':>10}{'cutoff':>9}  rule / meaning of label=1")
for name, r in stats.iloc[::-1].iterrows():
    print(f"{name:<12}{r.convention:>9}{r.n:>10,}{r.n_pos:>10,}{r.n_neg:>10,}"
          f"{r.frac_pos:>9.1%}{r.cutoff:>9.4f}  affinity {r.rule} cutoff "
          f"-> {r.pos_class}")

# No pooled total: label=1 means "Binds" for three antibodies and "Escaped" for
# the other three, so summing them is not a single quantity.
for kind in ("ceiling", "floor"):
    s = stats[stats.convention == kind]
    cls = pc.positive_label(s.index[0])
    print(f"\n{kind} trio ({', '.join(sorted(s.index))}): "
          f"{s.n_pos.sum() / s.n.sum():.1%} {cls} "
          f"({s.n_pos.sum():,} of {s.n.sum():,})")

fig, ax = plt.subplots(figsize=(11.5, 5.6))
y = range(len(stats))
colors = [BINDS if c == "ceiling" else ESCAPES for c in stats["convention"]]
b = ax.barh(y, stats["frac_pos"], height=0.62, color=colors)

for bar, (name, r) in zip(b, stats.iterrows()):
    yc = bar.get_y() + bar.get_height() / 2
    ax.text(r.frac_pos + 0.012, yc + 0.055, f"{r.frac_pos:.0%} {r.pos_class}",
            ha="left", va="bottom", fontsize=16, color=INK, fontweight="semibold")
    ax.text(r.frac_pos + 0.012, yc - 0.07, f"{int(r.n_pos):,} variants",
            ha="left", va="top", fontsize=13.5, color="#6b6a66")

ax.set_yticks(list(y)); ax.set_yticklabels(stats.index)
ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of variants above the antibody's median cutoff", labelpad=12)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

# Bracket the two conventions so the split is visible, not just colour-coded.
for kind, label in (("ceiling", "higher = more binding"),
                    ("floor", "higher = more escape")):
    rows = [i for i, c in enumerate(stats["convention"]) if c == kind]
    color = BINDS if kind == "ceiling" else ESCAPES
    ax.text(0.012, (min(rows) + max(rows)) / 2, f"{kind}\n{label}",
            ha="left", va="center", fontsize=12.5, color=color,
            fontweight="semibold", linespacing=1.4)

ax.text(0, 1.10, "Variants above each antibody's median cutoff",
        transform=ax.transAxes, fontsize=19, color=INK, fontweight="semibold",
        va="bottom")
ax.text(0, 1.035,
        "The cutoff rule's positive class is not the same phenotype in both sets: "
        "more binding above, more escape below.",
        transform=ax.transAxes, fontsize=12.5, color="#6b6a66", va="bottom")

for ext in ("png", "svg"):
    fig.savefig(os.path.join(OUT, f"binds_escaped_split.{ext}"), dpi=300)
plt.close(fig)
print("\nwrote", os.path.join(OUT, "binds_escaped_split.{png,svg}"))
