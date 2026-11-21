"""Single source of truth for which antibodies use which phenotype convention.

Library M reports every antibody as `Enrichment at 100nM` in
`RBD Data/File_Details.csv`, with no field distinguishing them. But the six
antibodies split into two sets whose phenotype runs in OPPOSITE directions, and
the upstream `RBD Data/README.txt` gives the mechanism: an enrichment score
depends on which subpopulation was sorted.

    evading-population sort  -> score = s      (escape; 0 = none, may exceed 1)
    binding-population sort  -> score = 1 - s  (1 = full binding, may go negative)

So:

    FLOOR   antibodies: affinity in [0, ~4],   mass at 0.  HIGHER = MORE ESCAPE
    CEILING antibodies: affinity in (~-3, 1],  mass at 1.  HIGHER = MORE BINDING

This is inferred from the bound structure, not recorded upstream. Treat it as
the best available reading, not as confirmed fact.

Why this module exists
----------------------
The convention was previously re-declared by hand in several scripts (as
`CEILING_GROUPS`) and then used as if it were only a tie-handling detail for a
median cutoff. It is not: it inverts the SIGN of the phenotype. Figures that
labelled every antibody's rising curve as "binding", and a classification label
called "Binds" for all six, were wrong for the floor trio. Import from here
instead of restating the sets, and call `verify()` so a data change fails loudly
rather than silently flipping a label.

See `analysis/epistasis/README.md` ("The six groups sit on two incompatible
scales") for the full trace of what this affects.
"""
from __future__ import annotations

FLOOR = frozenset({"002-S21F2", "AZD8895", "S309"})
CEILING = frozenset({"AZD1061", "LY-CoV1404", "REGN10987"})
ALL = FLOOR | CEILING

# Where each set's point mass sits.
BOUND = {"floor": 0.0, "ceiling": 1.0}


def convention(antibody: str) -> str:
    """'floor' or 'ceiling'. Raises on an unknown antibody rather than guessing."""
    if antibody in FLOOR:
        return "floor"
    if antibody in CEILING:
        return "ceiling"
    raise KeyError(
        f"unknown phenotype convention for {antibody!r}. Do not assume: "
        f"determine the sorted subpopulation, then add it to FLOOR or CEILING "
        f"in src/phenotype_convention.py."
    )


def is_escape_scale(antibody: str) -> bool:
    """True when a HIGHER affinity value means MORE ESCAPE (less binding)."""
    return convention(antibody) == "floor"


def measured_quantity(antibody: str) -> str:
    """Axis/legend wording for the raw phenotype. Use instead of 'binding'."""
    return "escape" if is_escape_scale(antibody) else "binding"


def rising_means(antibody: str) -> str:
    """What an increasing phenotype value means, for axis labels and captions."""
    return "more escape" if is_escape_scale(antibody) else "more binding"


def positive_label(antibody: str) -> str:
    """What `label == 1` means under the median-cutoff rule in
    make_group_cutoff_labels.py / make_variant_holdout_labels.py.

    Floor groups label `affinity > median` as 1, which is MORE ESCAPE.
    Ceiling groups label `affinity >= median` as 1, which is MORE BINDING.
    The same integer therefore means opposite things across the two sets.
    """
    return "Escaped" if is_escape_scale(antibody) else "Binds"


def interior_mask(antibody: str, values):
    """Rows NOT sitting on this antibody's saturation bound.

    `values` is anything numpy-comparable (Series or ndarray).
    """
    if is_escape_scale(antibody):
        return values > BOUND["floor"]
    return values < BOUND["ceiling"]


def verify(df, antibody_col: str = "antibody_name", value_col: str = "affinity",
           min_mass: float = 0.30, tol: float = 1e-9) -> None:
    """Check the declared convention against the data. Raises AssertionError.

    For each antibody present, confirms a real point mass sits on the bound this
    module claims and that essentially nothing falls beyond it. Deliberately
    strict: if the upstream data is reprocessed or rescaled, this should fail
    rather than let a figure quietly mislabel half its panels.
    """
    for ab, g in df.groupby(antibody_col):
        if ab not in ALL:
            raise KeyError(f"{ab!r} has no declared convention; see convention()")
        v = g[value_col].dropna()
        if not len(v):
            continue
        kind = convention(ab)
        if kind == "floor":
            mass = float((v <= BOUND["floor"] + tol).mean())
            beyond = float((v < BOUND["floor"] - tol).mean())
            where = "at 0"
        else:
            mass = float((v >= BOUND["ceiling"] - tol).mean())
            beyond = float((v > BOUND["ceiling"] + tol).mean())
            where = "at 1"
        assert mass >= min_mass, (
            f"{ab} declared {kind} but only {mass:.1%} of values sit {where} "
            f"(expected >= {min_mass:.0%}). The convention may have changed - "
            f"do not relabel figures until this is resolved."
        )
        assert beyond <= 0.001, (
            f"{ab} declared {kind} but {beyond:.2%} of values fall beyond the "
            f"{kind} bound ({where}), so that bound is not a saturation point."
        )


def summary() -> str:
    lines = ["phenotype conventions (Library M, Enrichment at 100nM):"]
    for kind, members in (("floor", sorted(FLOOR)), ("ceiling", sorted(CEILING))):
        arrow = "higher = MORE ESCAPE" if kind == "floor" else "higher = MORE BINDING"
        lines.append(f"  {kind:8s} mass at {BOUND[kind]:.0f}  {arrow}")
        for m in members:
            lines.append(f"             {m}")
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    import pandas as pd

    print(summary())
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "data", "paired", "antibody_rbd_affinity_pairs.csv")
    if os.path.exists(src):
        df = pd.read_csv(src, usecols=["antibody_name", "affinity"])
        verify(df)
        print(f"\nverified against {os.path.relpath(src, root)} "
              f"({len(df):,} rows): conventions hold")
