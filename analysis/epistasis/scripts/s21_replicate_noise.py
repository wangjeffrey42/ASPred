"""Measurement-noise floor from the MABAa/MABAb replicate pair.

Replaces an earlier "682 repeated genotypes" estimate, which does not reconstruct
from the data under any scope (see README). The paired dataset is built from
replicate *a* only; replicate *b* is the real technical replicate and is the
right basis for a noise estimate.

MABAb lives only in the parent repo's `RBD Data/`, zipped, so this script reaches
outside the submodule and is skipped if that directory is absent.

Per-measurement sd is sd(b - a) / sqrt(2). "Interior" excludes rows sitting on the
group's saturation bound in either replicate, since a bound pile-up has no variance
and would deflate the estimate.
"""
import pandas as pd, numpy as np, zipfile, io
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
RS = _os.path.join(_os.path.dirname(_HERE), "results")
P = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
PARENT = _os.path.dirname(P)
RBD = _os.path.join(PARENT, "RBD Data")

ABS = ["002-S21F2", "AZD1061", "AZD8895", "LY-CoV1404", "REGN10987", "S309"]
FLOOR = {"002-S21F2", "AZD8895", "S309"}   # see README: floor vs ceiling sets

if not _os.path.isdir(RBD):
    raise SystemExit(f"parent 'RBD Data/' not found at {RBD} - skipping")


def read_b(ab):
    z = _os.path.join(RBD, f"MABAb_{ab}_phenotype_data.csv.zip")
    with zipfile.ZipFile(z) as zf:
        name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(name) as fh:
            return pd.read_csv(io.TextIOWrapper(fh, "utf-8"),
                               usecols=["ph_mean", "aa_geno"]).dropna()


print(f"{'group':12s} {'shared':>8s} {'interior':>9s} {'noise_all':>10s} "
      f"{'noise_int':>10s} {'pearson':>8s}")
rows = []
for ab in ABS:
    a = pd.read_csv(f"{P}/data/rbd_dms/MABAa_{ab}_phenotype_data.csv",
                    usecols=["ph_mean", "aa_geno"]).dropna()
    b = read_b(ab)
    am = a.groupby("aa_geno").ph_mean.mean()
    bm = b.groupby("aa_geno").ph_mean.mean()
    sh = am.index.intersection(bm.index)
    x, y = am[sh].to_numpy(), bm[sh].to_numpy()
    d = y - x
    n_all = d.std(ddof=1) / np.sqrt(2)
    m = (x > 0) & (y > 0) if ab in FLOOR else (x < 1) & (y < 1)
    n_int = d[m].std(ddof=1) / np.sqrt(2)
    r = float(np.corrcoef(x, y)[0, 1])
    rows.append(dict(group=ab, n_shared=len(sh), n_interior=int(m.sum()),
                     noise_sd_all=n_all, noise_sd_interior=n_int,
                     replicate_pearson=r))
    print(f"{ab:12s} {len(sh):8d} {int(m.sum()):9d} {n_all:10.4f} "
          f"{n_int:10.4f} {r:8.4f}")

O = pd.DataFrame(rows)
print(f"\nmedian interior per-measurement noise sd: "
      f"{O.noise_sd_interior.median():.4f}")
print("\nCompare against RMS fitted interaction coefficients in "
      "main_vs_interaction_slope.csv (0.054-0.163): the per-measurement noise is\n"
      "the same order as the effects, but each coefficient is estimated from ~1e5\n"
      "rows, so the reported Tobit SEs (~0.006-0.008) stay well below the effects\n"
      "and the FDR-significant interactions remain identified.")
O.to_csv(f"{RS}/replicate_noise.csv", index=False)
print(f"\nwrote {RS}/replicate_noise.csv")
