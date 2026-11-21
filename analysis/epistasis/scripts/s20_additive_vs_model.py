"""Held-out comparison: 47-parameter additive Tobit vs the trained paired-encoder model.

The Tobit fit comes from s2 (train rows only), so neither model has seen the test
split. Both are scored on the same 17,130 rows per antibody.
"""
import pandas as pd, numpy as np, pickle, warnings
import os as _os
from scipy import stats

warnings.filterwarnings("ignore")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
SP = _os.path.join(_os.path.dirname(_HERE), "cache")
RS = _os.path.join(_os.path.dirname(_HERE), "results")
P = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))

F = pickle.load(open(f"{SP}/feat.pkl", "rb"))
ad = pickle.load(open(f"{SP}/additive.pkl", "rb"))
feat, NP, ABS = F["feat"], F["NP"], F["ABS"]
fidx = {k: i for i, k in enumerate(feat)}

te = pd.read_csv(f"{P}/data/paired/rbd_split_test.csv",
                 usecols=["antibody_name", "rbd_sequence", "affinity"])
pr = pd.read_csv(f"{P}/results/rbd_split_test_predictions.csv",
                 usecols=["antibody_name", "rbd_sequence", "affinity",
                          "predicted_affinity"])

g = te.rbd_sequence.unique()
idx = {s: i for i, s in enumerate(g)}
X = np.zeros((len(g), NP))
for r, s in enumerate(g):
    for p, a in feat:
        if s[p] == a:
            X[r, fidx[(p, a)]] = 1.0
X = np.hstack([np.ones((len(g), 1)), X])
print(f"additive design: {X.shape[1]} params (intercept + {NP} indicators)")
print(f"test genotypes: {len(g)}\n")

print(f"{'antibody':12s} {'n':>7s} {'additive rho':>12s} {'model rho':>10s} {'diff':>8s}")
print("-" * 54)
rows = []
for ab in ABS:
    d = te[te.antibody_name == ab]
    beta = np.asarray(ad[ab]["beta"])
    assert np.isfinite(beta).all() and ad[ab]["converged"]
    yh = X[d.rbd_sequence.map(idx).to_numpy()] @ beta
    ra = stats.spearmanr(yh, d.affinity.to_numpy()).statistic
    m = pr[pr.antibody_name == ab]
    rm = stats.spearmanr(m.predicted_affinity.to_numpy(),
                         m.affinity.to_numpy()).statistic
    rows.append(dict(group=ab, n=len(d), additive_spearman=ra,
                     model_spearman=rm, diff=ra - rm))
    print(f"{ab:12s} {len(d):7d} {ra:12.4f} {rm:10.4f} {ra - rm:+8.4f}")

out = pd.DataFrame(rows)
a, m = out.additive_spearman.to_numpy(), out.model_spearman.to_numpy()
print(f"\nmean additive {a.mean():.4f}   mean model {m.mean():.4f}   "
      f"mean diff {(a - m).mean():+.4f}")
print(f"additive >= model in {(a >= m).sum()} of {len(a)} groups")
out.to_csv(f"{RS}/additive_vs_model_test.csv", index=False)
print(f"\nwrote {RS}/additive_vs_model_test.csv")
