"""Is the interaction-specific (excess) compression censoring-driven or architectural?"""
import os as _os, pickle, warnings
warnings.filterwarnings("ignore")
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache"); RES=_os.path.join(_os.path.dirname(_HERE),"results")
import numpy as np, pandas as pd
from scipy import stats as sst
O=pd.read_csv(f"{RES}/main_vs_interaction_slope.csv")
SEL=pickle.load(open(f"{SP}/selection.pkl","rb")); D=pickle.load(open(f"{SP}/scan3.pkl","rb"))
F=pickle.load(open(f"{SP}/feat.pkl","rb")); res=D["res"]; tw=D["toward"]; names=D["names"]

O["main_ratio"]=O.rms_model_main/O.rms_fitted_main
O["int_ratio"]=O.rms_model_int/O.rms_fitted_int
O["excess"]=O.main_ratio/O.int_ratio
O["pct_censored"]=[SEL["sev"][g]["cens"] for g in O.group]
O["n_interior"]=[SEL["sev"][g]["nint"] for g in O.group]
O["max_depletion"]=[SEL["sev"][g]["maxdep"] for g in O.group]
O["median_attenuation"]=[SEL["att"][g] for g in O.group]
O["n_artifact_terms"]=[sum(1 for i,j in res[("tobit",g)]["top"] if tw[g][i] and tw[g][j]) for g in O.group]
O["rms_fitted_int"]=O.rms_fitted_int

print("="*104)
print("EXCESS (INTERACTION-SPECIFIC) COMPRESSION vs CENSORING, per group")
print("="*104)
print("excess = (main-effect RMS ratio) / (interaction RMS ratio);  1.0 = interactions no worse than mains\n")
print(f"{'group':12s} {'%cens':>7s} {'n interior':>11s} {'main ratio':>11s} {'int ratio':>10s} "
      f"{'excess':>7s} {'max depl':>9s} {'artifact terms':>15s}")
print("-"*104)
for r in O.sort_values("excess").itertuples():
    print(f"{r.group:12s} {r.pct_censored:6.1f}% {r.n_interior:11d} {r.main_ratio:11.3f} "
          f"{r.int_ratio:10.3f} {r.excess:7.2f} {r.max_depletion:9.3f} {r.n_artifact_terms:12d}/25")

print("\n"+"="*104); print("CORRELATIONS (n = 6 groups)"); print("="*104)
tests=[("% censored","pct_censored"),("n interior rows","n_interior"),
       ("max allele depletion","max_depletion"),("median additive attenuation","median_attenuation"),
       ("# artifact-risk terms","n_artifact_terms"),("RMS fitted interaction","rms_fitted_int"),
       ("RMS fitted main effect","rms_fitted_main")]
print(f"{'predictor of excess compression':32s} {'Pearson':>8s} {'p':>8s} {'Spearman':>9s} {'p':>8s}")
print("-"*72)
rows=[]
for lab,col in tests:
    pr=sst.pearsonr(O[col],O.excess); sr=sst.spearmanr(O[col],O.excess)
    rows.append(dict(predictor=lab,pearson=pr.statistic,p_pearson=pr.pvalue,
                     spearman=sr.statistic,p_spearman=sr.pvalue))
    print(f"{lab:32s} {pr.statistic:8.3f} {pr.pvalue:8.3f} {sr.statistic:9.3f} {sr.pvalue:8.3f}")
pd.DataFrame(rows).to_csv(f"{RES}/excess_compression_correlates.csv",index=False)
O.to_csv(f"{RES}/excess_compression_by_group.csv",index=False)

print("\n"+"="*104); print("WITHOUT THE TWO FLAGGED GROUPS (S309, AZD8895)"); print("="*104)
K=O[~O.group.isin(["S309","AZD8895"])]
print(f"remaining four groups span %censored {K.pct_censored.min():.1f}-{K.pct_censored.max():.1f} "
      f"and excess {K.excess.min():.2f}-{K.excess.max():.2f}")
pr=sst.pearsonr(K.pct_censored,K.excess)
print(f"Pearson(censoring, excess) among the four = {pr.statistic:+.3f} (p = {pr.pvalue:.3f}, n=4)")
print("\nPermutation check on the full six: how often does a random assignment of")
print("censoring rates to groups give |Pearson| at least as large?")
rng=np.random.default_rng(3)
obs=abs(sst.pearsonr(O.pct_censored,O.excess).statistic)
perm=np.array([abs(sst.pearsonr(rng.permutation(O.pct_censored.to_numpy()),O.excess.to_numpy()).statistic)
               for _ in range(20000)])
print(f"  observed |r| = {obs:.3f}   permutation p = {(perm>=obs).mean():.4f}  "
      f"(exact over {720} orderings this is {(perm>=obs).mean()*720:.0f}/720)")
