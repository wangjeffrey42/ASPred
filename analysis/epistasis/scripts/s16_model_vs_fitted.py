"""Correlate model-implied interactions against the fitted Tobit coefficients."""
import os as _os, pickle, warnings
warnings.filterwarnings("ignore")
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache"); RES=_os.path.join(_os.path.dirname(_HERE),"results")
import numpy as np, pandas as pd
from scipy import stats as sst
F=pickle.load(open(f"{SP}/feat.pkl","rb")); D=pickle.load(open(f"{SP}/scan3.pkl","rb"))
names,ABS=D["names"],F["ABS"]; res=D["res"]
M=pd.read_csv(f"{RES}/model_implied_interactions.csv")

# fitted coefficients live only for the 25 confirmed terms per group
fit=[]
for ab in ABS:
    r=res[("tobit",ab)]
    for k,(i,j) in enumerate(r["top"]):
        fit.append(dict(group=ab,term=f"{names[i]} x {names[j]}",coef=r["b"][k],
                        se=r["se"][k],q=r["q"][k],fdr_sig=r["q"][k]<0.05))
FIT=pd.DataFrame(fit)
J=FIT.merge(M,on=["group","term"],how="left")
print("="*100); print("MODEL-IMPLIED vs FITTED INTERACTION COEFFICIENTS"); print("="*100)
print("Model-implied d = y11 - y10 - y01 + y00 from ASPred-Affinity forward passes,")
print("averaged over backgrounds.  Fitted coef = Tobit interaction coefficient (confirm half).\n")
print(f"{'group':12s} {'n terms':>7s} | {'Pearson':>8s} {'p':>9s} {'Spearman':>9s} {'p':>9s} | "
      f"{'--- FDR-significant only ---':>30s}")
print(f"{'':12s} {'':>7s} | {'':>8s} {'':>9s} {'':>9s} {'':>9s} | {'n':>4s} {'Pearson':>8s} {'Spearman':>9s} {'slope':>7s}")
print("-"*118)
out=[]
for ab in ABS:
    d=J[J.group==ab].dropna(subset=["model_interaction"])
    pr=sst.pearsonr(d.model_interaction,d.coef); sr=sst.spearmanr(d.model_interaction,d.coef)
    ds=d[d.fdr_sig]
    if len(ds)>2:
        pr2=sst.pearsonr(ds.model_interaction,ds.coef); sr2=sst.spearmanr(ds.model_interaction,ds.coef)
        sl=np.polyfit(ds.coef,ds.model_interaction,1)[0]
        s2=f"{len(ds):4d} {pr2.statistic:8.3f} {sr2.statistic:9.3f} {sl:7.3f}"
    else:
        pr2=sr2=None; sl=np.nan; s2=f"{len(ds):4d} {'--':>8s} {'--':>9s} {'--':>7s}"
    print(f"{ab:12s} {len(d):7d} | {pr.statistic:8.3f} {pr.pvalue:9.2e} {sr.statistic:9.3f} {sr.pvalue:9.2e} | {s2}")
    out.append(dict(group=ab,n=len(d),pearson=pr.statistic,p_pearson=pr.pvalue,
        spearman=sr.statistic,p_spearman=sr.pvalue,n_sig=len(ds),
        pearson_sig=pr2.statistic if pr2 else np.nan,
        p_pearson_sig=pr2.pvalue if pr2 else np.nan,
        spearman_sig=sr2.statistic if sr2 else np.nan,
        p_spearman_sig=sr2.pvalue if sr2 else np.nan,slope_sig=sl))
pd.DataFrame(out).to_csv(f"{RES}/model_vs_fitted_correlation.csv",index=False)
J.to_csv(f"{RES}/model_vs_fitted_terms.csv",index=False)

print("\n\n"+"="*100); print("MAGNITUDE COMPARISON"); print("="*100)
print(f"{'group':12s} {'RMS fitted coef':>16s} {'RMS model-implied':>18s} {'ratio model/fitted':>19s}")
for ab in ABS:
    d=J[J.group==ab].dropna(subset=["model_interaction"])
    rf=np.sqrt((d.coef**2).mean()); rm=np.sqrt((d.model_interaction**2).mean())
    print(f"{ab:12s} {rf:16.4f} {rm:18.4f} {rm/rf:19.3f}")

print("\n\n"+"="*100); print("ACROSS ALL 867 ESTIMABLE PAIRS: spread of the model's implied interaction")
print("="*100)
print("(fitted coefficients exist only for the 25 confirmed terms, so this is the model side alone)")
print(f"{'group':12s} {'mean':>9s} {'sd':>9s} {'p5':>9s} {'p95':>9s} {'max|d|':>9s} {'median sd across bg':>20s}")
for ab in ABS:
    m=M[M.group==ab]
    print(f"{ab:12s} {m.model_interaction.mean():9.4f} {m.model_interaction.std():9.4f} "
          f"{m.model_interaction.quantile(.05):9.4f} {m.model_interaction.quantile(.95):9.4f} "
          f"{m.model_interaction.abs().max():9.4f} {m.model_interaction_sd.median():20.4f}")
