import pandas as pd, numpy as np, pickle, sys, warnings
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
_os.makedirs(SP,exist_ok=True)

warnings.filterwarnings("ignore"); sys.path.insert(0,_HERE)
from scipy import stats
from scipy.special import ndtr
from scipy.stats import norm

F=pickle.load(open(f"{SP}/feat.pkl","rb")); cen=pickle.load(open(f"{SP}/cen.pkl","rb"))
ad=pickle.load(open(f"{SP}/additive.pkl","rb"))
feat,names,NP,ABS=F["feat"],F["names"],F["NP"],F["ABS"]
fidx={k:i for i,k in enumerate(feat)}
tr=pd.read_csv(f"{P}/data/paired/rbd_split_train.csv",usecols=["antibody_name","rbd_sequence","affinity"])
g=tr.rbd_sequence.unique()
X=np.zeros((len(g),NP))
for r,s in enumerate(g):
    for p,a in feat:
        if s[p]==a: X[r,fidx[(p,a)]]=1.0

print("================ DESIGN MATRIX DIAGNOSTICS (genotype level, n=%d) ================"%len(g))
fr=X.mean(0)
print("allele frequencies: min %.4f  median %.4f  max %.4f"%(fr.min(),np.median(fr),fr.max()))
Xc=np.hstack([np.ones((len(g),1)),X])
s=np.linalg.svd(Xc,compute_uv=False)
print("design rank: %d / %d   condition number: %.3g"%(np.linalg.matrix_rank(Xc),Xc.shape[1],s[0]/s[-1]))

C=np.corrcoef(X.T)
np.fill_diagonal(C,0)
print("\nmost correlated indicator pairs (|r|>0.5):")
iu=np.triu_indices(NP,1)
ordr=np.argsort(-np.abs(C[iu]))
nshow=0
for t in ordr:
    i,j=iu[0][t],iu[1][t]
    if abs(C[i,j])<=0.5: break
    print(f"  {names[i]:14s} {names[j]:14s} r={C[i,j]:+.4f}")
    nshow+=1
print(f"  ({nshow} pairs above 0.5)" if nshow else "  none")

# VIF
print("\nhighest VIFs:")
vif=[]
for j in range(NP):
    o=[k for k in range(NP) if k!=j]
    A=np.hstack([np.ones((len(g),1)),X[:,o]])
    b,*_=np.linalg.lstsq(A,X[:,j],rcond=None)
    r2=1-((X[:,j]-A@b)**2).sum()/((X[:,j]-X[:,j].mean())**2).sum()
    vif.append((1/max(1-r2,1e-12),names[j]))
for v,n in sorted(vif,reverse=True)[:10]: print(f"  {n:14s} VIF={v:12.1f}")

print("\n\n================ IS THE NEGATIVE INTERIOR R2 A BUG OR A METRIC ARTIFACT? ================")
print("Tobit fits the LATENT mean Xb.  On interior rows the error is truncated, so E[y|interior] != Xb")
print("and plain R2 of Xb against interior y is biased downward by construction.  The right checks are")
print("(a) does the model predict WHICH rows censor, and (b) observed-scale E[y|x] fit on all rows.\n")
itr={s_:i for i,s_ in enumerate(g)}
print(f"{'antibody':12s} | {'pred %cens':>10s} {'actual %cens':>12s} {'AUC cens':>9s} | {'obs-scale R2':>12s} {'rho(all)':>9s} | {'LLR vs null':>12s}")
print("-"*105)
for ab in ABS:
    lo,hi=cen[ab]; r=ad[ab]
    d=tr[tr.antibody_name==ab]; rows=d.rbd_sequence.map(itr).to_numpy()
    Xa=Xc[rows]; y=d.affinity.to_numpy()
    mu=Xa@r["beta"]; sg=r["sigma"]
    if np.isfinite(lo):
        pc=ndtr((lo-mu)/sg); isc=(y<=lo)
        a=(mu-lo)/sg
        Ey=lo*ndtr(-a)+mu*ndtr(a)+sg*norm.pdf(a)
    else:
        pc=ndtr((mu-hi)/sg); isc=(y>=hi)
        a=(hi-mu)/sg
        Ey=hi*(1-ndtr(a))+mu*ndtr(a)-sg*norm.pdf(a)
    auc=stats.mannwhitneyu(pc[isc],pc[~isc]).statistic/(isc.sum()*(~isc).sum())
    r2o=1-((y-Ey)**2).sum()/((y-y.mean())**2).sum()
    rho=stats.spearmanr(mu,y).statistic
    # null loglik: intercept + sigma only
    from tobit import tobit_fit
    n0=tobit_fit(np.ones((len(y),1)),y,lower=lo,upper=hi)
    llr=2*(r["loglik"]-n0["loglik"])
    print(f"{ab:12s} | {100*pc.mean():9.2f}% {100*isc.mean():11.2f}% {auc:9.4f} | {r2o:12.4f} {rho:9.4f} | {llr:12.0f}")
print("\nAUC cens = how well the fitted model ranks which rows hit the bound (0.5=chance)")
print("obs-scale R2 = R2 of the Tobit's E[y|x] (censoring-aware) against observed y, ALL rows")
print("LLR vs null = 2*(loglik - intercept-only loglik), df=46; chi2 0.1% crit ~ 85")
