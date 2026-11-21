import numpy as np, pandas as pd, pickle, warnings
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
_os.makedirs(SP,exist_ok=True)

warnings.filterwarnings("ignore")


F=pickle.load(open(f"{SP}/feat.pkl","rb")); cen=pickle.load(open(f"{SP}/cen.pkl","rb"))
D=pickle.load(open(f"{SP}/scan2.pkl","rb"))
feat,names,NP,ABS=F["feat"],F["names"],F["NP"],F["ABS"]
fidx={k:i for i,k in enumerate(feat)}
tr=pd.read_csv(f"{P}/data/paired/rbd_split_train.csv",usecols=["antibody_name","rbd_sequence","affinity"])
g=tr.rbd_sequence.unique()
Xi=np.zeros((len(g),NP))
for r,s in enumerate(g):
    for p,a in feat:
        if s[p]==a: Xi[r,fidx[(p,a)]]=1.0
ig={s:i for i,s in enumerate(g)}
reps=D["reps"]

print("=========== INTERIOR-SUBSET IDENTIFIABILITY, per group ===========")
print("Base design = intercept + 46 allele indicators, restricted to interior rows.")
print(f"{'group':12s} {'n int':>8s} {'rank/ncol':>12s} {'cond':>10s} | collinear direction (top loadings)")
print("-"*110)
drop=[]
for ab in ABS:
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]; rows=d.rbd_sequence.map(ig).to_numpy()
    y=d.affinity.to_numpy(); X=Xi[rows]
    un=(y>lo)&(y<hi)
    Xc=np.hstack([np.ones((un.sum(),1)),X[un]])
    rk=np.linalg.matrix_rank(Xc); cn=np.linalg.cond(Xc)
    U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
    v=Vt[-1]
    load=np.argsort(-np.abs(v))[:4]
    lab=", ".join(("intercept" if k==0 else names[k-1])+f"({v[k]:+.2f})" for k in load)
    bad = rk<Xc.shape[1] or cn>1e8
    if bad: drop.append(ab)
    print(f"{ab:12s} {un.sum():8d} {rk:6d}/{Xc.shape[1]:<5d} {cn:10.2e} | {lab}{'   <== DEGENERATE' if bad else ''}")
print("\nGroups where interior-only is not estimable:", drop)

print("\n\n=========== WHY: alleles that disappear from the interior subset ===========")
for ab in drop:
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]; rows=d.rbd_sequence.map(ig).to_numpy()
    y=d.affinity.to_numpy(); X=Xi[rows]; un=(y>lo)&(y<hi)
    bnd="lower" if np.isfinite(lo) else "upper"
    print(f"\n--- {ab}  (censored at the {bnd} bound; {100*(~un).mean():.1f}% of rows censored)")
    fa=X.mean(0); fi=X[un].mean(0)
    # conditional censoring rate given allele present
    for j in np.argsort(fi/np.maximum(fa,1e-12))[:4]:
        pres=X[:,j]==1
        print(f"    {names[j]:12s} freq all {fa[j]:.3f} -> interior {fi[j]:.3f}   "
              f"P(censored | allele) = {100*(~un[pres]).mean():5.1f}%   vs {100*(~un[~pres]).mean():5.1f}% without")
pickle.dump(drop,open(f"{SP}/drop_groups.pkl","wb"))
