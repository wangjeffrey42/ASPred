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
reps=D["reps"]; K=D["K"]
tr=pd.read_csv(f"{P}/data/paired/rbd_split_train.csv",usecols=["antibody_name","rbd_sequence","affinity"])
g=tr.rbd_sequence.unique()
Xi=np.zeros((len(g),NP))
for r,s in enumerate(g):
    for p,a in feat:
        if s[p]==a: Xi[r,fidx[(p,a)]]=1.0
ig={s:i for i,s in enumerate(g)}
R=Xi[:,reps]   # alias-collapsed design

print("=========== INTERIOR IDENTIFIABILITY on the alias-collapsed design ===========")
print("Design = intercept + 43 representative indicators, restricted to interior rows.\n")
print(f"{'group':12s} {'n all':>8s} {'n int':>8s} {'%cens':>6s} {'rank/ncol':>11s} {'cond':>10s}  {'min n_int':>9s}  status")
print("-"*105)
stat={}
for ab in ABS:
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]; rows=d.rbd_sequence.map(ig).to_numpy()
    y=d.affinity.to_numpy(); X=R[rows]; un=(y>lo)&(y<hi)
    Xc=np.hstack([np.ones((un.sum(),1)),X[un]])
    rk=np.linalg.matrix_rank(Xc); cn=np.linalg.cond(Xc)
    mn=X[un].sum(0).min()
    bad = rk<Xc.shape[1] or cn>1e8
    stat[ab]=dict(n=len(y),nint=int(un.sum()),cens=100*(~un).mean(),rank=rk,ncol=Xc.shape[1],cond=cn,minint=int(mn),bad=bad)
    print(f"{ab:12s} {len(y):8d} {un.sum():8d} {100*(~un).mean():5.1f}% {rk:5d}/{Xc.shape[1]:<5d} {cn:10.2e} {mn:9.0f}  {'DEGENERATE' if bad else 'ok'}")

print("\n\n=========== INTERIOR design WITH the 25 confirmed interaction columns ===========")
print("(this is the design the interior-only confirm fit actually used)\n")
print(f"{'group':12s} {'rank/ncol':>11s} {'cond':>10s}  {'worst interaction cell in interior':>36s}")
print("-"*95)
drop=[]
for ab in ABS:
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]; rows=d.rbd_sequence.map(ig).to_numpy()
    y=d.affinity.to_numpy(); X=Xi[rows]; un=(y>lo)&(y<hi)
    top=D["res"][("interior",ab)]["top"]
    Xr=X[un][:,reps]
    IA=np.column_stack([X[un][:,i]*X[un][:,j] for i,j in top])
    Xc=np.hstack([np.ones((un.sum(),1)),Xr,IA])
    rk=np.linalg.matrix_rank(Xc); cn=np.linalg.cond(Xc)
    # worst 2x2 cell among top terms within interior rows
    worst=1e18; wlab=""
    for i,j in top:
        a=X[un][:,i]; b=X[un][:,j]
        c=min(((a==0)&(b==0)).sum(),((a==1)&(b==0)).sum(),((a==0)&(b==1)).sum(),((a==1)&(b==1)).sum())
        if c<worst: worst=c; wlab=f"{names[i]}*{names[j]}: min cell {c}"
    bad = rk<Xc.shape[1] or cn>1e8
    if bad: drop.append(ab)
    print(f"{ab:12s} {rk:5d}/{Xc.shape[1]:<5d} {cn:10.2e}  {wlab:>36s}  {'DEGENERATE' if bad else ''}")
print("\nGroups to drop from the interior-only comparison:", drop)
pickle.dump({"drop":drop,"stat":stat},open(f"{SP}/drop_groups.pkl","wb"))

print("\n\n=========== WHAT CENSORING DOES TO THOSE GROUPS' ALLELES ===========")
for ab in drop:
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]; rows=d.rbd_sequence.map(ig).to_numpy()
    y=d.affinity.to_numpy(); X=Xi[rows]; un=(y>lo)&(y<hi)
    bnd="upper" if np.isfinite(hi) else "lower"
    print(f"\n--- {ab}: {100*(~un).mean():.1f}% of rows at the {bnd} bound, interior n={un.sum()}")
    fa=X.mean(0); fi=X[un].mean(0)
    for j in np.argsort(fi/np.maximum(fa,1e-12))[:5]:
        pres=X[:,j]==1
        print(f"    {names[j]:12s} freq {fa[j]:.3f} -> {fi[j]:.3f} in interior   "
              f"P(cens|allele)={100*(~un[pres]).mean():5.1f}%  P(cens|no allele)={100*(~un[~pres]).mean():5.1f}%")
