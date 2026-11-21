import numpy as np, pandas as pd, pickle, sys, warnings
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
_os.makedirs(SP,exist_ok=True)

warnings.filterwarnings("ignore")

sys.path.insert(0,_HERE)
from tobit import tobit_fit
from screen import score_screen
from scipy import stats as sstats

F=pickle.load(open(f"{SP}/feat.pkl","rb")); cen=pickle.load(open(f"{SP}/cen.pkl","rb"))
ad=pickle.load(open(f"{SP}/additive.pkl","rb"))
feat,names,NP,ABS=F["feat"],F["names"],F["NP"],F["ABS"]
fidx={k:i for i,k in enumerate(feat)}
K=25

tr=pd.read_csv(f"{P}/data/paired/rbd_split_train.csv",usecols=["antibody_name","rbd_sequence","affinity"])
g=tr.rbd_sequence.unique()
Xi=np.zeros((len(g),NP))
for r,s in enumerate(g):
    for p,a in feat:
        if s[p]==a: Xi[r,fidx[(p,a)]]=1.0
ig={s:i for i,s in enumerate(g)}
cells=pd.read_pickle(f"{SP}/cells.pkl")
EST=cells[(cells.mincell>=100)&(~cells.same_group)].reset_index(drop=True)
pairs=list(zip(EST.i,EST.j))
cellmap={(r.i,r.j):(r.n00,r.n10,r.n01,r.n11) for r in EST.itertuples()}

# ---- genotype-level screen/confirm split --------------------------------
rng=np.random.default_rng(20251120)
half=rng.random(len(g))<0.5
print("================ SCREEN / CONFIRM SPLIT (by genotype, disjoint) ================")
print(f"screen genotypes {half.sum()}   confirm genotypes {(~half).sum()}")
print("screening (which terms to look at) and inference (their coefficients/p-values) use")
print("DISJOINT genotype sets, so the reported p-values are not selection-contaminated.\n")

def fit_and_test(ab, variant, mask_geno):
    """variant 'tobit' = all rows, censoring-aware;  'interior' = uncensored rows, plain Gaussian."""
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]
    rows=d.rbd_sequence.map(ig).to_numpy()
    keep=mask_geno[rows]
    d=d[keep]; rows=rows[keep]
    y=d.affinity.to_numpy()
    if variant=="interior":
        m=(y>lo)&(y<hi); y=y[m]; rows=rows[m]
        L,H=-np.inf,np.inf
    else:
        L,H=lo,hi
    Xia=Xi[rows]; X=np.hstack([np.ones((len(y),1)),Xia])
    return X,Xia,y,L,H

res={}
for variant in ["tobit","interior"]:
    for ab in ABS:
        X,Xia,y,L,H=fit_and_test(ab,variant,half)
        f0=tobit_fit(X,y,lower=L,upper=H)
        st,sc=score_screen(X,y,f0["beta"],f0["sigma"],L,H,pairs,Xia)
        top=np.argsort(-st)[:K]
        # ---- confirm on the other half ----
        Xc,Xic,yc,Lc,Hc=fit_and_test(ab,variant,~half)
        add=[(Xic[:,i]*Xic[:,j]) for i,j in [pairs[t] for t in top]]
        Xj=np.hstack([Xc]+[a[:,None] for a in add])
        fj=tobit_fit(Xj,yc,lower=Lc,upper=Hc)
        b=fj["beta"][-K:]; se=fj["se"][-K:]
        z=b/se; p=2*sstats.norm.sf(np.abs(z))
        o=np.argsort(p); q=np.empty(K)
        ranked=p[o]*K/np.arange(1,K+1)
        q[o]=np.minimum.accumulate(ranked[::-1])[::-1]
        res[(variant,ab)]={"top":[pairs[t] for t in top],"screen_chi2":st[top],
            "b":b,"se":se,"z":z,"p":p,"q":np.clip(q,0,1),"n_screen":len(y),"n_conf":len(yc),
            "sigma":fj["sigma"],"conv":fj["converged"]}
        print(f"  {variant:9s} {ab:12s} screen n={len(y):7d}  confirm n={len(yc):7d}  conv={fj['converged']}")
pickle.dump({"res":res,"pairs":pairs,"half":half,"K":K,"cellmap":cellmap},open(f"{SP}/scan.pkl","wb"))
print("\nsaved scan.pkl")
