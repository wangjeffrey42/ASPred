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
D2=pickle.load(open(f"{SP}/scan2.pkl","rb")); ad=pickle.load(open(f"{SP}/additive.pkl","rb"))
feat,names,NP,ABS=F["feat"],F["names"],F["NP"],F["ABS"]
fidx={k:i for i,k in enumerate(feat)}
reps=D2["reps"]; alias_of=D2["alias_of"]; pairs=D2["pairs"]; K=25
tr=pd.read_csv(f"{P}/data/paired/rbd_split_train.csv",usecols=["antibody_name","rbd_sequence","affinity"])
g=tr.rbd_sequence.unique()
Xi=np.zeros((len(g),NP))
for r,s in enumerate(g):
    for p,a in feat:
        if s[p]==a: Xi[r,fidx[(p,a)]]=1.0
ig={s:i for i,s in enumerate(g)}
rng=np.random.default_rng(20251120); half=rng.random(len(g))<0.5

toward={}
for ab in ABS:
    lo,hi=cen[ab]; b=ad[ab]["beta"][1:]
    toward[ab]=(b<0) if np.isfinite(lo) else (b>0)

def build(ab,variant,mask):
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]; rows=d.rbd_sequence.map(ig).to_numpy()
    keep=mask[rows]; d=d[keep]; rows=rows[keep]; y=d.affinity.to_numpy()
    if variant=="interior":
        m=(y>lo)&(y<hi); y=y[m]; rows=rows[m]; L,H=-np.inf,np.inf
    else: L,H=lo,hi
    Xf=Xi[rows]
    return np.hstack([np.ones((len(y),1)),Xf[:,reps]]),Xf,y,L,H

res={}
print("=== SCAN (alias-collapsed base: intercept + 43 representatives) ===")
print("screen on genotype half A, confirm on disjoint half B\n")
for variant in ["tobit","interior"]:
    for ab in ABS:
        X,Xf,y,L,H=build(ab,variant,half)
        f0=tobit_fit(X,y,lower=L,upper=H)
        st,_=score_screen(X,y,f0["beta"],f0["sigma"],L,H,pairs,Xf)
        o=np.argsort(-st)[:K]; top=[pairs[t] for t in o]
        Xc,Xfc,yc,Lc,Hc=build(ab,variant,~half)
        Xj=np.hstack([Xc]+[(Xfc[:,i]*Xfc[:,j])[:,None] for i,j in top])
        cn=np.linalg.cond(Xj); rk=np.linalg.matrix_rank(Xj)
        fj=tobit_fit(Xj,yc,lower=Lc,upper=Hc,maxiter=5000,tol=1e-12)
        b=fj["beta"][-K:]; se=fj["se"][-K:]; z=b/se; p=2*sstats.norm.sf(np.abs(z))
        ordp=np.argsort(p); q=np.empty(K)
        rk_=p[ordp]*K/np.arange(1,K+1); q[ordp]=np.minimum.accumulate(rk_[::-1])[::-1]
        res[(variant,ab)]={"top":top,"screen_chi2":st[o],"b":b,"se":se,"z":z,"p":p,
            "q":np.clip(q,0,1),"n_screen":len(y),"n_conf":len(yc),"sigma":fj["sigma"],
            "conv":fj["converged"],"cond":cn,"rank":rk,"ncol":Xj.shape[1]}
        print(f"  {variant:8s} {ab:12s} screen n={len(y):7d} confirm n={len(yc):7d} "
              f"conv={str(fj['converged']):5s} rank={rk}/{Xj.shape[1]} cond={cn:7.1f}")
pickle.dump({"res":res,"pairs":pairs,"half":half,"K":K,"reps":reps,"alias_of":alias_of,
             "toward":toward,"names":names,"cellmap":D2["cellmap"]},open(f"{SP}/scan3.pkl","wb"))
print("\nsaved scan3.pkl")
