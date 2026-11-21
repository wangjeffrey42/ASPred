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
AL=pickle.load(open(f"{SP}/alias.pkl","rb"))
feat,names,NP,ABS=F["feat"],F["names"],F["NP"],F["ABS"]
fidx={k:i for i,k in enumerate(feat)}
K=25

# ---- one representative per alias group --------------------------------
groups=AL["groups"]
reps=[]; alias_of={}
for root,mem in groups.items():
    mem=sorted(mem)
    reps.append(mem[0])
    if len(mem)>1: alias_of[mem[0]]=[names[m] for m in mem[1:]]
reps=sorted(reps)
print("================ CANDIDATE SET (alias-collapsed) ================")
print(f"indicators: {NP}  ->  representatives: {len(reps)}")
for r,part in alias_of.items():
    print(f"  {names[r]:14s} also stands for {', '.join(part)}  (r>0.99; not separable)")

tr=pd.read_csv(f"{P}/data/paired/rbd_split_train.csv",usecols=["antibody_name","rbd_sequence","affinity"])
g=tr.rbd_sequence.unique()
Xi=np.zeros((len(g),NP))
for r,s in enumerate(g):
    for p,a in feat:
        if s[p]==a: Xi[r,fidx[(p,a)]]=1.0
ig={s:i for i,s in enumerate(g)}

rng=np.random.default_rng(20251120)
half=rng.random(len(g))<0.5

# candidate pairs among representatives, cell counts checked in BOTH halves
cand=[(i,j) for a,i in enumerate(reps) for j in reps[a+1:]]
def mincells(mask):
    Xh=Xi[mask]; n=len(Xh); n1=Xh.sum(0); n11=Xh.T@Xh
    out={}
    for i,j in cand:
        a=n11[i,j]; b=n1[i]-a; c=n1[j]-a; d=n-a-b-c
        out[(i,j)]=min(a,b,c,d)
    return out
ms,mc=mincells(half),mincells(~half)
mall=mincells(np.ones(len(g),bool))
pairs=[p for p in cand if min(ms[p],mc[p])>=50]
print(f"\npairs among representatives: {len(cand)}")
print(f"pairs with all four states >=50 in BOTH halves: {len(pairs)}  (dropped {len(cand)-len(pairs)})")
cellmap={}
Xall=Xi; n=len(Xall); n1=Xall.sum(0); n11=Xall.T@Xall
for i,j in cand:
    a=n11[i,j]; b=n1[i]-a; c=n1[j]-a; d=n-a-b-c
    cellmap[(i,j)]=(int(d),int(b),int(c),int(a))

# ---- additive coefficient signs, for the censoring-artifact flag -------
ad=pickle.load(open(f"{SP}/additive.pkl","rb"))
toward={}
for ab in ABS:
    lo,hi=cen[ab]; b=ad[ab]["beta"][1:]
    toward[ab]=(b<0) if np.isfinite(lo) else (b>0)   # True = pushes response toward the bound

def build(ab,variant,mask_geno):
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]; rows=d.rbd_sequence.map(ig).to_numpy()
    keep=mask_geno[rows]; d=d[keep]; rows=rows[keep]
    y=d.affinity.to_numpy()
    if variant=="interior":
        m=(y>lo)&(y<hi); y=y[m]; rows=rows[m]; L,H=-np.inf,np.inf
    else: L,H=lo,hi
    Xia=Xi[rows]
    return np.hstack([np.ones((len(y),1)),Xia]),Xia,y,L,H

res={}
print("\n================ SCAN: screen on half A, confirm on half B ================")
for variant in ["tobit","interior"]:
    for ab in ABS:
        X,Xia,y,L,H=build(ab,variant,half)
        f0=tobit_fit(X,y,lower=L,upper=H)
        st,sc=score_screen(X,y,f0["beta"],f0["sigma"],L,H,pairs,Xia)
        top=[pairs[t] for t in np.argsort(-st)[:K]]
        tst=st[np.argsort(-st)[:K]]
        Xc,Xic,yc,Lc,Hc=build(ab,variant,~half)
        Xj=np.hstack([Xc]+[(Xic[:,i]*Xic[:,j])[:,None] for i,j in top])
        cn=np.linalg.cond(Xj)
        fj=tobit_fit(Xj,yc,lower=Lc,upper=Hc,maxiter=5000,tol=1e-12)
        b=fj["beta"][-K:]; se=fj["se"][-K:]
        z=np.where(se>0,b/np.where(se>0,se,1),np.nan)
        p=2*sstats.norm.sf(np.abs(z))
        o=np.argsort(np.nan_to_num(p,nan=1.0)); q=np.empty(K)
        rk=np.nan_to_num(p,nan=1.0)[o]*K/np.arange(1,K+1)
        q[o]=np.minimum.accumulate(rk[::-1])[::-1]
        res[(variant,ab)]={"top":top,"screen_chi2":tst,"b":b,"se":se,"z":z,"p":p,
            "q":np.clip(q,0,1),"n_screen":len(y),"n_conf":len(yc),"sigma":fj["sigma"],
            "conv":fj["converged"],"cond":cn,"bad_se":int((se<=0).sum()|np.isnan(se).sum())}
        print(f"  {variant:9s} {ab:12s} screen n={len(y):7d} confirm n={len(yc):7d} "
              f"conv={str(fj['converged']):5s} cond={cn:8.1f} bad_se={int((se<=0).sum()+np.isnan(se).sum())}")
pickle.dump({"res":res,"pairs":pairs,"half":half,"K":K,"cellmap":cellmap,"reps":reps,
             "alias_of":alias_of,"toward":toward,"names":names},open(f"{SP}/scan2.pkl","wb"))
print("\nsaved scan2.pkl")
