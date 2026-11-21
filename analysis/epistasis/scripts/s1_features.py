import pandas as pd, numpy as np, collections, pickle
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
_os.makedirs(SP,exist_ok=True)


OFF=331

tr=pd.read_csv(f"{P}/data/paired/rbd_split_train.csv",usecols=["antibody_name","rbd_sequence","affinity"])
va=pd.read_csv(f"{P}/data/paired/rbd_split_val.csv",usecols=["antibody_name","rbd_sequence","affinity"])
te=pd.read_csv(f"{P}/data/paired/rbd_split_test.csv",usecols=["antibody_name","rbd_sequence","affinity"])
ABS=sorted(tr.antibody_name.unique())
print("antibodies:",ABS)
for nm,d in [("train",tr),("val",va),("test",te)]:
    print(f"{nm:6s} rows {len(d):8d}  genotypes {d.rbd_sequence.nunique():7d}  abs {d.antibody_name.nunique()}")
# leakage guard
sets={nm:set(d.rbd_sequence.unique()) for nm,d in [("train",tr),("val",va),("test",te)]}
for a in ["train","val","test"]:
    for b in ["train","val","test"]:
        if a<b: print(f"genotype overlap {a}/{b}: {len(sets[a]&sets[b])}")

# ---- feature definition: TRAIN genotypes only, >=1% frequency ----------
g=tr.rbd_sequence.unique(); L=len(g[0])
A=np.frombuffer("".join(g).encode(),dtype="S1").reshape(len(g),L)
ref={}; cand=[]
for p in range(L):
    c=collections.Counter(A[:,p].tobytes().decode())
    if len(c)>1:
        ref[p]=c.most_common(1)[0][0]
        for a,n in c.most_common()[1:]:
            cand.append((p,a,n,n/len(g)))
fr=pd.DataFrame(cand,columns=["p","alt","n","freq"])
keep=fr[fr.freq>=0.01].copy()
feat=[(int(r.p),r.alt) for r in keep.itertuples()]
NP=len(feat); fidx={k:i for i,k in enumerate(feat)}
names=[f"{p+OFF}{ref[p]}>{a}" for p,a in feat]
print(f"\nvariable positions (train): {fr.p.nunique()}")
print(f"indicators at >=1% freq: {NP}   pairs: {NP*(NP-1)//2}")
print(f"dropped rare alleles: {len(fr)-NP} (max freq dropped {fr[fr.freq<0.01].freq.max():.5f})")

def design(genos):
    X=np.zeros((len(genos),NP),dtype=np.float64)
    for r,s in enumerate(genos):
        for p,a in feat:
            if s[p]==a: X[r,fidx[(p,a)]]=1.0
    return X

pickle.dump({"feat":feat,"names":names,"ref":ref,"NP":NP,"ABS":ABS},open(f"{SP}/feat.pkl","wb"))

# ---- censoring structure ------------------------------------------------
print("\n\n================ CENSORING STRUCTURE (train split) ================")
print(f"{'antibody':12s} {'n':>8s} {'min':>8s} {'max':>8s} | {'@min':>8s} {'%':>7s} | {'@max':>8s} {'%':>7s} | {'interior':>9s} {'%':>7s}")
print("-"*100)
cen={}
for ab in ABS:
    y=tr.loc[tr.antibody_name==ab,"affinity"].to_numpy()
    lo,hi=y.min(),y.max()
    nlo=(y==lo).sum(); nhi=(y==hi).sum()
    # a bound counts as censoring only if a real point mass sits on it
    L_=lo if nlo/len(y)>0.02 else -np.inf
    H_=hi if nhi/len(y)>0.02 else np.inf
    inter=((y>L_)&(y<H_)).sum()
    cen[ab]=(L_,H_)
    print(f"{ab:12s} {len(y):8d} {lo:8.3f} {hi:8.3f} | {nlo:8d} {100*nlo/len(y):6.2f}% | {nhi:8d} {100*nhi/len(y):6.2f}% | {inter:9d} {100*inter/len(y):6.2f}%")
print("\ndetected censoring bounds per antibody (used as Tobit limits):")
for ab in ABS: print(f"  {ab:12s} lower={cen[ab][0]}  upper={cen[ab][1]}")
pickle.dump(cen,open(f"{SP}/cen.pkl","wb"))
