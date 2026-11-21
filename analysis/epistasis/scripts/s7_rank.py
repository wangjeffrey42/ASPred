import numpy as np, pandas as pd, pickle, sys, warnings
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
_os.makedirs(SP,exist_ok=True)

warnings.filterwarnings("ignore")

sys.path.insert(0,_HERE)

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

print("================ WHY INTERIOR-ONLY LOSES IDENTIFIABILITY ================")
print("For each antibody: allele frequency among ALL rows vs among INTERIOR rows.")
print("An allele that almost always drives the response past the bound nearly vanishes")
print("from the interior subset, so its effect (and any interaction with it) is unidentified.\n")
for ab in ABS:
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]; rows=d.rbd_sequence.map(ig).to_numpy()
    y=d.affinity.to_numpy(); X=Xi[rows]
    un=(y>lo)&(y<hi)
    fa=X.mean(0); fi=X[un].mean(0)
    Xc=np.hstack([np.ones((un.sum(),1)),X[un]])
    rk=np.linalg.matrix_rank(Xc); cn=np.linalg.cond(Xc)
    print(f"--- {ab}   interior n={un.sum()}   design rank {rk}/{Xc.shape[1]}   cond {cn:.3g}")
    bad=np.argsort(fi/np.maximum(fa,1e-12))[:5]
    print(f"    {'allele':14s} {'freq(all)':>10s} {'freq(interior)':>15s} {'ratio':>8s} {'n interior':>11s}")
    for j in bad:
        print(f"    {names[j]:14s} {fa[j]:10.4f} {fi[j]:15.6f} {fi[j]/max(fa[j],1e-12):8.4f} {X[un][:,j].sum():11.0f}")
    nz=(X[un].sum(0)==0).sum(); few=(X[un].sum(0)<20).sum()
    print(f"    alleles with ZERO interior observations: {nz};  with <20: {few}")
