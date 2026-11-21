import numpy as np, pandas as pd, pickle, sys, warnings
warnings.filterwarnings("ignore")
SP="/private/tmp/claude-501/-Users-jeffrey-ASPred---KGI-ASPred/64d1a943-5906-4598-b634-23276c958be3/scratchpad"
sys.path.insert(0,SP)
from tobit import tobit_fit
from screen import score_screen
P="/Users/jeffrey/ASPred - KGI/ASPred/ASPred"
F=pickle.load(open(f"{SP}/feat.pkl","rb")); cen=pickle.load(open(f"{SP}/cen.pkl","rb"))
feat,names,NP=F["feat"],F["names"],F["NP"]
fidx={k:i for i,k in enumerate(feat)}
tr=pd.read_csv(f"{P}/data/paired/rbd_split_train.csv",usecols=["antibody_name","rbd_sequence","affinity"])
g=tr.rbd_sequence.unique()
Xi=np.zeros((len(g),NP))
for r,s in enumerate(g):
    for p,a in feat:
        if s[p]==a: Xi[r,fidx[(p,a)]]=1.0
ig={s:i for i,s in enumerate(g)}
cells=pd.read_pickle(f"{SP}/cells.pkl")
EST=cells[(cells.mincell>=100)&(~cells.same_group)]
pairs=list(zip(EST.i,EST.j))

ab="REGN10987"; lo,hi=cen[ab]
d=tr[tr.antibody_name==ab]
rng=np.random.default_rng(1); sub=rng.choice(len(d),30000,replace=False)
d=d.iloc[sub]
rows=d.rbd_sequence.map(ig).to_numpy(); y=d.affinity.to_numpy()
Xia=Xi[rows]; X=np.hstack([np.ones((len(y),1)),Xia])
fit=tobit_fit(X,y,lower=lo,upper=hi)
st,sc=score_screen(X,y,fit["beta"],fit["sigma"],lo,hi,pairs,Xia)

print(f"VALIDATION: score test vs direct Tobit refit  (antibody {ab}, n={len(y)})")
o=np.argsort(-st)
check=[o[0],o[1],o[2],o[len(o)//2],o[-1]]
print(f"{'pair':30s} {'score chi2':>11s} {'score z':>9s} | {'Wald z (refit)':>15s} {'LR chi2 (refit)':>16s}")
print("-"*92)
for t in check:
    i,j=pairs[t]
    c=(Xia[:,i]*Xia[:,j])[:,None]
    f2=tobit_fit(np.hstack([X,c]),y,lower=lo,upper=hi)
    wald=f2["beta"][-1]/f2["se"][-1]
    lr=2*(f2["loglik"]-fit["loglik"])
    print(f"{names[i]+' x '+names[j]:30s} {st[t]:11.2f} {sc[t]:9.2f} | {wald:15.2f} {lr:16.2f}")
