import numpy as np, pandas as pd, pickle, warnings
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
_os.makedirs(SP,exist_ok=True)

warnings.filterwarnings("ignore")


F=pickle.load(open(f"{SP}/feat.pkl","rb")); cen=pickle.load(open(f"{SP}/cen.pkl","rb"))
ad=pickle.load(open(f"{SP}/additive.pkl","rb")); D=pickle.load(open(f"{SP}/scan2.pkl","rb"))
feat,names,NP,ABS=F["feat"],F["names"],F["NP"],F["ABS"]
fidx={k:i for i,k in enumerate(feat)}
reps=D["reps"]
tr=pd.read_csv(f"{P}/data/paired/rbd_split_train.csv",usecols=["antibody_name","rbd_sequence","affinity"])
g=tr.rbd_sequence.unique()
Xi=np.zeros((len(g),NP))
for r,s in enumerate(g):
    for p,a in feat:
        if s[p]==a: Xi[r,fidx[(p,a)]]=1.0
ig={s:i for i,s in enumerate(g)}
from scipy import stats as sst

print("=========== SELECTION-ON-OUTCOME SEVERITY (interior subsetting) ===========")
print("For each group: how strongly does an allele's ADDITIVE effect toward the bound")
print("predict its depletion in the interior subset?  Strong coupling => conditioning on")
print("interior rows selects on the outcome and removes the very effect being estimated.\n")
print(f"{'group':12s} {'%cens':>6s} {'int n':>7s} | {'corr(effect,depletion)':>22s} {'max depletion':>13s} {'allele':>10s} | {'# alleles >50% depleted':>23s}")
print("-"*125)
sev={}
for ab in ABS:
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]; rows=d.rbd_sequence.map(ig).to_numpy()
    y=d.affinity.to_numpy(); X=Xi[rows]; un=(y>lo)&(y<hi)
    fa=X.mean(0); fi=X[un].mean(0)
    dep=1-fi/np.maximum(fa,1e-12)              # fraction of the allele's mass lost
    b=ad[ab]["beta"][1:]
    push = -b if np.isfinite(lo) else b        # >0 means pushes toward the bound
    r=sst.spearmanr(push[reps],dep[reps]).statistic
    k=int(np.argmax(dep[reps])); worst=reps[k]
    n50=int((dep[reps]>0.5).sum())
    sev[ab]=dict(rho=r,maxdep=dep[worst],worst=names[worst],n50=n50,cens=100*(~un).mean(),nint=int(un.sum()))
    print(f"{ab:12s} {100*(~un).mean():5.1f}% {un.sum():7d} | {r:22.3f} {dep[worst]:13.3f} {names[worst]:>10s} | {n50:23d}")

print("\n\n=========== EFFECT ATTENUATION: additive coefficients, all rows vs interior ===========")
print("Tobit on all rows (censoring-aware) vs OLS on interior rows only.")
print("Attenuation = interior coef / tobit coef, for the 5 alleles that push hardest toward the bound.\n")
att={}
for ab in ABS:
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]; rows=d.rbd_sequence.map(ig).to_numpy()
    y=d.affinity.to_numpy(); X=Xi[rows]; un=(y>lo)&(y<hi)
    Xr=np.hstack([np.ones((un.sum(),1)),X[un][:,reps]])
    bi,*_=np.linalg.lstsq(Xr,y[un],rcond=None); bi=bi[1:]
    bt=ad[ab]["beta"][1:][reps]
    push=-bt if np.isfinite(lo) else bt
    o=np.argsort(-push)[:5]
    rat=bi[o]/bt[o]
    att[ab]=float(np.median(bi[push>0]/bt[push>0]))
    print(f"{ab:12s} median attenuation over all bound-pushing alleles: {att[ab]:.2f}")
    print("    "+"  ".join(f"{names[reps[j]]}: {bt[j]:+.3f}->{bi[j]:+.3f} ({rat[k]:.2f}x)" for k,j in enumerate(o)))
pickle.dump({"sev":sev,"att":att},open(f"{SP}/selection.pkl","wb"))
