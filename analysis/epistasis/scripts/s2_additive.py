import pandas as pd, numpy as np, pickle, sys, warnings
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
_os.makedirs(SP,exist_ok=True)

warnings.filterwarnings("ignore")
sys.path.insert(0,_HERE)
from tobit import tobit_fit
from scipy import stats

F=pickle.load(open(f"{SP}/feat.pkl","rb")); cen=pickle.load(open(f"{SP}/cen.pkl","rb"))
feat,names,NP,ABS=F["feat"],F["names"],F["NP"],F["ABS"]
fidx={k:i for i,k in enumerate(feat)}

tr=pd.read_csv(f"{P}/data/paired/rbd_split_train.csv",usecols=["antibody_name","rbd_sequence","affinity"])
va=pd.read_csv(f"{P}/data/paired/rbd_split_val.csv",usecols=["antibody_name","rbd_sequence","affinity"])

def design(genos):
    X=np.zeros((len(genos),NP))
    for r,s in enumerate(genos):
        for p,a in feat:
            if s[p]==a: X[r,fidx[(p,a)]]=1.0
    return X

g_tr=tr.rbd_sequence.unique(); g_va=va.rbd_sequence.unique()
Xtr=np.hstack([np.ones((len(g_tr),1)),design(g_tr)])
Xva=np.hstack([np.ones((len(g_va),1)),design(g_va)])
itr={s:i for i,s in enumerate(g_tr)}; iva={s:i for i,s in enumerate(g_va)}
print(f"design: {Xtr.shape[1]} params (intercept + {NP} indicators)")

print("\n================ ADDITIVE TOBIT per antibody (fit on train) ================")
print(f"{'antibody':12s} {'n':>8s} {'%cens':>6s} {'sigma':>7s} {'conv':>5s} | {'interior R2':>11s} {'interior rho':>12s} | {'val rho':>8s}")
print("-"*108)
out={}
for ab in ABS:
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]
    X=Xtr[d.rbd_sequence.map(itr).to_numpy()]; y=d.affinity.to_numpy()
    r=tobit_fit(X,y,lower=lo,upper=hi)
    out[ab]=r
    mu=X@r["beta"]
    un=(y>lo)&(y<hi)
    r2=1-((y[un]-mu[un])**2).sum()/((y[un]-y[un].mean())**2).sum()
    rho=stats.spearmanr(mu[un],y[un]).statistic
    dv=va[va.antibody_name==ab]
    Xv=Xva[dv.rbd_sequence.map(iva).to_numpy()]; yv=dv.affinity.to_numpy()
    muv=Xv@r["beta"]; unv=(yv>lo)&(yv<hi)
    rhov=stats.spearmanr(muv[unv],yv[unv]).statistic
    print(f"{ab:12s} {r['n']:8d} {100*r['n_cens']/r['n']:5.1f}% {r['sigma']:7.4f} {str(r['converged']):>5s} | "
          f"{r2:11.4f} {rho:12.4f} | {rhov:8.4f}")
pickle.dump(out,open(f"{SP}/additive.pkl","wb"))

print("\n(interior R2/rho = latent-scale fit judged on uncensored rows only;")
print(" val rho = same, on the held-out validation genotypes — no tuning done on it)")

print("\n\n================ Largest additive effects (sign = direction of the latent response) ================")
for ab in ABS:
    r=out[ab]; lo,hi=cen[ab]
    b=r["beta"][1:]; se=r["se"][1:]
    o=np.argsort(-np.abs(b))[:8]
    bnd = "lower (0)" if np.isfinite(lo) else "upper (1)"
    toward = "toward bound" if np.isfinite(lo) else "toward bound"
    print(f"\n{ab}   censored at {bnd};  'toward bound' = {'negative' if np.isfinite(lo) else 'positive'} coefficient")
    print(f"  {'allele':14s} {'coef':>9s} {'SE':>8s} {'z':>8s}  dir")
    for j in o:
        pushes = (b[j]<0) if np.isfinite(lo) else (b[j]>0)
        print(f"  {names[j]:14s} {b[j]:9.4f} {se[j]:8.4f} {b[j]/se[j]:8.2f}  {'TOWARD bound' if pushes else 'away'}")
