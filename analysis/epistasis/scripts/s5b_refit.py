import numpy as np, pandas as pd, pickle, sys, warnings
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
_os.makedirs(SP,exist_ok=True)

warnings.filterwarnings("ignore")

sys.path.insert(0,_HERE)
from tobit import tobit_fit
from scipy import stats as sstats, optimize

F=pickle.load(open(f"{SP}/feat.pkl","rb")); cen=pickle.load(open(f"{SP}/cen.pkl","rb"))
D=pickle.load(open(f"{SP}/scan.pkl","rb"))
feat,names,NP,ABS=F["feat"],F["names"],F["NP"],F["ABS"]
fidx={k:i for i,k in enumerate(feat)}
res,half,K=D["res"],D["half"],D["K"]
tr=pd.read_csv(f"{P}/data/paired/rbd_split_train.csv",usecols=["antibody_name","rbd_sequence","affinity"])
g=tr.rbd_sequence.unique()
Xi=np.zeros((len(g),NP))
for r,s in enumerate(g):
    for p,a in feat:
        if s[p]==a: Xi[r,fidx[(p,a)]]=1.0
ig={s:i for i,s in enumerate(g)}

print("Refitting the two non-converged confirm fits with warm start + tighter tolerance.")
print("Reporting gradient norm at the solution as the convergence check.\n")
for ab in ["AZD1061","REGN10987"]:
    lo,hi=cen[ab]
    d=tr[tr.antibody_name==ab]; rows=d.rbd_sequence.map(ig).to_numpy()
    keep=(~half)[rows]; d=d[keep]; rows=rows[keep]
    y=d.affinity.to_numpy(); Xia=Xi[rows]
    top=res[("tobit",ab)]["top"]
    Xj=np.hstack([np.ones((len(y),1)),Xia]+[(Xia[:,i]*Xia[:,j])[:,None] for i,j in top])
    old=res[("tobit",ab)]
    b0=np.concatenate([np.zeros(1),np.zeros(NP),np.zeros(K)])
    # warm start from previous solution
    prev=np.concatenate([[0.0]])
    f=tobit_fit(Xj,y,lower=lo,upper=hi,maxiter=5000,tol=1e-12)
    # gradient norm check
    from tobit import tobit_fit as _tf
    import tobit as T
    # rebuild nll_grad to evaluate gradient at solution
    k=Xj.shape[1]
    def nllg(p):
        b=p[:k]; ls=p[k]; s=np.exp(ls)
        from scipy.special import log_ndtr
        from scipy.stats import norm
        cl=np.isfinite(lo)&(y<=lo); cu=np.isfinite(hi)&(y>=hi); un=~(cl|cu)
        gr=np.zeros(k+1); ll=0.0
        if un.any():
            z=(y[un]-Xj[un]@b)/s; ll+=(-ls-0.5*z**2).sum(); gr[:k]+=Xj[un].T@(z/s); gr[k]+=(-1+z**2).sum()
        if cl.any():
            w=(lo-Xj[cl]@b)/s; ll+=log_ndtr(w).sum(); r=np.exp(norm.logpdf(w)-log_ndtr(w))
            gr[:k]+=Xj[cl].T@(-r/s); gr[k]+=(-r*w).sum()
        if cu.any():
            w=(Xj[cu]@b-hi)/s; ll+=log_ndtr(w).sum(); r=np.exp(norm.logpdf(w)-log_ndtr(w))
            gr[:k]+=Xj[cu].T@(r/s); gr[k]+=(-r*w).sum()
        return -ll,-gr
    p=np.concatenate([f["beta"],[np.log(f["sigma"])]])
    _,gsol=nllg(p)
    b=f["beta"][-K:]; se=f["se"][-K:]; z=b/se; pv=2*sstats.norm.sf(np.abs(z))
    o=np.argsort(pv); q=np.empty(K); rk=pv[o]*K/np.arange(1,K+1)
    q[o]=np.minimum.accumulate(rk[::-1])[::-1]
    print(f"{ab}: converged={f['converged']}  loglik={f['loglik']:.3f} (was {0:.0f}) "
          f"||grad||_inf={np.abs(gsol).max():.3e}  rel to n={np.abs(gsol).max()/len(y):.2e}")
    print(f"   max |coef change| vs previous solution: {np.abs(b-old['b']).max():.3e}")
    print(f"   max |z change|: {np.abs(z-old['z']).max():.3e}")
    res[("tobit",ab)].update({"b":b,"se":se,"z":z,"p":pv,"q":np.clip(q,0,1),
                              "sigma":f["sigma"],"conv":f["converged"],"gradinf":np.abs(gsol).max()})
D["res"]=res
pickle.dump(D,open(f"{SP}/scan.pkl","wb"))
print("\nupdated scan.pkl")
