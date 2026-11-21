import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
_os.makedirs(SP,exist_ok=True)


def gen_resid(X, y, beta, sigma, lower, upper):
    """Tobit generalized residual u_i = d loglik_i / d(eta_i), and v_i = d loglik_i / d log sigma."""
    eta=X@beta; s=sigma
    u=np.zeros(len(y)); v=np.zeros(len(y))
    cl=np.isfinite(lower)&(y<=lower); cu=np.isfinite(upper)&(y>=upper)
    un=~(cl|cu)
    z=(y[un]-eta[un])/s; u[un]=z/s; v[un]=-1+z**2
    if cl.any():
        w=(lower-eta[cl])/s; r=np.exp(norm.logpdf(w)-log_ndtr(w))
        u[cl]=-r/s; v[cl]=-r*w
    if cu.any():
        w=(eta[cu]-upper)/s; r=np.exp(norm.logpdf(w)-log_ndtr(w))
        u[cu]=r/s; v[cu]=-r*w
    return u,v

def score_screen(X, y, beta, sigma, lower, upper, pairs, Xi, chunk=120):
    """
    Rao score test for adding each interaction Xi[:,i]*Xi[:,j] to the additive Tobit,
    with OPG (BHHH) information.  Returns chi2_1 statistics and signed score.
    X: additive design incl intercept (n x k).  Xi: indicator matrix (n x NP).
    """
    u,v=gen_resid(X,y,beta,sigma,lower,upper)
    A=np.hstack([u[:,None]*X, v[:,None]])          # n x (k+1) score wrt estimated params
    Jtt=A.T@A
    Jtt_inv=np.linalg.pinv(Jtt)
    stats=np.empty(len(pairs)); scores=np.empty(len(pairs))
    for st in range(0,len(pairs),chunk):
        pr=pairs[st:st+chunk]
        Cd=np.empty((len(y),len(pr)))
        for m,(i,j) in enumerate(pr): Cd[:,m]=Xi[:,i]*Xi[:,j]
        B=u[:,None]*Cd
        S=B.sum(0)
        Jcc=(B*B).sum(0)
        Jct=B.T@A
        var=Jcc-np.einsum("ck,kl,cl->c",Jct,Jtt_inv,Jct)
        var=np.clip(var,1e-300,None)
        stats[st:st+chunk]=S**2/var
        scores[st:st+chunk]=S/np.sqrt(var)
    return stats,scores
