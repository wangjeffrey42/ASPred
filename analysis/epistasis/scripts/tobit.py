import numpy as np
from scipy import optimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

LOG2PI = np.log(2*np.pi)

def _ratio(w):
    """phi(w)/Phi(w), stable."""
    return np.exp(norm.logpdf(w) - log_ndtr(w))

def tobit_fit(X, y, lower=-np.inf, upper=np.inf, beta0=None, tol=1e-9, maxiter=800):
    """
    Censored-normal (Tobit) MLE.  Latent y* = Xb + e, e~N(0,s^2);
    observed y = clip(y*, lower, upper).  One-sided or two-sided.
    Returns dict with beta, sigma, se (sandwich-free observed-information SE), loglik, converged.
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    cl = np.isfinite(lower) & (y <= lower + 0)   # at lower bound
    cu = np.isfinite(upper) & (y >= upper - 0)
    cl = cl if np.isfinite(lower) else np.zeros(n, bool)
    cu = cu if np.isfinite(upper) else np.zeros(n, bool)
    un = ~(cl | cu)

    Xu, yu = X[un], y[un]
    Xl = X[cl]; Xh = X[cu]

    if beta0 is None:
        beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    s0 = max(y.std(), 1e-3)
    p0 = np.concatenate([beta0, [np.log(s0)]])

    def nll_grad(p):
        b = p[:k]; ls = p[k]; s = np.exp(ls)
        g = np.zeros(k+1); ll = 0.0
        # uncensored
        if Xu.shape[0]:
            z = (yu - Xu@b)/s
            ll += (-ls - 0.5*z**2 - 0.5*LOG2PI).sum()
            g[:k] += Xu.T @ (z/s)
            g[k]  += (-1 + z**2).sum()
        # left censored
        if Xl.shape[0]:
            w = (lower - Xl@b)/s
            ll += log_ndtr(w).sum()
            r = _ratio(w)
            g[:k] += Xl.T @ (-r/s)
            g[k]  += (-r*w).sum()
        # right censored
        if Xh.shape[0]:
            w = (Xh@b - upper)/s
            ll += log_ndtr(w).sum()
            r = _ratio(w)
            g[:k] += Xh.T @ (r/s)
            g[k]  += (-r*w).sum()
        return -ll, -g

    res = optimize.minimize(nll_grad, p0, jac=True, method="L-BFGS-B",
                            options={"maxiter":maxiter,"ftol":tol,"gtol":1e-8})
    b = res.x[:k]; s = np.exp(res.x[k])

    # observed information by numeric Hessian of the NLL in (beta, log sigma)
    def nll(p): return nll_grad(p)[0]
    H = _num_hess(nll_grad, res.x)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov)[:k], 0, None))
    except np.linalg.LinAlgError:
        cov = None; se = np.full(k, np.nan)
    return {"beta":b,"sigma":s,"se":se,"cov":cov,"loglik":-res.fun,
            "converged":bool(res.success),"n":n,"n_cens":int(cl.sum()+cu.sum()),
            "n_interior":int(un.sum()),"msg":res.message}

def _num_hess(fg, p, eps=1e-5):
    """Hessian from analytic gradients (central differences)."""
    m = len(p); H = np.zeros((m,m))
    for j in range(m):
        d = np.zeros(m); d[j] = eps*max(1.0, abs(p[j]))
        _, gp = fg(p+d); _, gm = fg(p-d)
        H[:,j] = (gp-gm)/(2*d[j])
    return 0.5*(H+H.T)
