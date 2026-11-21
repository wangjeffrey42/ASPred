"""Main-effect attenuation vs interaction attenuation, side by side."""
import os as _os, pickle, warnings
warnings.filterwarnings("ignore")
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache"); RES=_os.path.join(_os.path.dirname(_HERE),"results")
import numpy as np, pandas as pd
from scipy import stats as sst
F=pickle.load(open(f"{SP}/feat.pkl","rb")); D=pickle.load(open(f"{SP}/scan3.pkl","rb"))
ad=pickle.load(open(f"{SP}/additive.pkl","rb"))
names,ABS=D["names"],F["ABS"]; res=D["res"]; reps=D["reps"]
MM=pd.read_csv(f"{RES}/model_implied_main_effects.csv")
MI=pd.read_csv(f"{RES}/model_implied_interactions.csv")

def stats(x,y):
    """x = fitted coefficient, y = model-implied.  slope regresses model on fitted."""
    if len(x)<3: return dict(n=len(x),pearson=np.nan,spearman=np.nan,slope=np.nan,p_pearson=np.nan)
    pr=sst.pearsonr(x,y); sr=sst.spearmanr(x,y)
    sl=np.polyfit(x,y,1)[0]
    return dict(n=len(x),pearson=pr.statistic,p_pearson=pr.pvalue,spearman=sr.statistic,
                p_spearman=sr.pvalue,slope=sl)

out=[]
for ab in ABS:
    # --- main effects: all 43 representatives, and the z>3 subset ---
    b=ad[ab]["beta"][1:]; se=ad[ab]["se"][1:]
    fit=pd.DataFrame(dict(feature=[names[i] for i in reps],coef=b[reps],se=se[reps]))
    fit["z"]=fit.coef/fit.se
    m=fit.merge(MM[MM.group==ab],on="feature")
    sa=stats(m.coef.to_numpy(),m.model_main.to_numpy())
    ms=m[m.z.abs()>3]
    ss=stats(ms.coef.to_numpy(),ms.model_main.to_numpy())
    # --- interactions: all 25 confirmed, and the FDR-significant subset ---
    r=res[("tobit",ab)]
    ifit=pd.DataFrame(dict(term=[f"{names[i]} x {names[j]}" for i,j in r["top"]],
                           coef=r["b"],q=r["q"]))
    mi=ifit.merge(MI[MI.group==ab],on="term")
    ia=stats(mi.coef.to_numpy(),mi.model_interaction.to_numpy())
    isg=mi[mi.q<0.05]
    isx=stats(isg.coef.to_numpy(),isg.model_interaction.to_numpy())
    out.append(dict(group=ab,
        main_n=sa["n"],main_pearson=sa["pearson"],main_spearman=sa["spearman"],main_slope=sa["slope"],
        main_sig_n=ss["n"],main_sig_slope=ss["slope"],main_sig_pearson=ss["pearson"],
        int_n=ia["n"],int_pearson=ia["pearson"],int_spearman=ia["spearman"],int_slope=ia["slope"],
        int_sig_n=isx["n"],int_sig_slope=isx["slope"],int_sig_pearson=isx["pearson"],
        rms_fitted_main=np.sqrt((m.coef**2).mean()),rms_model_main=np.sqrt((m.model_main**2).mean()),
        rms_fitted_int=np.sqrt((mi.coef**2).mean()),rms_model_int=np.sqrt((mi.model_interaction**2).mean())))
O=pd.DataFrame(out); O.to_csv(f"{RES}/main_vs_interaction_slope.csv",index=False)

print("="*118)
print("IS THE MAGNITUDE ATTENUATION SPECIFIC TO INTERACTIONS?")
print("="*118)
print("slope = regression of model-implied value on fitted Tobit coefficient.")
print("slope 1 = model reproduces the effect at full size;  slope 0 = model shows none of it.\n")
print(f"{'group':12s} | {'--------- MAIN EFFECTS (43 reps) ---------':^43s} | {'------ INTERACTIONS (25 terms) ------':^39s}")
print(f"{'':12s} | {'Pearson':>8s} {'Spearman':>9s} {'slope':>8s} {'slope(|z|>3)':>13s} | {'Pearson':>8s} {'Spearman':>9s} {'slope':>8s} {'slope(sig)':>10s}")
print("-"*118)
for r in out:
    print(f"{r['group']:12s} | {r['main_pearson']:8.3f} {r['main_spearman']:9.3f} {r['main_slope']:8.3f} "
          f"{r['main_sig_slope']:13.3f} | {r['int_pearson']:8.3f} {r['int_spearman']:9.3f} "
          f"{r['int_slope']:8.3f} {r['int_sig_slope']:10.3f}")
print("-"*118)
print(f"{'median':12s} | {O.main_pearson.median():8.3f} {O.main_spearman.median():9.3f} "
      f"{O.main_slope.median():8.3f} {O.main_sig_slope.median():13.3f} | {O.int_pearson.median():8.3f} "
      f"{O.int_spearman.median():9.3f} {O.int_slope.median():8.3f} {O.int_sig_slope.median():10.3f}")

print("\n\n"+"="*118); print("RMS MAGNITUDE RATIO (model-implied / fitted)"); print("="*118)
print(f"{'group':12s} {'main: fitted':>13s} {'model':>9s} {'ratio':>8s}  |  "
      f"{'int: fitted':>12s} {'model':>9s} {'ratio':>8s}  |  {'ratio of ratios':>16s}")
for r in out:
    rm=r['rms_model_main']/r['rms_fitted_main']; ri=r['rms_model_int']/r['rms_fitted_int']
    print(f"{r['group']:12s} {r['rms_fitted_main']:13.4f} {r['rms_model_main']:9.4f} {rm:8.3f}  |  "
          f"{r['rms_fitted_int']:12.4f} {r['rms_model_int']:9.4f} {ri:8.3f}  |  {rm/ri:16.2f}")
mr=(O.rms_model_main/O.rms_fitted_main); ir=(O.rms_model_int/O.rms_fitted_int)
print(f"\nmedian main ratio {mr.median():.3f}   median interaction ratio {ir.median():.3f}   "
      f"main/interaction {mr.median()/ir.median():.2f}x")
