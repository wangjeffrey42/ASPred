import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings("ignore")
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache")
RES=_os.path.join(_os.path.dirname(_HERE),"results")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
_os.makedirs(RES,exist_ok=True)
F=pickle.load(open(f"{SP}/feat.pkl","rb")); cen=pickle.load(open(f"{SP}/cen.pkl","rb"))
D=pickle.load(open(f"{SP}/scan3.pkl","rb")); SEL=pickle.load(open(f"{SP}/selection.pkl","rb"))
ad=pickle.load(open(f"{SP}/additive.pkl","rb"))
names,ABS=D["names"],F["ABS"]; res=D["res"]; cm=D["cellmap"]; tw=D["toward"]; reps=D["reps"]

rows=[]
for (variant,ab),r in res.items():
    for k,(i,j) in enumerate(r["top"]):
        c=cm[(i,j)]
        rows.append(dict(group=ab,variant=variant,rank_by_screen=k+1,
            feat_i=names[i],feat_j=names[j],term=f"{names[i]} x {names[j]}",
            screen_chi2=r["screen_chi2"][k],coef=r["b"][k],se=r["se"][k],z=r["z"][k],
            p=r["p"][k],q=r["q"][k],fdr_sig=bool(r["q"][k]<0.05),
            n00=c[0],n10=c[1],n01=c[2],n11=c[3],min_cell=min(c),
            i_pushes_toward_bound=bool(tw[ab][i]),j_pushes_toward_bound=bool(tw[ab][j]),
            artifact_risk=bool(tw[ab][i] and tw[ab][j]),
            n_confirm=r["n_conf"],n_screen=r["n_screen"],cond=r["cond"]))
T=pd.DataFrame(rows).sort_values(["variant","group","rank_by_screen"])
T.to_csv(f"{RES}/interaction_terms_full.csv",index=False)
print("interaction_terms_full.csv:",T.shape)

# per-group summary
s=[]
for ab in ABS:
    r=res[("tobit",ab)]; sv=SEL["sev"][ab]
    s.append(dict(group=ab,pct_censored=sv["cens"],n_interior=sv["nint"],
        bound="upper" if np.isfinite(cen[ab][1]) else "lower",
        n_fdr_sig=int((r["q"]<0.05).sum()),
        n_artifact_risk=sum(1 for i,j in r["top"] if tw[ab][i] and tw[ab][j]),
        median_attenuation=SEL["att"][ab],max_allele_depletion=sv["maxdep"],
        most_depleted_allele=sv["worst"],
        corr_effect_depletion=sv["rho"],
        interior_usable=ab not in ("S309","AZD8895")))
pd.DataFrame(s).to_csv(f"{RES}/group_summary.csv",index=False)
print("group_summary.csv written")

# alias groups
pd.DataFrame([dict(representative=k,aliases=";".join(v)) for k,v in D["alias_of"].items()]
    ).to_csv(f"{RES}/alias_groups.csv",index=False)
# additive effects
arow=[]
for ab in ABS:
    b=ad[ab]["beta"][1:]; se=ad[ab]["se"][1:]
    for j in reps:
        arow.append(dict(group=ab,allele=names[j],coef=b[j],se=se[j],z=b[j]/se[j],
                         pushes_toward_bound=bool(tw[ab][j])))
pd.DataFrame(arow).to_csv(f"{RES}/additive_effects.csv",index=False)
print("alias_groups.csv, additive_effects.csv written")
