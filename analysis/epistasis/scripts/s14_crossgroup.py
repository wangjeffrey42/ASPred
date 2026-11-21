import numpy as np, pandas as pd, pickle, warnings, itertools
warnings.filterwarnings("ignore")
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache"); RES=_os.path.join(_os.path.dirname(_HERE),"results")
F=pickle.load(open(f"{SP}/feat.pkl","rb")); D=pickle.load(open(f"{SP}/scan3.pkl","rb"))
ad=pickle.load(open(f"{SP}/additive.pkl","rb")); cen=pickle.load(open(f"{SP}/cen.pkl","rb"))
names,ABS=D["names"],F["ABS"]; res=D["res"]; pairs=D["pairs"]; reps=D["reps"]; K=D["K"]
from scipy import stats as sst
rng=np.random.default_rng(7)
NPERM=20000

top={ab:[tuple(p) for p in res[("tobit",ab)]["top"]] for ab in ABS}
coef={ab:dict(zip(top[ab],res[("tobit",ab)]["b"])) for ab in ABS}
qv={ab:dict(zip(top[ab],res[("tobit",ab)]["q"])) for ab in ABS}

print("="*100); print("CROSS-GROUP SHARING OF TOP-25 INTERACTION TERMS"); print("="*100)
print(f"Candidate space: {len(pairs)} estimable pairs.  Uniform null E[overlap] = 25*25/{len(pairs)} = {625/len(pairs):.2f}\n")

# ---- null 1: uniform random draws of 25 from the candidate space -------
# ---- null 2: degree-preserving -- resample each group's 25 terms from pairs
#              weighted to preserve how often each POSITION appears in that
#              group's top set (so sharing driven by shared hot positions is nulled out)
def pos_of(i): return F["feat"][i][0]
P_OF={i:pos_of(i) for i in reps}

def weighted_draw(ab,n=K):
    """draw K pairs matching the group's position-usage profile"""
    cnt={}
    for i,j in top[ab]:
        cnt[P_OF[i]]=cnt.get(P_OF[i],0)+1; cnt[P_OF[j]]=cnt.get(P_OF[j],0)+1
    w=np.array([cnt.get(P_OF[i],0.0)+cnt.get(P_OF[j],0.0)+1e-9 for i,j in pairs])
    w=w/w.sum()
    idx=rng.choice(len(pairs),size=n,replace=False,p=w)
    return {pairs[t] for t in idx}

rows=[]
print(f"{'group A':12s} {'group B':12s} {'overlap':>7s} {'sign agree':>10s} | {'uniform null':>12s} {'p':>8s} | {'pos-matched null':>16s} {'p':>8s}")
print("-"*105)
for a,b in itertools.combinations(ABS,2):
    sa,sb=set(top[a]),set(top[b]); inter=sa&sb
    ov=len(inter)
    agree=sum(1 for t in inter if np.sign(coef[a][t])==np.sign(coef[b][t]))
    # uniform null
    nu=np.array([len(set(map(tuple,[pairs[t] for t in rng.choice(len(pairs),K,replace=False)]))&
                     set(map(tuple,[pairs[t] for t in rng.choice(len(pairs),K,replace=False)])))
                 for _ in range(2000)])
    pu=(nu>=ov).mean()
    # position-matched null
    npm=np.array([len(weighted_draw(a)&weighted_draw(b)) for _ in range(2000)])
    ppm=(npm>=ov).mean()
    rows.append(dict(group_a=a,group_b=b,overlap=ov,
        sign_agree=agree,sign_agree_frac=agree/ov if ov else np.nan,
        uniform_null_mean=nu.mean(),p_uniform=pu,
        posmatched_null_mean=npm.mean(),p_posmatched=ppm))
    print(f"{a:12s} {b:12s} {ov:7d} {agree:4d}/{ov:<5d} | {nu.mean():12.2f} {pu:8.4f} | {npm.mean():16.2f} {ppm:8.4f}")
X=pd.DataFrame(rows); X.to_csv(f"{RES}/crossgroup_overlap.csv",index=False)

# ---- does sharing track shared strong-main-effect positions? -----------
print("\n"+"="*100); print("DOES SHARING TRACK EACH GROUP'S STRONGEST MAIN-EFFECT POSITIONS?"); print("="*100)
# per group: rank positions by max |additive coef| among its alleles
strength={}
for ab in ABS:
    bb=np.abs(ad[ab]["beta"][1:])
    d={}
    for i in reps: d[P_OF[i]]=max(d.get(P_OF[i],0.0),bb[i])
    strength[ab]=d
POS=sorted({P_OF[i] for i in reps})
M=np.array([[strength[ab].get(p,0.0) for p in POS] for ab in ABS])
Mn=M/np.linalg.norm(M,axis=1,keepdims=True)

print("\nTop-5 main-effect positions per group (position numbering is 1-based RBD):")
for k,ab in enumerate(ABS):
    o=np.argsort(-M[k])[:5]
    print(f"  {ab:12s} "+", ".join(f"{POS[t]+331}({M[k][t]:.2f})" for t in o))

sim=[];ovs=[];lab=[]
for a,b in itertools.combinations(ABS,2):
    ia,ib=ABS.index(a),ABS.index(b)
    sim.append(float(Mn[ia]@Mn[ib])); ovs.append(len(set(top[a])&set(top[b]))); lab.append(f"{a[:9]}/{b[:9]}")
sim=np.array(sim);ovs=np.array(ovs)
pr=sst.pearsonr(sim,ovs); sr=sst.spearmanr(sim,ovs)
print(f"\nmain-effect profile similarity (cosine) vs top-25 overlap, across the {len(sim)} group pairs:")
print(f"  Pearson  r = {pr.statistic:+.3f}  (p = {pr.pvalue:.4f})")
print(f"  Spearman rho = {sr.statistic:+.3f}  (p = {sr.pvalue:.4f})")
print(f"\n  {'group pair':22s} {'profile cos':>11s} {'overlap':>8s}")
for t in np.argsort(-ovs):
    print(f"  {lab[t]:22s} {sim[t]:11.3f} {ovs[t]:8d}")
pd.DataFrame(dict(group_pair=lab,maineffect_cosine=sim,overlap=ovs)).to_csv(
    f"{RES}/crossgroup_maineffect_similarity.csv",index=False)

# ---- positional composition of shared vs unshared terms ---------------
print("\n"+"="*100); print("ARE SHARED TERMS BUILT FROM EACH GROUP'S STRONGEST POSITIONS?"); print("="*100)
out=[]
for ab in ABS:
    others=set().union(*[set(top[o]) for o in ABS if o!=ab])
    sh=[t for t in top[ab] if t in others]; un=[t for t in top[ab] if t not in others]
    def meanrank(ts):
        if not ts: return np.nan
        r=[]
        srt=sorted(strength[ab],key=lambda p:-strength[ab][p]); rk={p:k+1 for k,p in enumerate(srt)}
        for i,j in ts: r.append((rk[P_OF[i]]+rk[P_OF[j]])/2)
        return float(np.mean(r))
    out.append(dict(group=ab,n_shared=len(sh),n_unique=len(un),
        mean_maineffect_rank_shared=meanrank(sh),mean_maineffect_rank_unique=meanrank(un)))
    print(f"  {ab:12s} shared {len(sh):2d} (mean main-effect rank of its positions {meanrank(sh):5.1f})   "
          f"unique {len(un):2d} (rank {meanrank(un):5.1f})")
pd.DataFrame(out).to_csv(f"{RES}/crossgroup_composition.csv",index=False)
print("\n(lower rank = the group's stronger main-effect positions; 1 = strongest of ~21 positions)")
