import numpy as np, pandas as pd, pickle, warnings, itertools
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
_os.makedirs(SP,exist_ok=True)

warnings.filterwarnings("ignore")

F=pickle.load(open(f"{SP}/feat.pkl","rb")); cen=pickle.load(open(f"{SP}/cen.pkl","rb"))
D=pickle.load(open(f"{SP}/scan3.pkl","rb")); SEL=pickle.load(open(f"{SP}/selection.pkl","rb"))
names,ABS=D["names"],F["ABS"]; res=D["res"]; cm=D["cellmap"]; tw=D["toward"]
DROP=["S309","AZD8895"]
def nm(i,j): return f"{names[i]} x {names[j]}"

print("="*104)
print("PAIRWISE EPISTASIS SCAN — per-group results")
print("="*104)
print("Model: Tobit (censoring-aware) on all rows; 'interior' variant = Gaussian on uncensored rows only.")
print("Screen and confirm use disjoint genotype halves, so p-values are not selection-contaminated.")
print("q = Benjamini-Hochberg FDR across the 25 confirmed terms within a group.")
print("ARTIFACT flag = both constituent alleles have additive effects pushing toward that group's bound.")

for ab in ABS:
    r=res[("tobit",ab)]
    o=np.argsort(-np.abs(r["b"]))[:20]
    lo,hi=cen[ab]
    print(f"\n{'-'*104}\n{ab}   censored at {'upper' if np.isfinite(hi) else 'lower'} bound "
          f"({SEL['sev'][ab]['cens']:.1f}% of rows)   confirm n={r['n_conf']}")
    print(f"  {'interaction':30s} {'coef':>9s} {'SE':>8s} {'z':>7s} {'p':>10s} {'q':>9s} | "
          f"{'n00':>7s} {'n10':>7s} {'n01':>7s} {'n11':>7s} | flag")
    for t in o:
        i,j=r["top"][t]; c=cm[(i,j)]
        art = tw[ab][i] and tw[ab][j]
        sig = "*" if r["q"][t]<0.05 else " "
        print(f"  {nm(i,j):30s} {r['b'][t]:9.4f} {r['se'][t]:8.4f} {r['z'][t]:7.2f} "
              f"{r['p'][t]:10.2e} {r['q'][t]:8.3f}{sig} | {c[0]:7d} {c[1]:7d} {c[2]:7d} {c[3]:7d} | "
              f"{'ARTIFACT-RISK' if art else ''}")
    nsig=int((r["q"]<0.05).sum()); nart=sum(1 for t in range(25) for i,j in [r["top"][t]] if tw[ab][i] and tw[ab][j])
    print(f"  -> {nsig}/25 terms pass FDR 0.05;  {nart}/25 carry the both-push-toward-bound artifact risk")

print("\n\n"+"="*104)
print("OVERLAP: all-rows (Tobit) vs interior-only top sets")
print("="*104)
print("Dropped from this comparison: "+", ".join(DROP))
print("Reason: for these groups the main effect of the top alleles IS pushing the response past the")
print("bound, so conditioning on interior rows selects on the outcome and removes the effect being")
print("estimated.  The interior-only control is unavailable precisely where censoring is strongest.")
print(f"\n  {'group':12s} {'%cens':>6s} {'median attenuation':>19s} {'max allele depletion':>21s} {'status':>12s}")
for ab in ABS:
    s=SEL["sev"][ab]
    print(f"  {ab:12s} {s['cens']:5.1f}% {SEL['att'][ab]:19.2f} {s['maxdep']:20.3f} ({s['worst']}) "
          f"{'DROPPED' if ab in DROP else 'kept':>12s}")
print(f"\n  {'group':12s} {'|top25 both|':>12s} {'overlap':>8s} {'sig-in-both (q<.05)':>20s} {'sign agree':>11s}")
kept=[a for a in ABS if a not in DROP]
for ab in kept:
    A=res[("tobit",ab)]; B=res[("interior",ab)]
    sa={tuple(p) for p in A["top"]}; sb={tuple(p) for p in B["top"]}
    inter=sa&sb
    qa={tuple(p):(A["q"][k],A["b"][k]) for k,p in enumerate(A["top"])}
    qb={tuple(p):(B["q"][k],B["b"][k]) for k,p in enumerate(B["top"])}
    both=[t for t in inter if qa[t][0]<0.05 and qb[t][0]<0.05]
    agree=sum(1 for t in inter if np.sign(qa[t][1])==np.sign(qb[t][1]))
    print(f"  {ab:12s} {len(inter):12d} {len(inter)/25:8.2f} {len(both):20d} "
          f"{(agree/len(inter) if inter else float('nan')):11.2f}")

print("\n\n"+"="*104)
print("OVERLAP ACROSS GROUPS (all-rows Tobit top-25 sets)")
print("="*104)
print(f"  {'':12s}"+"".join(f"{a[:10]:>12s}" for a in ABS))
for a in ABS:
    sa={tuple(p) for p in res[("tobit",a)]["top"]}
    row=f"  {a:12s}"
    for b in ABS:
        sb={tuple(p) for p in res[("tobit",b)]["top"]}
        row+=f"{(len(sa&sb) if a!=b else 25):12d}"
    print(row)
cnt={}
for a in ABS:
    for p in res[("tobit",a)]["top"]: cnt[tuple(p)]=cnt.get(tuple(p),0)+1
sh=sorted(cnt.items(),key=lambda kv:-kv[1])[:12]
print("\n  terms recurring across groups:")
for (i,j),c in sh:
    if c<2: continue
    gs=[a for a in ABS if (i,j) in {tuple(p) for p in res[("tobit",a)]["top"]}]
    sig=[a for a in gs if res[("tobit",a)]["q"][[tuple(p) for p in res[("tobit",a)]["top"]].index((i,j))]<0.05]
    print(f"    {nm(i,j):30s} in {c} groups ({', '.join(g[:10] for g in gs)});  FDR-sig in {len(sig)}")
