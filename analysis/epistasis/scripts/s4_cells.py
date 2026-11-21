import pandas as pd, numpy as np, pickle, sys, warnings
import os as _os
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
_os.makedirs(SP,exist_ok=True)

warnings.filterwarnings("ignore")

sys.path.insert(0,_HERE)

F=pickle.load(open(f"{SP}/feat.pkl","rb")); cen=pickle.load(open(f"{SP}/cen.pkl","rb"))
feat,names,NP,ABS=F["feat"],F["names"],F["NP"],F["ABS"]
fidx={k:i for i,k in enumerate(feat)}
tr=pd.read_csv(f"{P}/data/paired/rbd_split_train.csv",usecols=["antibody_name","rbd_sequence","affinity"])
g=tr.rbd_sequence.unique()
X=np.zeros((len(g),NP),dtype=np.int8)
for r,s in enumerate(g):
    for p,a in feat:
        if s[p]==a: X[r,fidx[(p,a)]]=1
Xf=X.astype(np.float64)

# ---- aliasing groups: |r| > 0.99 -------------------------------------
C=np.corrcoef(Xf.T); np.fill_diagonal(C,0)
parent=list(range(NP))
def find(a):
    while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
    return a
for i in range(NP):
    for j in range(i+1,NP):
        if abs(C[i,j])>0.99: parent[find(i)]=find(j)
groups={}
for i in range(NP): groups.setdefault(find(i),[]).append(i)
print("================ ALIASED INDICATOR GROUPS (|r| > 0.99) ================")
multi=[v for v in groups.values() if len(v)>1]
if not multi: print("none")
for v in multi:
    print("  {"+", ".join(names[i] for i in v)+"}  -- cannot be separated; "
          f"max |r| = {max(abs(C[i,j]) for i in v for j in v if i!=j):.4f}")
print(f"\n{NP} indicators collapse to {len(groups)} estimable groups.")
print("Within a group, individual coefficients are not identified; an interaction involving")
print("two members of the SAME group is not estimable at all.")

# ---- four-state cell counts for every pair ---------------------------
print("\n\n================ PAIR CELL COUNTS (genotype level, n=%d) ================"%len(g))
iu=np.triu_indices(NP,1)
n11=(Xf.T@Xf)
n1=Xf.sum(0)
rows=[]
for i,j in zip(*iu):
    a=n11[i,j]; b=n1[i]-a; c=n1[j]-a; d=len(g)-a-b-c
    rows.append((i,j,int(d),int(b),int(c),int(a)))
cells=pd.DataFrame(rows,columns=["i","j","n00","n10","n01","n11"])
cells["mincell"]=cells[["n00","n10","n01","n11"]].min(1)
cells["same_group"]=[find(i)==find(j) for i,j in zip(cells.i,cells.j)]
print(f"total pairs: {len(cells)}")
for th in [1,10,50,100,500,1000]:
    print(f"  pairs with all four states >= {th:5d}: {(cells.mincell>=th).sum():5d}"
          f"   (excluding aliased: {((cells.mincell>=th)&~cells.same_group).sum():5d})")
print(f"  pairs with an EMPTY state (mincell=0): {(cells.mincell==0).sum()}")
print(f"  pairs aliased (same group): {cells.same_group.sum()}")
EST=cells[(cells.mincell>=100)&(~cells.same_group)].copy()
print(f"\n=> estimable pair set used downstream: {len(EST)} pairs (all four states >=100, not aliased)")
cells.to_pickle(f"{SP}/cells.pkl")
pickle.dump({"groups":groups,"find_parent":parent},open(f"{SP}/alias.pkl","wb"))

# ---- interior counts per antibody, which drives power ----------------
print("\n\n================ INTERIOR ROWS AVAILABLE PER ANTIBODY ================")
print("(interaction terms are identified mainly by interior rows; censored rows carry only 'past the bound')")
print(f"{'antibody':12s} {'n interior':>11s} {'rows/param (46+1035)':>22s} {'rows/param (46 only)':>21s}")
for ab in ABS:
    lo,hi=cen[ab]; y=tr.loc[tr.antibody_name==ab,"affinity"].to_numpy()
    ni=((y>lo)&(y<hi)).sum()
    print(f"{ab:12s} {ni:11d} {ni/(47+len(EST)):22.1f} {ni/47:21.1f}")

# ---- three-way feasibility ------------------------------------------
print("\n\n================ THREE-WAY SCAN FEASIBILITY ================")
ng=len(groups)
import itertools, math
print(f"non-aliased indicator groups: {ng}  ->  three-way terms: {math.comb(ng,3)}")
# sample triples and count min cell of 8 states
rng=np.random.default_rng(0)
reps=[g_[0] for g_ in groups.values()]
trip=list(itertools.combinations(range(len(reps)),3))
sel=trip if len(trip)<=4000 else [trip[t] for t in rng.choice(len(trip),4000,replace=False)]
mins=[]
for a,b,c in sel:
    A,B,Cc=X[:,reps[a]],X[:,reps[b]],X[:,reps[c]]
    cnt=np.bincount(A*4+B*2+Cc,minlength=8)
    mins.append(cnt.min())
mins=np.array(mins)
print(f"sampled {len(sel)} triples; min-of-8-cells distribution:")
for q in [0,5,25,50,75,95,100]:
    print(f"  {q:3d}th pct: {np.percentile(mins,q):8.0f}")
print(f"  triples with an empty state: {100*(mins==0).mean():.1f}%")
print(f"  triples with all 8 states >=100: {100*(mins>=100).mean():.1f}%")
