"""
Model-implied pairwise interaction for every estimable allele pair.

For each pair (i,j) and each background genotype g drawn from real test genotypes,
build the four sequences (ref/ref, alt/ref, ref/alt, alt/alt) at the two positions,
run ASPred-Affinity forward for each antibody, and form the second difference
    d_ij(g) = y11 - y10 - y01 + y00
then average over backgrounds.  Compare against the fitted Tobit coefficients.
"""
import os as _os, sys, pickle, warnings, argparse
warnings.filterwarnings("ignore")
_HERE=_os.path.dirname(_os.path.abspath(__file__))
SP=_os.path.join(_os.path.dirname(_HERE),"cache"); RES=_os.path.join(_os.path.dirname(_HERE),"results")
P=_os.path.abspath(_os.path.join(_HERE,"..","..",".."))
sys.path.insert(0,_os.path.join(P,"src"))
import numpy as np, pandas as pd, torch
from transformers import EsmTokenizer
from paired_affinity_model import MODEL_NAME, PairedAffinityModel
from train_paired_affinity import get_device, load_checkpoint

ap=argparse.ArgumentParser()
ap.add_argument("--n_backgrounds",type=int,default=16)
ap.add_argument("--batch_size",type=int,default=256)
ap.add_argument("--checkpoint_dir",default=_os.path.join(P,"models/paired_affinity/run5_rbd_split_continued"))
A=ap.parse_args()

F=pickle.load(open(f"{SP}/feat.pkl","rb")); D=pickle.load(open(f"{SP}/scan3.pkl","rb"))
feat,names,ABS=F["feat"],F["names"],F["ABS"]; ref=F["ref"]; pairs=D["pairs"]

# ---- data: backgrounds from TEST genotypes, antibody sequences -----------
te=pd.read_csv(f"{P}/data/paired/rbd_split_test.csv",
               usecols=["antibody_name","antibody_sequence","rbd_sequence"])
abseq={r.antibody_name:r.antibody_sequence for r in
       te.drop_duplicates("antibody_name").itertuples()}
ABS=[a for a in ABS if a in abseq]
geno=te.rbd_sequence.unique()
rng=np.random.default_rng(11)
bg=[geno[t] for t in rng.choice(len(geno),A.n_backgrounds,replace=False)]
print(f"backgrounds: {len(bg)} drawn from {len(geno)} held-out test genotypes")
print(f"pairs: {len(pairs)}   antibodies: {len(ABS)}")

def mutate(s,changes):
    l=list(s)
    for p,a in changes: l[p]=a
    return "".join(l)

# ---- enumerate the four corners for every (pair, background) -------------
seqs={}; index=[]
for pi,(i,j) in enumerate(pairs):
    pi_,ai=feat[i]; pj_,aj=feat[j]
    ri,rj=ref[pi_],ref[pj_]
    for bi,b in enumerate(bg):
        corner=[]
        for vi in (0,1):
            for vj in (0,1):
                s=mutate(b,[(pi_,ai if vi else ri),(pj_,aj if vj else rj)])
                if s not in seqs: seqs[s]=len(seqs)
                corner.append(seqs[s])
        index.append((pi,bi,*corner))
seqlist=[None]*len(seqs)
for s,k in seqs.items(): seqlist[k]=s
print(f"unique RBD sequences to score: {len(seqlist)}  "
      f"({len(pairs)*len(bg)*4} corner evaluations, deduplicated)")

dev=get_device(); print("device:",dev)
tok=EsmTokenizer.from_pretrained(MODEL_NAME)
model=PairedAffinityModel().to(dev); model=load_checkpoint(model,A.checkpoint_dir,dev); model.eval()

@torch.no_grad()
def score(ab_name):
    out=np.empty(len(seqlist),dtype=np.float64)
    ab=abseq[ab_name]
    ab_tok_1=tok([ab],padding=True,truncation=True,max_length=512,return_tensors="pt")
    for st in range(0,len(seqlist),A.batch_size):
        chunk=seqlist[st:st+A.batch_size]
        ag=tok(chunk,padding=True,truncation=True,max_length=512,return_tensors="pt")
        ag={k:v.to(dev) for k,v in ag.items()}
        abt={k:v.repeat(len(chunk),1).to(dev) for k,v in ab_tok_1.items()}
        out[st:st+len(chunk)]=model(abt,ag).float().cpu().numpy()
    return out

idx=np.array(index)   # (npair*nbg, 2+4): pair, bg, c00,c01,c10,c11
rows=[]
for ab in ABS:
    y=score(ab)
    d=y[idx[:,2]]-y[idx[:,3]]-y[idx[:,4]]+y[idx[:,5]]   # y00 - y01 - y10 + y11
    Dm=d.reshape(len(pairs),len(bg))
    rows.append(pd.DataFrame(dict(group=ab,
        feat_i=[names[i] for i,_ in pairs],feat_j=[names[j] for _,j in pairs],
        term=[f"{names[i]} x {names[j]}" for i,j in pairs],
        model_interaction=Dm.mean(1),model_interaction_sd=Dm.std(1),
        model_interaction_med=np.median(Dm,1))))
    print(f"  {ab:12s} mean |d| = {np.abs(Dm.mean(1)).mean():.5f}   "
          f"sd across backgrounds (median) = {np.median(Dm.std(1)):.5f}")
M=pd.concat(rows,ignore_index=True)
M.to_csv(f"{RES}/model_implied_interactions.csv",index=False)
np.save(f"{SP}/model_interaction_by_background.npy",
        np.stack([r.model_interaction.to_numpy() for r in rows]))
print(f"\nwrote {RES}/model_implied_interactions.csv  ({len(M)} rows)")
