"""
Model-implied MAIN effect for each of the 43 representative features.

Same 16 backgrounds as s15 (seed 11, held-out test genotypes).  For each feature,
set its position to the reference allele and to the alternate allele on each
background, run ASPred-Affinity forward, take the difference, average over
backgrounds.  Compare against the additive Tobit coefficients per group.

Sign convention matches the Tobit fit: coefficient is the effect of carrying the
ALT allele, so d = y(alt) - y(ref).
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
feat,names,ABS=F["feat"],F["names"],F["ABS"]; ref=F["ref"]; reps=D["reps"]

te=pd.read_csv(f"{P}/data/paired/rbd_split_test.csv",
               usecols=["antibody_name","antibody_sequence","rbd_sequence"])
abseq={r.antibody_name:r.antibody_sequence for r in te.drop_duplicates("antibody_name").itertuples()}
ABS=[a for a in ABS if a in abseq]
geno=te.rbd_sequence.unique()
rng=np.random.default_rng(11)                      # SAME seed/draw as s15
bg=[geno[t] for t in rng.choice(len(geno),A.n_backgrounds,replace=False)]
print(f"backgrounds: {len(bg)} (same draw as s15)   features: {len(reps)}   antibodies: {len(ABS)}")

def mutate(s,changes):
    l=list(s)
    for p,a in changes: l[p]=a
    return "".join(l)

seqs={}; index=[]
for fi,i in enumerate(reps):
    p,alt=feat[i]; r=ref[p]
    for bi,b in enumerate(bg):
        pair=[]
        for al in (r,alt):
            s=mutate(b,[(p,al)])
            if s not in seqs: seqs[s]=len(seqs)
            pair.append(seqs[s])
        index.append((fi,bi,*pair))
seqlist=[None]*len(seqs)
for s,k in seqs.items(): seqlist[k]=s
print(f"unique RBD sequences to score: {len(seqlist)}")

dev=get_device(); print("device:",dev)
tok=EsmTokenizer.from_pretrained(MODEL_NAME)
model=PairedAffinityModel().to(dev); model=load_checkpoint(model,A.checkpoint_dir,dev); model.eval()

@torch.no_grad()
def score(ab_name):
    out=np.empty(len(seqlist))
    ab_tok_1=tok([abseq[ab_name]],padding=True,truncation=True,max_length=512,return_tensors="pt")
    for st in range(0,len(seqlist),A.batch_size):
        chunk=seqlist[st:st+A.batch_size]
        ag=tok(chunk,padding=True,truncation=True,max_length=512,return_tensors="pt")
        ag={k:v.to(dev) for k,v in ag.items()}
        abt={k:v.repeat(len(chunk),1).to(dev) for k,v in ab_tok_1.items()}
        out[st:st+len(chunk)]=model(abt,ag).float().cpu().numpy()
    return out

idx=np.array(index)
rows=[]
for ab in ABS:
    y=score(ab)
    d=y[idx[:,3]]-y[idx[:,2]]              # alt - ref
    Dm=d.reshape(len(reps),len(bg))
    rows.append(pd.DataFrame(dict(group=ab,feature=[names[i] for i in reps],
        model_main=Dm.mean(1),model_main_sd=Dm.std(1))))
    print(f"  {ab:12s} mean |d| = {np.abs(Dm.mean(1)).mean():.5f}   "
          f"median sd across backgrounds = {np.median(Dm.std(1)):.5f}")
M=pd.concat(rows,ignore_index=True)
M.to_csv(f"{RES}/model_implied_main_effects.csv",index=False)
print(f"\nwrote {RES}/model_implied_main_effects.csv ({len(M)} rows)")
