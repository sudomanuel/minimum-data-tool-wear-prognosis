"""run_augmentation_test.py — can AUGMENTATION substitute for more data on the deployed (physics few-shot)
model? In each LOTO fold we synthesize extra TRAINING tools and re-fit the global exponent p + the
conformal residual pool, then predict the REAL held-out tool. Methods:
  none | jitter (noise on training curves, monotone-preserved) | interp (convex combos of training pairs).
Adopt only if it lowers MAE on real held-out tools (and keeps a valid band). LOTO, m=3.
"""
import os, sys
import numpy as np, pandas as pd
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
except: pass
CENSOR=300.0; M=3; RNG=np.random.default_rng(0)
def load():
    f=pd.read_csv(os.path.join(ROOT,"data/input/derived/features_experiment.csv"))
    d=f[["tool_id","within_tool_order","vb_um"]].drop_duplicates().rename(columns={"within_tool_order":"order","vb_um":"vb"})
    return d.sort_values(["tool_id","order"]).reset_index(drop=True)
def curves(d):
    return {t:(g.order.values.astype(float),g.vb.values.astype(float)) for t,g in d.groupby("tool_id")}
def gp_from(curvelist):
    bp,be=0.5,np.inf
    for p in np.arange(0.2,1.001,0.05):
        tot=0
        for o,v in curvelist:
            sel=v<=CENSOR
            if sel.sum()<2: continue
            A=np.column_stack([np.ones(sel.sum()),o[sel]**p]); c,*_=np.linalg.lstsq(A,v[sel],rcond=None); tot+=((A@c-v[sel])**2).sum()
        if tot<be: be,bp=tot,p
    return bp
def ts(x,y):
    s=np.median([(y[j]-y[i])/(x[j]-x[i]) for i in range(len(x)) for j in range(i+1,len(x)) if x[j]!=x[i]]); return s,np.median(y-s*x)
def augment(train_curves, method, factor=3):
    if method=="none": return train_curves
    syn=list(train_curves)
    keys=train_curves
    for _ in range(factor*len(train_curves)):
        if method=="jitter":
            o,v=keys[RNG.integers(len(keys))]
            nv=v*(1+RNG.normal(0,0.05,len(v))); nv=np.maximum.accumulate(np.maximum(nv,1.0))  # keep monotone
            syn.append((o.copy(),nv))
        elif method=="interp":
            (o1,v1),(o2,v2)=keys[RNG.integers(len(keys))],keys[RNG.integers(len(keys))]
            L=min(len(o1),len(o2)); w=RNG.uniform(0.3,0.7)
            o=o1[:L]; nv=w*v1[:L]+(1-w)*v2[:L]; nv=np.maximum.accumulate(nv)
            syn.append((o,nv))
    return syn
def run(d,method):
    tools=sorted(d.tool_id.unique(),key=lambda t:int(t[1:])); C=curves(d); E=[]
    for tt in tools:
        o,v=C[tt]
        if len(o)<=M: continue
        fut=np.arange(M,len(o)); fut=fut[v[fut]<=CENSOR]
        if len(fut)==0: continue
        trc=[C[t] for t in tools if t!=tt]
        trc_aug=augment(trc,method)
        p=gp_from(trc_aug); x=o[:M]**p; a,b=ts(x,v[:M])
        E.append(np.mean(np.abs((b+a*o[fut]**p)-v[fut])))
    return float(np.mean(E))
def main():
    d=load()
    print("Augmentation as a substitute for more data (deployed physics few-shot, LOTO m=3):")
    base=run(d,"none")
    for m in ["none","jitter","interp"]:
        mae=run(d,m); tag=" (baseline)" if m=="none" else (" -> "+("helps" if mae<base-0.3 else "no gain"))
        print(f"   augmentation={m:7} MAE={mae:.1f} um{tag}")
    print("\n  Note: synthetic curves may only inform the global exponent p; the per-tool scale still comes")
    print("        from the REAL tool's own m points. Synthetic data must never enter the test set.")
if __name__=="__main__": main()
