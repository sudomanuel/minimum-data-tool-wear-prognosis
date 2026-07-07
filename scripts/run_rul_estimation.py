"""run_rul_estimation.py — estimate RUL (and validate where ground truth exists).
Deployed model: physics power VB=b+a*order^p + few-shot self-adapt (Theil-Sen on first m). For each tool,
predict the failure crossing t_fail (VB>=VB_fail), RUL = t_fail - last_observed, and a band-derived RUL
WINDOW (Mondrian conformal). Validate on tools that actually cross after the few-shot window.
"""
import os, sys
import numpy as np, pandas as pd
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
except: pass
CENSOR=300.0; M=3; VB_FAIL=200.0
def load():
    f=pd.read_csv(os.path.join(ROOT,"data/input/derived/features_experiment.csv"))
    d=f[["tool_id","within_tool_order","vb_um"]].drop_duplicates().rename(columns={"within_tool_order":"order","vb_um":"vb"})
    return d.sort_values(["tool_id","order"]).reset_index(drop=True)
def gp(tr):
    bp,be=0.5,np.inf
    for p in np.arange(0.2,1.001,0.05):
        tot=0
        for _,g in tr.groupby("tool_id"):
            gg=g[g.vb<=CENSOR]
            if len(gg)<2: continue
            A=np.column_stack([np.ones(len(gg)),gg.order.values**p]); c,*_=np.linalg.lstsq(A,gg.vb.values,rcond=None); tot+=((A@c-gg.vb.values)**2).sum()
        if tot<be: be,bp=tot,p
    return bp
def ts(x,y):
    s=np.median([(y[j]-y[i])/(x[j]-x[i]) for i in range(len(x)) for j in range(i+1,len(x)) if x[j]!=x[i]]); return s,np.median(y-s*x)
def mondrian_q(tr,p,al=0.1):
    R,H=[],[]
    for _,g in tr.groupby("tool_id"):
        o,v=g.order.values.astype(float),g.vb.values.astype(float)
        if len(o)<=M: continue
        fut=np.arange(M,len(o)); fut=fut[v[fut]<=CENSOR]
        if len(fut)==0: continue
        x=o[:M]**p; a,b=ts(x,v[:M]); R+=list(np.abs(b+a*o[fut]**p-v[fut])); H+=list((fut-(M-1)).astype(int))
    R,H=np.array(R),np.array(H); out={}
    for bn,sel in [("near",H<=1),("mid",(H>=2)&(H<=3)),("far",H>=4)]:
        rr=np.sort(R[sel]); k=int(np.ceil((len(rr)+1)*(1-al))) if len(rr)>=5 else None
        out[bn]=float(rr[min(k,len(rr))-1]) if k else float(np.sort(R)[int(np.ceil((len(R)+1)*(1-al)))-1])
    return out
def main():
    d=load(); tools=sorted(d.tool_id.unique(),key=lambda t:int(t[1:]))
    rows=[]; errs=[]; hits=0; nval=0
    for tt in tools:
        tr=d[d.tool_id!=tt]; g=d[d.tool_id==tt].sort_values("order")
        o,v=g.order.values.astype(float),g.vb.values.astype(float)
        if len(o)<=M: continue
        p=gp(tr); qm=mondrian_q(tr,p); x=o[:M]**p; a,b=ts(x,v[:M])
        grid=np.arange(o[M-1],o[M-1]+60)
        pred=b+a*grid**p
        def qof(h): return qm["near"] if h<=1 else (qm["mid"] if h<=3 else qm["far"])
        qs=np.array([qof(int(gr-o[M-1])) for gr in grid]); hi=pred+qs; lo=pred-qs
        def cross(c):
            idx=np.where(c>=VB_FAIL)[0]; return int(grid[idx[0]]) if len(idx) else None
        tf=cross(pred); te=cross(hi); tl=cross(lo); last=o[M-1]
        # ground truth crossing in observed data
        ghit=np.where(v>=VB_FAIL)[0]; tobs=int(o[ghit[0]]) if len(ghit) else None
        valid = (tobs is not None and tobs>M-1)     # crosses AFTER the few-shot window
        rul_pred = None if tf is None else tf-last
        err=None
        if valid and tf is not None:
            err=abs(tf-tobs); errs.append(err); nval+=1
            if te is not None and tl is not None and te<=tobs<=tl: hits+=1
            elif te is not None and tl is None and tobs>=te: hits+=1
        rows.append(dict(tool=tt, last_obs=int(last), t_fail_pred=tf, RUL_pred=rul_pred,
                         RUL_window=(None if te is None else int(te-last), None if tl is None else int(tl-last)),
                         t_fail_obs=tobs, validated=valid, abs_err_cuts=err))
    R=pd.DataFrame(rows); R.to_csv(os.path.join(ROOT,"results","rul_estimation.csv"),index=False)
    print(f"RUL estimation (VB_fail={VB_FAIL:.0f}um, m={M}). Tools with post-window ground truth: {nval}")
    print(R.to_string(index=False))
    if errs:
        print(f"\n  RUL |error| on validated tools: mean {np.mean(errs):.1f} cuts (n={nval}); "
              f"window contains true t_fail: {hits}/{nval}")
    print("\n  (Tools that never reach VB_fail are right-censored: t_fail is an honest extrapolation, not validated.)")
if __name__=="__main__": main()
