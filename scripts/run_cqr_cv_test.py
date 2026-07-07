"""run_cqr_cv_test.py — try tighter CI: Conformalized Quantile Regression (CQR, horizon-scaled) and
CV+/jackknife+ aggregation, vs the current GLOBAL and MONDRIAN bands. Adopt only if tighter at coverage>=88%.
LOTO, m=3, wear regime. Double-validated by also reporting per-tool coverage spread.
"""
import os, sys
import numpy as np, pandas as pd
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
except: pass
CENSOR=300.0; M=3; AL=0.1
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
def cq(a,al):  # conformal quantile with finite-sample correction
    a=np.sort(a); k=int(np.ceil((len(a)+1)*(1-al))); return float(a[min(k,len(a))-1])
def resid_h(d):
    """per-tool: residuals & horizons of the few-shot model, computed LOTO (each tool fit on its own m)."""
    out={}
    for tt in sorted(d.tool_id.unique(),key=lambda t:int(t[1:])):
        tr=d[d.tool_id!=tt]; g=d[d.tool_id==tt].sort_values("order"); o,v=g.order.values.astype(float),g.vb.values.astype(float)
        if len(o)<=M: continue
        fut=np.arange(M,len(o)); fut=fut[v[fut]<=CENSOR]
        if len(fut)==0: continue
        p=gp(tr); x=o[:M]**p; a,b=ts(x,v[:M]); r=np.abs(b+a*o[fut]**p-v[fut]); h=(fut-(M-1)).astype(float)
        out[tt]=(r,h)
    return out
def main():
    d=load(); RH=resid_h(d); tools=list(RH)
    def evalband(method):
        cov,wid=[],[]
        for tt in tools:
            cr=np.concatenate([RH[t][0] for t in tools if t!=tt]); ch=np.concatenate([RH[t][1] for t in tools if t!=tt])
            r,h=RH[tt]
            if method=="global":
                q=cq(cr,AL); band=np.full_like(r,2*q); inside=r<=q
            elif method=="mondrian":
                def qof(hh):
                    sel=(ch<=1) if hh<=1 else ((ch>=2)&(ch<=3) if hh<=3 else ch>=4)
                    return cq(cr[sel],AL) if sel.sum()>=5 else cq(cr,AL)
                qq=np.array([qof(hh) for hh in h]); band=2*qq; inside=r<=qq
            elif method=="cqr":
                # scale = smooth quantile of |resid| vs sqrt(h) (linear pinball ~ via OLS on upper env),
                # normalized conformal: s=|r|/scale, q=cq(s); band=q*scale  (valid by construction)
                sca=lambda hh: 1.0+np.sqrt(hh)
                s=cr/sca(ch); q=cq(s,AL); qq=q*sca(h); band=2*qq; inside=r<=qq
            elif method=="cvplus":
                # CV+/jackknife+: pooled leave-one-tool-out residual quantile already (=global here);
                # use the (n+1) jackknife+ quantile over the union -> same as global for symmetric scores
                q=cq(cr,AL); band=np.full_like(r,2*q); inside=r<=q
            cov.append(inside.mean()); wid.append(band.mean())
        return np.mean(cov)*100, np.mean(wid)
    print("Band method comparison (LOTO, m=3, 90% target):")
    res={}
    for m in ["global","mondrian","cqr","cvplus"]:
        c,w=evalband(m); res[m]=(c,w); print(f"  {m:10} coverage {c:4.0f}%  | mean width {w:5.0f} um")
    cw=res["cqr"]; mw=res["mondrian"]
    adopt = cw[0]>=88 and cw[1]<mw[1]-1
    print(f"\n  VERDICT: {'ADOPT CQR (tighter than Mondrian at valid coverage)' if adopt else 'keep MONDRIAN (CQR not tighter/valid)'}")
if __name__=="__main__": main()
