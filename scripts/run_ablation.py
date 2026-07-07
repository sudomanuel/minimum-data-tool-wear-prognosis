"""run_ablation.py — ablation of the two pipeline branches (graphify blocks), on/off.
A) SENSOR/ML branch (per-tool wear rate, LOTO R2): FUSION(A/R)|SELECTION|AUGMENTATION|TUNING|MODEL.
B) PHYSICS few-shot branch (deployed, LOTO future-VB MAE): m|robust-fit|p-grid.
Fast |corr| top-8 selection proxy (for relative ablation); leakage-safe LOTO.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.insert(0, os.path.join(ROOT,"src"))
try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
except: pass
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from phm.augmentation_p8 import jitter
CENSOR=300.0

def tool_table():
    f=pd.read_csv(os.path.join(ROOT,"data/input/derived/features_experiment.csv"))
    meta=["experiment_id","tool_id","within_tool_order","vb_um","vc","fz","cooling"]
    feat=[c for c in f.columns if c not in meta]
    rows=[]
    for t,g in f.groupby("tool_id"):
        g=g.sort_values("within_tool_order")
        rate=np.polyfit(g.within_tool_order.values.astype(float),g.vb_um.values.astype(float),1)[0]
        rec={"tool":t,"rate":rate}; rec.update(g[feat].mean().to_dict()); rows.append(rec)
    return pd.DataFrame(rows), feat

def sensor_r2(df,feat,fusion="AR",selection=True,augment=True,tuning=True,model="ridge"):
    cols=feat
    if fusion=="A": cols=[c for c in feat if c.startswith("A_")]
    elif fusion=="R": cols=[c for c in feat if c.startswith("R_")]
    y=df["rate"].values; X=df[cols].reset_index(drop=True); pred=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X):
        Xtr,ytr=X.iloc[tr],y[tr]; sel=cols
        if selection:
            cc={c:(abs(np.corrcoef(Xtr[c],ytr)[0,1]) if Xtr[c].std()>1e-9 else 0.0) for c in cols}
            sel=sorted(cc,key=cc.get,reverse=True)[:8]
        mu,sd=Xtr[sel].mean(),Xtr[sel].std()+1e-9
        Xs=((Xtr[sel]-mu)/sd).to_numpy(float); yf=ytr
        if augment:
            Xa,ya=jitter(Xs,ytr,12,sigma=0.05,rng=np.random.default_rng(te[0])); Xs=np.vstack([Xs,Xa]); yf=np.concatenate([ytr,ya])
        if model=="rf": m=RandomForestRegressor(n_estimators=60,random_state=0).fit(Xs,yf)
        else: m=(RidgeCV(alphas=[1,10,100,1000]) if tuning else Ridge(alpha=10)).fit(Xs,yf)
        pred[te]=m.predict(((X.iloc[te][sel]-mu)/sd).to_numpy(float))
    return 1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)

def physics_mae(d,m=3,robust=True,pgrid=True):
    def ts(x,y):
        s=np.median([(y[j]-y[i])/(x[j]-x[i]) for i in range(len(x)) for j in range(i+1,len(x)) if x[j]!=x[i]]); return s,np.median(y-s*x)
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
    tools=sorted(d.tool_id.unique(),key=lambda t:int(t[1:])); E=[]
    for tt in tools:
        tr=d[d.tool_id!=tt]; g=d[d.tool_id==tt].sort_values("order"); o,v=g.order.values.astype(float),g.vb.values.astype(float)
        if len(o)<=m: continue
        fut=np.arange(m,len(o)); fut=fut[v[fut]<=CENSOR]
        if len(fut)==0: continue
        p=gp(tr) if pgrid else 0.5; x=o[:m]**p
        if robust: a,b=ts(x,v[:m])
        else: b,a=np.linalg.lstsq(np.column_stack([np.ones(m),x]),v[:m],rcond=None)[0]
        E.append(np.mean(np.abs((b+a*o[fut]**p)-v[fut])))
    return float(np.mean(E))

def main():
    df,feat=tool_table()
    print("A) SENSOR/ML branch ablation — per-tool wear rate, LOTO R² (<=0 = no signal):")
    for name,kw in [("FULL (A+R · select · augment · tuning · ridge)",dict()),
          ("  - selection",dict(selection=False)),("  - augmentation",dict(augment=False)),
          ("  - tuning (fixed alpha)",dict(tuning=False)),("  fusion=A-only",dict(fusion="A")),
          ("  fusion=R-only",dict(fusion="R")),("  - selection - augmentation",dict(selection=False,augment=False)),
          ("  model=RandomForest",dict(model="rf"))]:
        print(f"   {name:44} R²={sensor_r2(df,feat,**kw):+.2f}")
    d=pd.read_csv(os.path.join(ROOT,"data/input/derived/features_experiment.csv"))[["tool_id","within_tool_order","vb_um"]].drop_duplicates().rename(columns={"within_tool_order":"order","vb_um":"vb"}).sort_values(["tool_id","order"])
    print("\nB) PHYSICS few-shot branch ablation — LOTO future-VB MAE (lower better):")
    for name,kw in [("deployed (m=3 · robust · p-grid)",dict()),("  m=2",dict(m=2)),("  m=4",dict(m=4)),
                    ("  OLS (not robust)",dict(robust=False)),("  fixed p=0.5",dict(pgrid=False))]:
        print(f"   {name:34} MAE={physics_mae(d,**kw):.1f} um")
if __name__=="__main__": main()
