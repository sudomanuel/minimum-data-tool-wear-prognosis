"""run_readme_gap_audit.py — README modernization round: three methods shown in the legacy README
that were never adjudicated under the CURRENT 18-tool few-shot protocol.

  GAP-1 ELASTICNET (the legacy classical winner): added to the fair sensor harness (concurrent-VB
        reading, compact physics indicators, identical LOTO folds as F2/round-6).
  GAP-2 SHAP-GUIDED SELECTION (P8.7's 4th consensus vote): in-fold RandomForest + TreeExplainer over
        the full curated sensor bank -> top-12 features by mean |SHAP| -> Ridge; leakage-safe.
  GAP-3 PINN UNDER THE FEW-SHOT PROTOCOL (the legacy README's protagonist, absent from the current
        paper): fleet stage = small monotonicity-penalized MLP g(t) trained on the 17 training tools'
        curves (physics-informed shape); few-shot stage = affine adaptation y ~ a*g(t)+b on the m
        early points of the held-out tool (the project's fleet-shape + per-tool-scale philosophy).
        LOTO, m in {3,4}, records 11.02/5.63, base 11.57/9.67.

User rule: if any result VARIES the adjudicated picture, it enters the pipeline-flow graph;
otherwise it does not. Outputs: results/readme_gap_audit.csv
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from run_f2_fair_baseline import load as load_feats, tools_of as tools_of_f, phys_cols, num_cols, \
    loto_predict, m_pls_factory, CENSOR as CENSOR_F
from run_mcurve import load as load_curves, tools_of
CENSOR = 300.0
RECORDS = {3: 11.02, 4: 5.63}
BASE = {3: 11.57, 4: 9.67}


# ---------------- GAP-1: ElasticNet on the fair harness ----------------
def m_elasticnet(Xtr, ytr, Xte, tr):
    return ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=20000).fit(Xtr, ytr).predict(Xte)


# ---------------- GAP-2: SHAP-guided in-fold selection ----------------
def shap_selection_loto(f, k=12):
    import shap
    cols = [c for c in num_cols(f) if c not in ("vc", "fz")]
    P, Y, per = [], [], []
    for tt in tools_of_f(f):
        tr = f[(f.tool_id != tt) & (f.vb_um <= CENSOR_F)]
        te = f[(f.tool_id == tt) & (f.vb_um <= CENSOR_F)]
        if len(te) == 0 or len(tr) < 8:
            continue
        Xtr = tr[cols].to_numpy(float); ytr = tr.vb_um.to_numpy(float)
        rf = RandomForestRegressor(n_estimators=150, random_state=0).fit(Xtr, ytr)
        sv = shap.TreeExplainer(rf).shap_values(Xtr)
        top = list(np.array(cols)[np.argsort(np.abs(sv).mean(0))[::-1][:k]])
        sc = StandardScaler().fit(tr[top].to_numpy(float))
        pred = Ridge(alpha=10.0).fit(sc.transform(tr[top].to_numpy(float)), ytr) \
                                .predict(sc.transform(te[top].to_numpy(float)))
        yte = te.vb_um.to_numpy(float)
        P.append(pred); Y.append(yte); per.append(np.abs(pred - yte).mean())
    P, Y = np.concatenate(P), np.concatenate(Y)
    r2 = 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)
    return dict(MAE=float(np.mean(per)), R2=float(r2))


# ---------------- GAP-3: PINN under the few-shot protocol ----------------
def pinn_fewshot(d, m, seed=0, steps=1500, lam_mono=1.0):
    import torch
    torch.manual_seed(seed)

    def train_fleet(tr):
        ts, ys = [], []
        for _, g in tr.groupby("tool_id"):
            gg = g[g.vb <= CENSOR].sort_values("order")
            ts += list(gg.order.to_numpy(float)); ys += list(gg.vb.to_numpy(float))
        t = torch.tensor(ts, dtype=torch.float32).unsqueeze(1) / 20.0
        y = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
        net = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(),
                                  torch.nn.Linear(32, 32), torch.nn.Tanh(),
                                  torch.nn.Linear(32, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        grid = torch.linspace(0.05, 4.2, 120).unsqueeze(1)
        for _ in range(steps):
            opt.zero_grad()
            mse = torch.mean((net(t) - y) ** 2)
            gv = net(grid)
            mono = torch.mean(torch.relu(gv[:-1] - gv[1:]))     # penalize decreasing shape
            (mse + lam_mono * 100.0 * mono).backward()
            opt.step()
        return net

    per = []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        import torch
        net = train_fleet(tr)
        with torch.no_grad():
            gm = net(torch.tensor(o[:m], dtype=torch.float32).unsqueeze(1) / 20.0).numpy().ravel()
            gf = net(torch.tensor(o[fut], dtype=torch.float32).unsqueeze(1) / 20.0).numpy().ravel()
        # affine few-shot adaptation on the m early points (a >= 0)
        A = np.column_stack([gm, np.ones(m)])
        c, *_ = np.linalg.lstsq(A, v[:m], rcond=None)
        a = max(float(c[0]), 0.0); b = float(c[1]) if a > 0 else float(np.mean(v[:m] - 0 * gm))
        pred = a * gf + c[1]
        per.append(np.abs(pred - v[fut]).mean())
    return float(np.mean(per))


def main():
    print("README GAP AUDIT — three legacy-README methods under the CURRENT 18-tool protocol.\n")
    rows = []

    f = load_feats()
    cols = phys_cols(f)
    print("GAP-1 ElasticNet (fair sensor harness, concurrent VB; references: fair PLS R2 -0.14, "
          "R_only +0.01):")
    res = loto_predict(f, cols, m_elasticnet)
    print(f"  ElasticNet: MAE {res['MAE']:.1f}  R2 {res['R2']:+.2f}")
    rows.append(dict(gap="ElasticNet_fair", MAE=round(res["MAE"], 1), R2=round(res["R2"], 2)))

    print("\nGAP-2 SHAP-guided in-fold selection (full curated bank -> top-12 -> Ridge; "
          "references: naive ridge R2 -1.76/-0.97):")
    res = shap_selection_loto(f)
    print(f"  SHAP-select+Ridge: MAE {res['MAE']:.1f}  R2 {res['R2']:+.2f}")
    rows.append(dict(gap="SHAP_selection", MAE=round(res["MAE"], 1), R2=round(res["R2"], 2)))

    print("\nGAP-3 PINN under the few-shot protocol (fleet monotone MLP + affine few-shot):")
    d = load_curves()
    for m in (3, 4):
        mae = pinn_fewshot(d, m)
        beat = "** BEATS RECORD **" if mae < RECORDS[m] - 0.05 else ""
        print(f"  m={m}: PINN few-shot MAE {mae:6.2f}  (base {BASE[m]}, record {RECORDS[m]}) {beat}")
        rows.append(dict(gap=f"PINN_fewshot_m{m}", MAE=round(mae, 2), R2=None))

    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "readme_gap_audit.csv"), index=False)
    print("\nwrote results/readme_gap_audit.csv")


if __name__ == "__main__":
    main()
