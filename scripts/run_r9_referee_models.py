# -*- coding: utf-8 -*-
"""run_r9_referee_models.py — ROUND 9 (supervisor review): two referee requests, one script.

(A) EXTRA DATA-DRIVEN MODELS (comment: "Xgboost, LASSO, RIDGE, ELASTIC NET"):
    LASSO, ElasticNet and XGBoost added to the naive branch — reading concurrent VB from the full
    vibration-feature bank — under the identical leakage-safe LOOCV protocol of run_f2_fair_baseline
    (per-tool hold-out, StandardScaler fit on training folds only, wear regime VB <= 300).

(B) PHYSICS-EQUATION HYBRID (comment: "Incluir una ecuacion fisica para ver como mejora"):
    give the data-driven baseline the physics wear equation VB(t) = b + a t^p and measure how much
    it improves. Two standard hybridisations:
      - residual learning:  yhat = physics_fleet(t) + ML(features -> VB - physics_fleet(t))
      - physics feature:    ML([features, tau]) with tau = t^{p*} appended as a regressor
    where physics_fleet is the zero-shot fleet law (global exponent p* fitted on training tools by
    the pooled criterion of the manuscript's exponent equation; level/rate = training-fleet medians).

Pre-stated reading: these models answer the CONCURRENT-reading question of Section 4.1/4.8. The
records to beat for the deployed few-shot forecaster remain 11.0 (m=3) / 5.6 um (m=4).
Output: results/r9_referee_models.csv
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_f2_fair_baseline import load, tools_of, num_cols   # identical data + protocol helpers

CENSOR = 300.0
P_GRID = np.arange(0.05, 0.951, 0.05)

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAVE_XGB = False


# ---------------- physics fleet law (zero-shot, training tools only) ----------------
def fit_fleet_law(tr):
    """Global exponent p* by the pooled criterion of the manuscript (per-tool OLS refit per candidate
    p), then fleet-median level b and rate a in the linearising coordinate tau = t^p*."""
    best_p, best = 0.5, np.inf
    for p in P_GRID:
        tot = 0.0
        for _, g in tr.groupby("tool_id"):
            gg = g[g.vb_um <= CENSOR].sort_values("within_tool_order")
            if len(gg) < 2:
                continue
            tau = gg.within_tool_order.to_numpy(float) ** p
            A = np.column_stack([np.ones(len(tau)), tau])
            coef, res, *_ = np.linalg.lstsq(A, gg.vb_um.to_numpy(float), rcond=None)
            pred = A @ coef
            tot += float(np.sum((gg.vb_um.to_numpy(float) - pred) ** 2))
        if tot < best:
            best, best_p = tot, p
    bs, as_ = [], []
    for _, g in tr.groupby("tool_id"):
        gg = g[g.vb_um <= CENSOR].sort_values("within_tool_order")
        if len(gg) < 2:
            continue
        tau = gg.within_tool_order.to_numpy(float) ** best_p
        A = np.column_stack([np.ones(len(tau)), tau])
        coef, *_ = np.linalg.lstsq(A, gg.vb_um.to_numpy(float), rcond=None)
        bs.append(coef[0]); as_.append(coef[1])
    return best_p, float(np.median(bs)), float(np.median(as_))


def physics_pred(orders, p, b, a):
    return b + a * np.asarray(orders, float) ** p


# ---------------- model zoo ----------------
def make_models():
    zoo = {
        "Ridge (alpha=10)": lambda: Ridge(alpha=10.0),
        "LASSO (alpha=1)": lambda: Lasso(alpha=1.0, max_iter=20000),
        "ElasticNet (alpha=1, l1=0.5)": lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=20000),
        "Random forest (200 trees)": lambda: RandomForestRegressor(n_estimators=200, random_state=0),
    }
    if HAVE_XGB:
        zoo["XGBoost (500 trees, lr=0.05)"] = lambda: XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.9,
            colsample_bytree=0.9, random_state=0, verbosity=0)
    else:
        zoo["Gradient boosting (sklearn)"] = lambda: GradientBoostingRegressor(random_state=0)
    return zoo


def loocv_eval(f, cols, model_fn, mode="plain"):
    """Leakage-safe LOOCV by tool. mode: plain | resid (physics residual) | feat (tau feature)."""
    P, Y, per_mae = [], [], []
    for tt in tools_of(f):
        tr = f[(f.tool_id != tt) & (f.vb_um <= CENSOR)]
        te = f[(f.tool_id == tt) & (f.vb_um <= CENSOR)]
        if len(te) == 0 or len(tr) < 8:
            continue
        Xtr = tr[cols].to_numpy(float); Xte = te[cols].to_numpy(float)
        ytr = tr.vb_um.to_numpy(float); yte = te.vb_um.to_numpy(float)
        if mode in ("resid", "feat"):
            p, b, a = fit_fleet_law(tr)
        if mode == "feat":
            Xtr = np.column_stack([Xtr, tr.within_tool_order.to_numpy(float) ** p])
            Xte = np.column_stack([Xte, te.within_tool_order.to_numpy(float) ** p])
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        if mode == "resid":
            phys_tr = physics_pred(tr.within_tool_order, p, b, a)
            phys_te = physics_pred(te.within_tool_order, p, b, a)
            m = model_fn().fit(Xtr, ytr - phys_tr)
            pred = phys_te + np.asarray(m.predict(Xte), float).ravel()
        else:
            m = model_fn().fit(Xtr, ytr)
            pred = np.asarray(m.predict(Xte), float).ravel()
        P.extend(pred); Y.extend(yte); per_mae.append(float(np.mean(np.abs(pred - yte))))
    P, Y = np.asarray(P), np.asarray(Y)
    r2 = 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)
    return float(np.mean(per_mae)), float(r2)


def main():
    f = load()
    cols = num_cols(f)
    print(f"data: {f.tool_id.nunique()} tools, {len(f)} rows, {len(cols)} descriptors "
          f"(xgboost={'yes' if HAVE_XGB else 'sklearn fallback'})")
    rows = []
    # physics-only zero-shot fleet law (context row: the equation alone, no sensors, no per-tool data)
    P, Y, per = [], [], []
    for tt in tools_of(f):
        tr = f[(f.tool_id != tt) & (f.vb_um <= CENSOR)]
        te = f[(f.tool_id == tt) & (f.vb_um <= CENSOR)]
        if len(te) == 0:
            continue
        p, b, a = fit_fleet_law(tr)
        pred = physics_pred(te.within_tool_order, p, b, a)
        P.extend(pred); Y.extend(te.vb_um); per.append(float(np.mean(np.abs(pred - te.vb_um))))
    P, Y = np.asarray(P), np.asarray(Y)
    r2 = 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)
    rows.append(dict(model="Physics fleet law alone (zero-shot, no sensors)",
                     mode="physics", MAE=float(np.mean(per)), R2=float(r2)))
    print(rows[-1])

    zoo = make_models()
    for name, fn in zoo.items():
        mae, r2 = loocv_eval(f, cols, fn, mode="plain")
        rows.append(dict(model=name, mode="features only", MAE=mae, R2=r2))
        print(rows[-1])
    for name, fn in zoo.items():
        mae, r2 = loocv_eval(f, cols, fn, mode="resid")
        rows.append(dict(model=name + " + physics residual", mode="hybrid resid", MAE=mae, R2=r2))
        print(rows[-1])
    for name, fn in zoo.items():
        mae, r2 = loocv_eval(f, cols, fn, mode="feat")
        rows.append(dict(model=name + " + tau feature", mode="hybrid feat", MAE=mae, R2=r2))
        print(rows[-1])

    out = pd.DataFrame(rows).round(2)
    dst = os.path.join(ROOT, "results", "r9_referee_models.csv")
    out.to_csv(dst, index=False)
    print("wrote", dst)


if __name__ == "__main__":
    main()
