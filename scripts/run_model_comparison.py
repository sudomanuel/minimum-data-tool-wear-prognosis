"""
run_model_comparison.py — multi-tool LOTO model comparison on the current VB targets.

One protocol, identical future window for every model (leakage-safe):
  * Leave-One-Tool-Out; few-shot m=3 early VB points of the held-out tool.
  * Score = future-VB MAE on the held-out tool's points with order >= m, wear regime (VB <= 300).

Models (all scored on the SAME future points):
  1. Average-wear-curve (population)   : population mean VB(order) over training tools + few-shot offset.
  2. Physics-integrated (our model)    : population-anchored monotone curve + few-shot offset
                                         (matches the baseline by construction — same MAE).
  3. PINN power, condition-conditioned : condition predicts the per-tool wear RATE
                                         (OLS rate ~ Vc,fz,cool on training tools); anchored at the
                                         few-shot point: VB(o)=VB(m-1)+rate(c)*(o-o_{m-1}). Tests
                                         whether condition beats the population mean.

Output: results/model_comparison.csv + console.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from phm.prognostic_system import fit_population, fewshot_offset, predict_vb, _curve_at

M = 3
CENSOR = 300.0


def load():
    f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
    d = f[["tool_id", "within_tool_order", "vb_um", "vc", "fz", "cooling"]].drop_duplicates()
    d = d.rename(columns={"within_tool_order": "order", "vb_um": "vb"}).sort_values(["tool_id", "order"])
    d["cool"] = d.cooling.astype(str).str.lower().str.contains("cool").astype(float)
    d["vcn"] = (d.vc - 67.0) / 10.0
    d["fzn"] = (d.fz - 0.19) / 0.1
    return d.reset_index(drop=True)


def tool_rate(g):
    """OLS slope of VB vs order over the tool's wear-regime points."""
    gg = g[g.vb <= CENSOR]
    if len(gg) < 2:
        return np.nan
    return float(np.polyfit(gg.order.to_numpy(float), gg.vb.to_numpy(float), 1)[0])


def fit_global_p(tr):
    """Shared power exponent p minimizing pooled per-tool residual of VB = b + k*o^p (p<=1)."""
    best_p, best_err = 0.5, np.inf
    for p in np.arange(0.2, 1.01, 0.1):
        tot = 0.0
        for _, g2 in tr.groupby("tool_id"):
            gg = g2[g2.vb <= CENSOR]
            if len(gg) < 2:
                continue
            o2 = gg.order.to_numpy(float); y2 = gg.vb.to_numpy(float)
            A = np.column_stack([np.ones(len(o2)), o2 ** p])
            coef, *_ = np.linalg.lstsq(A, y2, rcond=None)
            tot += float(np.sum((A @ coef - y2) ** 2))
        if tot < best_err:
            best_err, best_p = tot, p
    return float(best_p)


def main():
    d = load()
    tools = sorted(d.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))
    mae_base, mae_phys, mae_pinn = [], [], []   # per-tool MAE (equal weight per tool, LOTO protocol)

    for tt in tools:
        tr = d[d.tool_id != tt]
        g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= M:
            continue
        fut = np.arange(M, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        ytru = v[fut]

        # 1) population (= our physics-integrated model, by construction)
        pop = fit_population(tr.rename(columns={"order": "order", "vb": "vb"}))
        off = fewshot_offset(pop, o, v, M)
        yb = predict_vb(pop, o[fut], off)
        mae_base.append(float(np.mean(np.abs(yb - ytru))))

        # 3) PINN power, condition-conditioned: VB(o)=anchor + k(c)*(o^p - o_a^p), k(c) OLS on condition
        p = fit_global_p(tr)
        ks, X = [], []
        for _, g2 in tr.groupby("tool_id"):
            gg = g2[g2.vb <= CENSOR]
            if len(gg) < 2:
                continue
            o2 = gg.order.to_numpy(float); y2 = gg.vb.to_numpy(float)
            A = np.column_stack([np.ones(len(o2)), o2 ** p])
            coef, *_ = np.linalg.lstsq(A, y2, rcond=None)
            ks.append(coef[1]); X.append(g2[["vcn", "fzn", "cool"]].iloc[0].to_numpy(float))
        X = np.column_stack([np.ones(len(X)), np.array(X)]); ks = np.array(ks)
        anchor_o, anchor_v = o[M - 1], v[M - 1]
        # 2) physics-integrated wear law + few-shot SELF-adaptation: fit the physics basis
        #    VB = b + a*order^p to the tool's OWN first m points (robust Theil-Sen slope -> noise-safe).
        #    The tool's early trajectory is the only signal that generalizes under scarcity.
        op = o[:M] ** p
        slopes = [(v[j] - v[i]) / (op[j] - op[i])
                  for i in range(M) for j in range(i + 1, M) if op[j] != op[i]]
        a_self = float(np.median(slopes)); b_self = float(np.median(v[:M] - a_self * op))
        yphys = b_self + a_self * (o[fut] ** p)
        mae_phys.append(float(np.mean(np.abs(yphys - ytru))))
        # 3) does CONDITION help? rate predicted from condition (OLS) instead of from the tool's own data
        kcoef, *_ = np.linalg.lstsq(X, ks, rcond=None)
        c = g[["vcn", "fzn", "cool"]].iloc[0].to_numpy(float)
        k_hat = float(kcoef[0] + kcoef[1:] @ c)
        yp = anchor_v + k_hat * (o[fut] ** p - anchor_o ** p)
        mae_pinn.append(float(np.mean(np.abs(yp - ytru))))

    base = float(np.mean(mae_base)); phys = float(np.mean(mae_phys)); pinn = float(np.mean(mae_pinn))
    rows = [dict(model="Average-wear-curve (population)", mae_um=round(base, 1), role="the bar"),
            dict(model="Physics-integrated + few-shot self-adaptation (our model)", mae_um=round(phys, 1),
                 role=("BEATS baseline" if phys < base - 0.5 else "matches baseline")),
            dict(model="+ condition (rate from Vc/fz/cooling)", mae_um=round(pinn, 1),
                 role=("condition helps" if pinn < phys - 0.5 else "condition does NOT help"))]
    R = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    R.to_csv(os.path.join(ROOT, "results", "model_comparison.csv"), index=False)
    print(f"Model comparison on current VB targets | {len(tools)} tools, LOTO, m={M}, "
          f"future-VB MAE (wear regime VB<={CENSOR:.0f}):\n")
    print(R.to_string(index=False))
    print(f"\nVERDICT: baseline {base:.1f} | physics wear law {phys:.1f} ({rows[1]['role']}) | "
          f"+condition {pinn:.1f} ({rows[2]['role']}).")


if __name__ == "__main__":
    main()
