"""run_r4_condition_p.py — Round-4 Module 1 adjudication: condition-parameterized wear exponent.

Claim under test (consortium round 4): a single fleet exponent p ≈ 0.20 "violates tribology"; p should
be a function of dimensionless process variables. True Peclet needs thermal properties not instrumented
in this campaign, so the testable surrogates are (i) the full kinematic map p(ln vc, ln fz, cooling)
and (ii) a scalar kinematic-severity index p(ln(vc·fz)).

Protocol: leakage-safe LOTO. Per fold, each TRAINING tool contributes its own best full-trajectory
exponent p̂_k (grid argmin SSE, Theil-Sen at each candidate p); the exponent model is regressed on the
training tools only and predicts the held-out tool's p from ITS CONDITION (zero-shot exponent). The
few-shot fit then uses that exponent (directly, blended 50/50 with the fleet value, or as the center of
the local-shrinkage grid of the record configuration).

Pre-stated rule: adopt only if a variant beats the record at the same m (m=3: 11.02, m=4: 5.63).
Outputs: results/r4_condition_p.csv
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load, theil_sen, tools_of
from run_optimal_config_search import fit_p, FIT
CENSOR = 300.0
FEAT = os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv")
RECORDS = {3: 11.02, 4: 5.63}
BASE = {3: 11.57, 4: 9.67}
P_GRID = np.arange(0.05, 0.6001, 0.01)


def tool_conditions():
    f = pd.read_csv(FEAT)
    g = f.groupby("tool_id").first()
    cool = (g["cooling"].astype(str).str.lower()
            .map(lambda s: 0.0 if ("dry" in s or "no" in s) else 1.0))
    return pd.DataFrame(dict(vc=g.vc.astype(float), fz=g.fz.astype(float), cool=cool))


def per_tool_p(g):
    """Best full-trajectory exponent of one tool (grid argmin SSE, Theil-Sen at each p)."""
    gg = g[g.vb <= CENSOR].sort_values("order")
    o, v = gg.order.to_numpy(float), gg.vb.to_numpy(float)
    if len(o) < 3:
        return None
    best_p, best = None, np.inf
    for p in P_GRID:
        tau = o ** p
        a, b = theil_sen(tau, v)
        sse = float(np.sum((b + a * tau - v) ** 2))
        if sse < best:
            best, best_p = sse, p
    return best_p


def p_regression(tr, cond, kind):
    """Fit exponent model on training tools; return predict(tool_id) and in-sample R²."""
    ps = {}
    for tt, g in tr.groupby("tool_id"):
        pk = per_tool_p(g)
        if pk is not None:
            ps[tt] = pk
    ids = [t for t in ps if t in cond.index]
    y = np.array([ps[t] for t in ids])
    if kind == "kinematic":            # full condition map
        X = np.column_stack([np.ones(len(ids)), np.log(cond.loc[ids, "vc"]),
                             np.log(cond.loc[ids, "fz"]), cond.loc[ids, "cool"]])
        feats = lambda t: np.array([1.0, np.log(cond.loc[t, "vc"]),
                                    np.log(cond.loc[t, "fz"]), cond.loc[t, "cool"]])
    else:                              # scalar severity index
        s = np.log(cond.loc[ids, "vc"] * cond.loc[ids, "fz"])
        X = np.column_stack([np.ones(len(ids)), s])
        feats = lambda t: np.array([1.0, np.log(cond.loc[t, "vc"] * cond.loc[t, "fz"])])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((X @ c - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return (lambda t: float(np.clip(feats(t) @ c, 0.05, 0.6))), float(r2), y


def local_p_grid(o, v, m, p_star, fitfn, lam=200.0):
    best_p, best = p_star, np.inf
    for pc in np.arange(max(p_star - 0.15, 0.05), p_star + 0.1501, 0.05):
        tau = o[:m] ** pc
        a, b = theil_sen(tau, v[:m])
        sse = float(np.sum((b + a * tau - v[:m]) ** 2)) + lam * (pc - p_star) ** 2
        if sse < best:
            best, best_p = sse, pc
    return best_p


def wls_gamma(tau, y, gamma=3.0):
    w = np.maximum(tau, 1e-9) ** gamma
    W = np.sqrt(w)
    A = np.column_stack([W, W * tau])
    c, *_ = np.linalg.lstsq(A, W * y, rcond=None)
    return float(c[1]), float(c[0])


def evaluate(d, cond, m, p_source, use_local=True):
    """p_source(tr, tt) -> exponent center for the held-out tool. Record fit per m."""
    fitfn = FIT["siegel"] if m == 3 else (lambda t, y: wls_gamma(t, y, 3.0))
    per, r2s = [], []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p_star, r2 = p_source(tr, tt)
        r2s.append(r2)
        p = local_p_grid(o, v, m, p_star, fitfn) if use_local else p_star
        tau = o[:m] ** p
        a, b = fitfn(tau, v[:m])
        per.append(np.abs(b + a * o[fut] ** p - v[fut]).mean())
    return float(np.mean(per)), (float(np.mean(r2s)) if r2s else float("nan"))


def main():
    d = load(); cond = tool_conditions()
    print("R4-M1: CONDITION-PARAMETERIZED EXPONENT under LOTO — adopt only if a variant beats the record.")
    print(f"reference: base m=3 {BASE[3]} / m=4 {BASE[4]}  |  records m=3 {RECORDS[3]} / m=4 {RECORDS[4]}\n")

    # descriptive: dispersion of per-tool exponents and in-sample fit of the condition map (full fleet)
    all_p = {tt: per_tool_p(g) for tt, g in d.groupby("tool_id")}
    pv = np.array([p for p in all_p.values() if p is not None])
    print(f"per-tool full-trajectory exponents: min {pv.min():.2f}  median {np.median(pv):.2f}  "
          f"max {pv.max():.2f}  (spread is REAL — the question is whether the CONDITION explains it)")
    for kind in ("kinematic", "severity"):
        _, r2, _ = p_regression(d, cond, kind)
        print(f"  in-sample R² of p ~ {kind}: {r2:+.2f} (18 tools, descriptive)")
    print()

    rows = []
    # p sources per fold
    def src_global(mm):
        def s(tr, tt):
            return fit_p(tr, m=(mm if mm == 4 else None)), float("nan")
        return s

    def src_cond(kind):
        def s(tr, tt):
            pred, r2, _ = p_regression(tr, cond, kind)
            return pred(tt), r2
        return s

    def src_blend(kind, mm, w=0.5):
        def s(tr, tt):
            pred, r2, _ = p_regression(tr, cond, kind)
            pg = fit_p(tr, m=(mm if mm == 4 else None))
            return (1 - w) * pg + w * pred(tt), r2
        return s

    for m in (3, 4):
        v, _ = evaluate(d, cond, m, src_global(m))
        rows.append(dict(m=m, p_source="fleet global (record cfg, sanity)", MAE=round(v, 2)))
        print(f"  m={m} fleet global (sanity)          : {v:6.2f}  (record {RECORDS[m]})")
        for kind in ("kinematic", "severity"):
            v, r2 = evaluate(d, cond, m, src_cond(kind))
            rows.append(dict(m=m, p_source=f"condition {kind} zero-shot", MAE=round(v, 2)))
            print(f"  m={m} condition {kind:9} zero-shot : {v:6.2f}  (LOTO mean in-sample R² {r2:+.2f})")
            v, _ = evaluate(d, cond, m, src_blend(kind, m))
            rows.append(dict(m=m, p_source=f"blend 0.5 fleet + 0.5 {kind}", MAE=round(v, 2)))
            print(f"  m={m} blend 0.5 fleet/{kind:9}    : {v:6.2f}")
        print()

    df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, "results", "r4_condition_p.csv"), index=False)
    winners = [(r["m"], r["p_source"], r["MAE"]) for _, r in df.iterrows()
               if "sanity" not in r["p_source"] and r["MAE"] < RECORDS[r["m"]] - 0.05]
    if winners:
        print("WINNERS (beat the record):", winners)
    else:
        print("VERDICT: no condition-parameterized exponent beats the record — the condition-to-shape map")
        print("is not identifiable with one tool per condition; the local-shrinkage mechanism (already")
        print("deployed) is the correct amount of per-tool shape freedom. Declare p(x) as replicated-")
        print("campaign protocol, not as fitted machinery.")
    print("\nwrote results/r4_condition_p.csv")


if __name__ == "__main__":
    main()
