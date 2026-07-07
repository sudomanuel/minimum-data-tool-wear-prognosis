"""run_e2_phm2010_hybrid.py — follow-up to the external check: can the FRAMEWORK (not the fixed
concave law) close the gap on PHM2010, i.e. in the replicated regime?

The external check showed the fleet average-curve wins on PHM2010 (15.7 um) because (i) replicates
of one condition make the population S-shape learnable and (ii) the concave power law cannot express
the tertiary acceleration those curves run through. Both are addressable INSIDE the framework:

  HYBRID-1 affine fleet ("fleet shape + few-shot personalization"): per held-out tool, fit
           VB ~ alpha*pop(t) + beta on the m early inspections (population shape from the training
           cutters, level+scale personalized few-shot). This is the paper's philosophy applied to a
           replicated fleet.
  HYBRID-2 EB blend: w*fleet + (1-w)*fewshot, w selected by inner LOTO on the training cutters only.
  HYBRID-3 two-phase law with fleet-fitted tertiary: the kappa-continuation of Sec 3.3 becomes
           IDENTIFIABLE here (curves run through tertiary); (VB_T, kappa) fitted on the training
           cutters' own few-shot-to-future error, then applied to the held-out tool.

Same protocol as run_e (LOTO over 3 cutters, K=10 cadence, m=3/4, report-as-comes). Exploratory:
none of this touches the manuscript without a separate decision.
Outputs: results/e2_phm2010_hybrid.csv
"""
import os, sys, itertools
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_e_phm2010_external import (theil_sen, siegel, wls_gamma, fit_global_p, local_p_grid,
                                    subsample, DATA)
K = 10


def fewshot_pred(tr_curves, o, v, m):
    if m == 3:
        p = local_p_grid(o, v, m, fit_global_p(tr_curves))
        a, b = siegel(o[:m] ** p, v[:m])
    else:
        p = local_p_grid(o, v, m, fit_global_p(tr_curves, m=m))
        a, b = wls_gamma(o[:m] ** p, v[:m], 3.0)
    return lambda oo: b + a * oo ** p, p, a, b


def pop_curve(tr_curves):
    L = min(len(t[1]) for t in tr_curves)
    return np.mean([t[1][:L] for t in tr_curves], axis=0)


def model_avg(tr_curves, o, v, m, fut):
    pop = pop_curve(tr_curves)
    off = float(np.mean(v[:m] - pop[:m]))
    return pop[fut] + off


def model_affine(tr_curves, o, v, m, fut):
    pop = pop_curve(tr_curves)
    X = np.column_stack([pop[:m], np.ones(m)])
    c, *_ = np.linalg.lstsq(X, v[:m], rcond=None)
    alpha = max(float(c[0]), 0.0)
    return alpha * pop[fut] + float(c[1])


def model_fewshot(tr_curves, o, v, m, fut):
    f, *_ = fewshot_pred(tr_curves, o, v, m)
    return f(o[fut])


def model_blend(tr_curves, names_tr, curves, o, v, m, fut):
    """w selected by inner LOTO on the 2 training cutters (grid)."""
    best_w, best = 1.0, np.inf
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        errs = []
        for i, u in enumerate(names_tr):
            inner_tr = [curves[x] for x in names_tr if x != u]
            if len(inner_tr) < 1:
                continue
            oo, vv = curves[u]
            ff = np.arange(m, len(oo))
            pr_f = model_fewshot(inner_tr * 2, oo, vv, m, ff)   # duplicate to satisfy pop mean
            pr_a = model_avg(inner_tr * 2, oo, vv, m, ff)
            errs.append(np.mean(np.abs(w * pr_a + (1 - w) * pr_f - vv[ff])))
        e = float(np.mean(errs))
        if e < best:
            best, best_w = e, w
    pr_f = model_fewshot(tr_curves, o, v, m, fut)
    pr_a = model_avg(tr_curves, o, v, m, fut)
    return best_w * pr_a + (1 - best_w) * pr_f, best_w


def twophase_apply(f, p, a, b, oo, vbT, kappa):
    """Continuation: beyond the time tT where the fitted curve reaches vbT, add quadratic accel."""
    base = f(oo)
    if a <= 0 or vbT <= b:
        return base
    tT = ((vbT - b) / a) ** (1.0 / p)
    rT = a * p * tT ** (p - 1)          # dVB/dt at transition
    out = base.copy()
    beyond = oo > tT
    dt = oo[beyond] - tT
    out[beyond] = vbT + rT * dt + 0.5 * kappa * rT * dt ** 2
    return out


def model_twophase(tr_curves, names_tr, curves, o, v, m, fut):
    """(VB_T, kappa) fitted on the TRAINING cutters' own few-shot->future error (inner, honest)."""
    grid = list(itertools.product((80, 100, 120, 140), (0.0, 0.02, 0.05, 0.1, 0.2)))
    best, best_cfg = np.inf, (120, 0.0)
    for vbT, kp in grid:
        errs = []
        for u in names_tr:
            inner_tr = [curves[x] for x in names_tr if x != u] * 2
            oo, vv = curves[u]
            ff = np.arange(m, len(oo))
            f, p, a, b = fewshot_pred(inner_tr, oo, vv, m)
            pr = twophase_apply(f, p, a, b, oo[ff], vbT, kp)
            errs.append(np.mean(np.abs(pr - vv[ff])))
        e = float(np.mean(errs))
        if e < best:
            best, best_cfg = e, (vbT, kp)
    f, p, a, b = fewshot_pred(tr_curves, o, v, m)
    return twophase_apply(f, p, a, b, o[fut], *best_cfg), best_cfg


def main():
    df = pd.read_csv(DATA)
    curves = subsample(df, K)
    names = sorted(curves)
    print("E2: HYBRIDS ON PHM2010 — can the framework close the replicated-regime gap? "
          "(LOTO, K=10, report-as-comes)\n")
    rows = []
    for m in (3, 4):
        per = {mod: [] for mod in ("fewshot", "avgcurve", "affine", "blend", "twophase")}
        extras = {"blend": [], "twophase": []}
        for held in names:
            names_tr = [c for c in names if c != held]
            tr = [curves[c] for c in names_tr]
            o, v = curves[held]
            fut = np.arange(m, len(o))
            per["fewshot"].append(np.mean(np.abs(model_fewshot(tr, o, v, m, fut) - v[fut])))
            per["avgcurve"].append(np.mean(np.abs(model_avg(tr, o, v, m, fut) - v[fut])))
            per["affine"].append(np.mean(np.abs(model_affine(tr, o, v, m, fut) - v[fut])))
            prb, w = model_blend(tr, names_tr, curves, o, v, m, fut)
            per["blend"].append(np.mean(np.abs(prb - v[fut]))); extras["blend"].append(w)
            prt, cfg = model_twophase(tr, names_tr, curves, o, v, m, fut)
            per["twophase"].append(np.mean(np.abs(prt - v[fut]))); extras["twophase"].append(cfg)
        print(f"m={m}:")
        for mod in per:
            mae = float(np.mean(per[mod]))
            note = ""
            if mod == "blend":
                note = f"  (w elegidos: {extras['blend']})"
            if mod == "twophase":
                note = f"  ((VB_T,κ) elegidos: {extras['twophase']})"
            print(f"  {mod:9}: {mae:6.2f} um{note}")
            rows.append(dict(m=m, model=mod, MAE=round(mae, 2)))
        print()
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "e2_phm2010_hybrid.csv"), index=False)
    print("wrote results/e2_phm2010_hybrid.csv")


if __name__ == "__main__":
    main()
