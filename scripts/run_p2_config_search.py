# -*- coding: utf-8 -*-
"""run_p2_config_search.py — PAPER 2 · joint configuration search, Paper-1 discipline.

The incumbent record of 5.6 um was NOT a default: it came from a joint, validity-constrained
search over 120 configurations (Sec. 3.8 of Pusma et al., 2025).  This script grants the
continuous formulation the SAME procedure, so the comparison is symmetric:

  factors      lam_phys in {0.5, 1.0, 2.0} x beta in {0.0, 0.30} x gamma in {2, 3, 4}
  eligibility  a configuration is admissible only if its conformal coverage stays >= 88%
  selection    stage 1 reports the in-search optimum (optimistic, declared as such)
               stage 2 re-runs the ENTIRE selection blind per fold (nested double LOOCV) and
               reports the honest number — exactly the check the foundational study applied
               to itself.

Fields are checkpointed per (fold, lam, beta) so the run is resumable.
Outputs: results/p2_config_search.csv, results/p2_nested.csv
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import run_p2_pinode as P
from run_optimal_config_search import fit_p

LAMS = (0.5, 1.0, 2.0)
BETAS = (0.0, 0.30)
GAMMAS = (2.0, 3.0, 4.0)
MIN_COVERAGE = 88.0          # validity constraint inherited from Paper 1
CKPT = os.path.join(ROOT, "results", "p2_fields")
os.makedirs(CKPT, exist_ok=True)


def cfg_key(lam, beta):
    return f"cfg_l{lam}_b{beta}"


def field_for(t, d, tools, p_star, lam, beta):
    """Checkpointed field for a (fold, lam, beta) cell."""
    return P.get_field(t, d, tools, p_star, cfg_key(lam, beta), beta, lam, 0.1, 0.05, 0)


def evaluate(d, tools, m, lam, beta, gamma, p_by_fold, subset=None):
    """LOOCV over `subset` (default: all tools). Returns per-tool errors and residual/delta lists."""
    errs, resid, delta = [], [], []
    for t in (subset or tools):
        o, v, z = P.tool_arrays(d, t)
        if len(o) <= m:
            continue
        fld = field_for(t, d, tools, p_by_fold[t], lam, beta)
        if fld is None:
            continue
        par = P.personalise(fld, o, v, z, m, p_by_fold[t], gamma=gamma)
        e = np.abs(P.forecast(fld, o, v, z, m, par)[m:] - v[m:])
        errs.append(float(np.mean(e)))
        resid.extend(e.tolist())
        delta.extend((o[m:] - o[m - 1]).tolist())
    return np.array(errs), resid, delta


def coverage_of(resid, delta):
    if not resid:
        return float("nan")
    q = P.mondrian_bands(resid, delta)
    return 100.0 * float(np.mean([r <= q[P.bin_of(dl)] for r, dl in zip(resid, delta)]))


def pretrain(budget_n):
    d = P.load_all(); tools = P.tools_of(d)
    todo = [(t, l, b) for t in tools for l in LAMS for b in BETAS
            if not os.path.exists(os.path.join(CKPT, f"{t}_{cfg_key(l,b)}_s0.pt"))]
    print(f"missing config fields: {len(todo)} of {len(tools)*len(LAMS)*len(BETAS)} — "
          f"training up to {budget_n}", flush=True)
    for i, (t, l, b) in enumerate(todo[:budget_n], 1):
        field_for(t, d, tools, fit_p(d[d.tool_id != t]), l, b)
        print(f"  [{i}/{min(budget_n,len(todo))}] {t} lam={l} beta={b} checkpointed", flush=True)
    print(f"done. remaining: {len(todo)-min(budget_n,len(todo))}", flush=True)


def search(m=4):
    d = P.load_all(); tools = P.tools_of(d)
    p_by_fold = {t: fit_p(d[d.tool_id != t]) for t in tools}

    # ---------- stage 1: in-search optimum (optimistic — declared) ----------
    rows = []
    for lam in LAMS:
        for beta in BETAS:
            for gamma in GAMMAS:
                errs, resid, delta = evaluate(d, tools, m, lam, beta, gamma, p_by_fold)
                cov = coverage_of(resid, delta)
                rows.append(dict(m=m, lam_phys=lam, beta=beta, gamma=gamma,
                                 MAE=round(float(errs.mean()), 3),
                                 MAE_median=round(float(np.median(errs)), 3),
                                 coverage=round(cov, 1),
                                 eligible=bool(cov >= MIN_COVERAGE), n=len(errs)))
                print(f"  lam={lam:<4} beta={beta:<5} gamma={gamma:<4} -> MAE {errs.mean():6.3f} "
                      f"cov {cov:5.1f}% {'OK' if cov >= MIN_COVERAGE else 'INVALID'}", flush=True)
    tab = pd.DataFrame(rows).sort_values("MAE")
    tab.to_csv(os.path.join(ROOT, "results", "p2_config_search.csv"), index=False)
    elig = tab[tab.eligible]
    best = elig.iloc[0] if len(elig) else tab.iloc[0]
    print(f"\nIN-SEARCH OPTIMUM (m={m}): lam={best.lam_phys} beta={best.beta} gamma={best.gamma} "
          f"-> {best.MAE} um (coverage {best.coverage}%)", flush=True)

    # ---------- stage 2: nested double LOOCV (the honest number) ----------
    print("\nnested check — re-selecting the configuration BLIND for every outer fold:", flush=True)
    nested_err, winners = [], []
    for t in tools:
        o, v, z = P.tool_arrays(d, t)
        if len(o) <= m:
            continue
        inner = [x for x in tools if x != t]
        best_cfg, best_score = None, np.inf
        for lam in LAMS:
            for beta in BETAS:
                for gamma in GAMMAS:
                    e, rs, dl = evaluate(d, tools, m, lam, beta, gamma, p_by_fold, subset=inner)
                    if len(e) == 0:
                        continue
                    if coverage_of(rs, dl) < MIN_COVERAGE:
                        continue
                    if e.mean() < best_score:
                        best_score, best_cfg = e.mean(), (lam, beta, gamma)
        if best_cfg is None:
            best_cfg = (1.0, 0.30, 3.0)
        lam, beta, gamma = best_cfg
        fld = field_for(t, d, tools, p_by_fold[t], lam, beta)
        par = P.personalise(fld, o, v, z, m, p_by_fold[t], gamma=gamma)
        err = float(np.mean(np.abs(P.forecast(fld, o, v, z, m, par)[m:] - v[m:])))
        nested_err.append(err); winners.append(best_cfg)
        print(f"  outer {t}: winner lam={lam} beta={beta} gamma={gamma} -> {err:6.2f} um", flush=True)

    nested = float(np.mean(nested_err))
    from collections import Counter
    share = Counter(winners).most_common(1)[0]
    print(f"\nNESTED MAE (m={m}) = {nested:.2f} um  (median {np.median(nested_err):.2f}, "
          f"n={len(nested_err)})", flush=True)
    print(f"most frequent blind winner: {share[0]} in {share[1]}/{len(winners)} folds", flush=True)
    print(f"selection optimism = {nested - best.MAE:+.2f} um", flush=True)
    pd.DataFrame(dict(tool=[t for t in tools if len(P.tool_arrays(d, t)[0]) > m],
                      nested_MAE=np.round(nested_err, 3),
                      winner=[str(w) for w in winners])).to_csv(
        os.path.join(ROOT, "results", "p2_nested.csv"), index=False)
    return tab, nested


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-only", type=int, default=0)
    ap.add_argument("--m", type=int, default=4)
    a = ap.parse_args()
    print("PAPER 2 · joint configuration search (Paper-1 discipline)\n" + "=" * 60)
    if a.pretrain_only:
        pretrain(a.pretrain_only)
    else:
        search(a.m)
