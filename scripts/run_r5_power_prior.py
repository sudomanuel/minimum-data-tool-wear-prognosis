"""run_r5_power_prior.py — Round-5 A3: the EB rate-shrinkage formalized as a POWER PRIOR, plus its
adaptive (commensurate) upgrade.

Theory (the derivation the round asked for): the deployed blend a_post = (1-lambda)*a_tool +
lambda*a_fleet is the posterior mean of a Gaussian model in which the FLEET likelihood enters raised
to a power delta (Ibrahim & Chen's power prior), with lambda = delta*prec_fleet / (prec_tool +
delta*prec_fleet). Fixed lambda = fixed delta. The commensurate upgrade (Hobbs et al.) sets delta
PER TOOL from the fleet-tool conflict: delta_u = 1/(1 + z_u^2), z_u = (log a_u - mu_fleet)/s_fleet —
a tool whose early slope disagrees with the fleet borrows less.

Test (pre-stated): m=2 and m=3 (records 12.72 / 11.02). Variants: fixed-delta grid (sanity: must
reproduce the lambda grid), commensurate delta (adaptive), commensurate with floor. LOTO throughout.
Adopt only if a variant beats the record at the same m.
Outputs: results/r5_power_prior.csv
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
RECORDS = {2: 12.72, 3: 11.02}


def local_p_grid(o, v, m, p_star, lam=200.0):
    best_p, best = p_star, np.inf
    for pc in np.arange(max(p_star - 0.15, 0.05), p_star + 0.1501, 0.05):
        tau = o[:m] ** pc
        a, b = theil_sen(tau, v[:m])
        sse = float(np.sum((b + a * tau - v[:m]) ** 2)) + lam * (pc - p_star) ** 2
        if sse < best:
            best, best_p = sse, pc
    return best_p


def fleet_rates(tr, p):
    out = []
    for _, g in tr.groupby("tool_id"):
        gg = g[g.vb <= CENSOR].sort_values("order")
        if len(gg) >= 2:
            a, _ = theil_sen(gg.order.to_numpy(float) ** p, gg.vb.to_numpy(float))
            out.append(max(a, 1e-3))
    return np.array(out)


def evaluate(d, m, mode, delta_fixed=None):
    """mode in {'fixed','commensurate','commensurate_floor'}; m=3 uses record cfg (siegel+local p)."""
    per = []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = fit_p(tr)
        if m == 3:
            p = local_p_grid(o, v, m, p)
        fitfn = FIT["siegel"] if m == 3 else FIT["theil_sen"]
        tau = o[:m] ** p
        a_tool, b = fitfn(tau, v[:m])
        rates = fleet_rates(tr, p)
        la = np.log(rates)
        mu, s = float(la.mean()), max(float(la.std(ddof=1)), 1e-6)
        a_t = max(a_tool, 1e-3)
        # precision of the tool's log-rate estimate from its m points: rough via pairwise-slope spread
        if m >= 3:
            sl = [(v[j] - v[i]) / (tau[j] - tau[i]) for i in range(m) for j in range(i + 1, m)
                  if tau[j] != tau[i]]
            s_tool = max(np.std(np.log(np.clip(sl, 1e-3, None)), ddof=0), 1e-3)
        else:
            s_tool = s  # at m=2 a single slope: tool precision ~ fleet spread
        prec_t, prec_f = 1 / s_tool ** 2, 1 / s ** 2
        z = (np.log(a_t) - mu) / s
        if mode == "fixed":
            delta = delta_fixed
        elif mode == "commensurate":
            delta = 1.0 / (1.0 + z ** 2)
        else:
            delta = max(1.0 / (1.0 + z ** 2), 0.2)
        w = delta * prec_f / (prec_t + delta * prec_f)      # power-prior posterior weight on fleet
        log_a = (1 - w) * np.log(a_t) + w * mu
        a = float(np.exp(log_a))
        b = float(np.median(v[:m] - a * tau))
        per.append(np.abs(b + a * o[fut] ** p - v[fut]).mean())
    return float(np.mean(per))


def main():
    d = load()
    print("R5-A3: POWER PRIOR / COMMENSURATE — the EB blend formalized; adopt only if beats record.\n")
    rows = []
    for m in (2, 3):
        print(f"--- m={m} (record {RECORDS[m]}) ---")
        for dv in (0.0, 0.1, 0.25, 0.5, 1.0):
            v = evaluate(d, m, "fixed", dv)
            rows.append(dict(m=m, variant=f"fixed delta={dv}", MAE=round(v, 2)))
            print(f"  fixed  delta={dv:4}: {v:6.2f}")
        for mode, tag in (("commensurate", "adaptive 1/(1+z^2)"), ("commensurate_floor", "adaptive, floor 0.2")):
            v = evaluate(d, m, mode)
            rows.append(dict(m=m, variant=tag, MAE=round(v, 2)))
            print(f"  {tag:20}: {v:6.2f}")
        print()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "r5_power_prior.csv"), index=False)
    winners = [(r["m"], r["variant"], r["MAE"]) for _, r in df.iterrows()
               if r["MAE"] < RECORDS[r["m"]] - 0.05]
    print("WINNERS:", winners if winners else "none — records stand; the value is the THEORY "
          "(deployed EB = power prior with fixed delta), not a new number.")
    print("wrote results/r5_power_prior.csv")


if __name__ == "__main__":
    main()
