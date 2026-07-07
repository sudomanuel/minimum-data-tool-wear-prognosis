"""run_r8b_augmentation_families.py — Round-8 B: two more augmentation generators under the same
fleet-size (K) x synthetic-multiplier (S) frontier harness as run_r8_physics_augmentation.

A2 MEGA-TREND-DIFFUSION / Virtual Sample Generation (Li, Wu & Chang 2007): the canonical
   small-sample manufacturing technique. Estimate diffused domain bounds [L,U] for each fleet
   parameter (log a, b) via the MTD diffusion function (skew-aware), sample virtual parameters
   under a triangular membership peaking at the central value, synthesize tools VB = b + a*t^p.
A3 GAMMA-PROCESS Monte-Carlo (seeded): monotone-by-construction degradation paths from a gamma
   process fitted to the fleet increments in the tau=t^p clock, with a per-tool random-effect on
   the rate; independent positive increments -> monotone.

Same protocol/rule as round 8: LOTO, sweep K in {4,6,9,13,17} x S in {0,1,3}, m in {2,3}, 6 seeds;
adopt only if a generator beats S=0 by >0.1 um in a monotone-in-K pattern. If neutral/harmful the
result does NOT touch the manuscript (user directive). Sanity K=17/S=0 ~ record.
Outputs: results/r8b_augmentation_families.csv
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
from run_r8_physics_augmentation import (curve_list, fit_pooled_p, fleet_law, predict_one)
CENSOR = 300.0
K_LIST = [4, 6, 9, 13, 17]
S_LIST = [0, 1, 3]
N_SEED = 6
RECORD = {2: 12.72, 3: 11.02}


# ---------------- A2: Mega-Trend-Diffusion ----------------
def mtd_bounds(x):
    x = np.asarray(x, float)
    CL = float(x.mean())
    NL = max(int((x <= CL).sum()), 1); NU = max(int((x > CL).sum()), 1)
    sL, sU = NL / (NL + NU), NU / (NL + NU)
    s2 = float(x.var(ddof=1)) if len(x) > 1 else 1.0
    span = np.sqrt(-2.0 * (s2) * np.log(1e-20))
    L = CL - sL * span / np.sqrt(NL)
    U = CL + sU * span / np.sqrt(NU)
    return L, U, CL


def mtd_sample(x, n, rng):
    L, U, CL = mtd_bounds(x)
    out = []
    guard = 0
    while len(out) < n and guard < 50 * n + 100:
        guard += 1
        u = rng.uniform(L, U)
        mf = (u - L) / (CL - L) if u <= CL else (U - u) / (U - CL)
        if rng.uniform() <= max(mf, 0.0):
            out.append(u)
    while len(out) < n:
        out.append(CL)
    return np.array(out[:n])


def mtd_tools(curves, p, law, n_syn, rng):
    las, bs = [], []
    for o, v in curves:
        a, b = theil_sen(o ** p, v)
        las.append(np.log(max(a, 1e-3))); bs.append(b)
    la_s = mtd_sample(np.array(las), n_syn, rng)
    b_s = mtd_sample(np.array(bs), n_syn, rng)
    sigma = law[4]
    orders = [o for o, _ in curves]
    out = []
    for i in range(n_syn):
        o = orders[rng.integers(len(orders))]
        v = b_s[i] + np.exp(la_s[i]) * o ** p + rng.normal(0, sigma, len(o))
        out.append((o, np.maximum.accumulate(v)))
    return out


# ---------------- A3: Gamma process ----------------
def gamma_tools(curves, p, law, n_syn, rng):
    dY, dT, etas, bs = [], [], [], []
    for o, v in curves:
        tau = o ** p
        d = np.diff(v); dt = np.diff(tau); ok = dt > 1e-9
        dY += list(np.clip(d[ok], 1e-6, None)); dT += list(dt[ok])
        etas.append((v[-1] - v[0]) / max(tau[-1] - tau[0], 1e-9)); bs.append(v[0])
    dY, dT = np.array(dY), np.array(dT)
    rate = dY / dT
    m, var = float(rate.mean()), float(rate.var()) + 1e-9
    theta = var / m                      # gamma scale
    shape_pu = m / theta                 # shape per unit tau
    eta_mu = max(np.mean(etas), 1e-6); eta_cv = np.std(etas) / eta_mu
    b_mu, b_sd = float(np.mean(bs)), max(float(np.std(bs)), 1e-3)
    orders = [o for o, _ in curves]
    out = []
    for _ in range(n_syn):
        o = orders[rng.integers(len(orders))]
        tau = o ** p
        mult = max(rng.normal(1.0, eta_cv), 0.1)     # per-tool random effect on rate
        v = [rng.normal(b_mu, b_sd)]
        for k in range(1, len(tau)):
            dt = tau[k] - tau[k - 1]
            v.append(v[-1] + rng.gamma(max(shape_pu * dt, 1e-3), theta * mult))
        out.append((o, np.array(v)))
    return out


GENERATORS = {"MTD_VSG": mtd_tools, "GAMMA_proc": gamma_tools}


def sweep(d, tools, gen, m, eb):
    grid = {}
    for K in K_LIST:
        for S in S_LIST:
            maes = []
            for seed in range(N_SEED):
                rng = np.random.default_rng(2000 * seed + K * 7 + S)
                per = []
                for tt in tools:
                    pool = [t for t in tools if t != tt]
                    sub = list(rng.choice(pool, size=min(K, len(pool)), replace=False))
                    real = curve_list(d, sub)
                    if len(real) < 2:
                        continue
                    p0 = fit_pooled_p(real); law = fleet_law(real, p0)
                    aug = real + (gen(real, p0, law, S * K, rng) if S > 0 else [])
                    p_star = fit_pooled_p(aug); a_pop = fleet_law(aug, p_star)[5]
                    g = d[d.tool_id == tt].sort_values("order")
                    o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
                    if len(o) <= m:
                        continue
                    e = predict_one(o, v, m, p_star, a_pop, eb)
                    if e is not None:
                        per.append(e)
                if per:
                    maes.append(np.mean(per))
            grid[(K, S)] = float(np.mean(maes))
    return grid


def main():
    d = load(); tools = tools_of(d)
    print("R8-B: MTD/VSG + GAMMA-process augmentation under the frontier harness "
          f"({N_SEED} seeds; sanity K=17/S=0 ~ record).\n")
    rows = []
    any_win = False
    for gname, gen in GENERATORS.items():
        print(f"################## generator: {gname} ##################")
        for m in (2, 3):
            eb = 0.2 if m == 2 else 0.0
            grid = sweep(d, tools, gen, m, eb)
            print(f"=== {gname} m={m} (record {RECORD[m]}) ===")
            print(f"{'K\\\\S':>5} " + " ".join(f"S={s:<5}" for s in S_LIST))
            for K in K_LIST:
                base = grid[(K, 0)]
                cells = []
                for S in S_LIST:
                    val = grid[(K, S)]
                    mk = "" if S == 0 else ("↓" if val < base - 0.05 else ("↑" if val > base + 0.05 else "="))
                    cells.append(f"{val:5.2f}{mk}")
                    rows.append(dict(generator=gname, m=m, K=K, S=S, MAE=round(val, 2)))
                print(f"{K:>5} " + " ".join(f"{c:<7}" for c in cells))
            # frontier verdict for this generator/m
            for K in K_LIST:
                g = {r["S"]: r["MAE"] for r in rows if r["generator"] == gname and r["m"] == m and r["K"] == K}
                bs = min(g, key=g.get); gain = g[0] - g[bs]
                if gain > 0.1 and bs != 0:
                    any_win = True
            print()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "r8b_augmentation_families.csv"), index=False)
    print("=" * 60)
    if any_win:
        print("SOME (K,S) beats S=0 by >0.1 um — inspect for a MONOTONE-in-K frontier before any claim.")
        for gname in GENERATORS:
            for m in (2, 3):
                for K in K_LIST:
                    g = {r["S"]: r["MAE"] for r in rows if r["generator"] == gname and r["m"] == m and r["K"] == K}
                    bs = min(g, key=g.get)
                    if g[0] - g[bs] > 0.1 and bs != 0:
                        print(f"  {gname} m={m} K={K}: S=0 {g[0]:.2f} -> S={bs} {g[bs]:.2f} (-{g[0]-g[bs]:.2f})")
    else:
        print("VERDICT: neither MTD/VSG nor the gamma process produces a usable augmentation frontier — "
              "no (K,S) beats S=0 by >0.1 um. Consistent with round-8 A1 and 'replication != "
              "augmentation'. Per user directive, this does NOT touch the manuscript in any way.")
    print("wrote results/r8b_augmentation_families.csv")


if __name__ == "__main__":
    main()
