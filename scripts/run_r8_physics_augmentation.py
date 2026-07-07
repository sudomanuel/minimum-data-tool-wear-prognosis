"""run_r8_physics_augmentation.py — Round-8 A1: the AUGMENTATION FRONTIER.

Physics-based Monte-Carlo augmentation exactly as requested (wear equation + seeds): sample per-tool
parameters from the fitted FLEET hierarchical law and synthesize extra training tools via
    VB_syn(t) = b + a * t^p + eps,  a=exp(la), eps ~ N(0, sigma_meas^2), monotone-enforced,
then add them to the training fleet and re-estimate the fleet quantities the few-shot fit consumes
(pooled exponent p*, and for m=2 the population rate used by the EB/power-prior shrinkage).

The scientific question is NOT 'does augmentation help at full fleet' (already answered: no, jitter
12.1 vs 11.6) but 'below how many REAL tools does physics augmentation stop helping?' — a FRONTIER
over K (fleet size) x S (synthetic multiplier), the K-axis of the project's central minimum-data
frame. Because synthetic tools are drawn FROM the K real tools' fitted law, they carry no information
beyond those K tools; any gain is finite-sample variance reduction of the fleet estimate, any loss is
bias. We report whichever it is.

Protocol: LOTO over 18 tools; per held-out tool, subsample K of the other 17 as the real fleet,
augment with S*K synthetic tools, predict at m in {2,3} with the deployed config. Averaged over
N_SEED subsample/synthesis draws (fixed seeds -> reproducible). Sanity: K=17, S=0 reproduces the
deployed record. Outputs: results/r8_physics_augmentation.csv
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
from run_optimal_config_search import FIT
CENSOR = 300.0
P_GRID = np.arange(0.05, 0.6001, 0.02)
K_LIST = [4, 6, 9, 13, 17]
S_LIST = [0, 1, 3, 10]
N_SEED = 8
RECORD = {2: 12.72, 3: 11.02}


def curve_list(d, tool_ids):
    out = []
    for tt in tool_ids:
        g = d[d.tool_id == tt].sort_values("order")
        gg = g[g.vb <= CENSOR]
        o, v = gg.order.to_numpy(float), gg.vb.to_numpy(float)
        if len(o) >= 3:
            out.append((o, v))
    return out


def fit_pooled_p(curves):
    best_p, best = 0.2, np.inf
    for p in P_GRID:
        sse = 0.0
        for o, v in curves:
            a, b = theil_sen(o ** p, v)
            sse += float(np.sum((b + a * o ** p - v) ** 2))
        if sse < best:
            best, best_p = sse, p
    return best_p


def fleet_law(curves, p):
    las, bs, res = [], [], []
    for o, v in curves:
        a, b = theil_sen(o ** p, v)
        las.append(np.log(max(a, 1e-3))); bs.append(b)
        res += list(v - (b + a * o ** p))
    return (float(np.mean(las)), max(float(np.std(las, ddof=1)), 1e-3),
            float(np.mean(bs)), max(float(np.std(bs, ddof=1)), 1e-3),
            max(float(np.std(res, ddof=1)), 1e-6),
            float(np.median(np.exp(las))))


def synth_tools(curves, p, law, n_syn, rng):
    mu_la, s_la, mu_b, s_b, sigma, _ = law
    orders_pool = [o for o, _ in curves]
    out = []
    for _ in range(n_syn):
        o = orders_pool[rng.integers(len(orders_pool))].copy()
        a = np.exp(rng.normal(mu_la, s_la)); b = rng.normal(mu_b, s_b)
        v = b + a * o ** p + rng.normal(0, sigma, len(o))
        v = np.maximum.accumulate(v)          # monotone by construction
        out.append((o, v))
    return out


def local_p_grid(o, v, m, p_star, lam=200.0):
    best_p, best = p_star, np.inf
    for pc in np.arange(max(p_star - 0.15, 0.05), p_star + 0.1501, 0.05):
        tau = o[:m] ** pc
        a, b = theil_sen(tau, v[:m])
        sse = float(np.sum((b + a * tau - v[:m]) ** 2)) + lam * (pc - p_star) ** 2
        if sse < best:
            best, best_p = sse, pc
    return best_p


def predict_one(o, v, m, p_star, a_pop, eb):
    fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
    if len(fut) == 0:
        return None
    fitfn = FIT["theil_sen"] if m == 2 else FIT["siegel"]
    p = local_p_grid(o, v, m, p_star) if m == 3 else p_star
    tau = o[:m] ** p
    a, b = fitfn(tau, v[:m])
    if eb > 0 and a_pop is not None:
        a = (1 - eb) * a + eb * a_pop
        b = float(np.median(v[:m] - a * tau))
    return float(np.mean(np.abs(b + a * o[fut] ** p - v[fut])))


def main():
    d = load(); tools = tools_of(d)
    print("R8-A1: AUGMENTATION FRONTIER — physics Monte-Carlo (VB=b+a·t^p+ε, seeded), LOTO.")
    print(f"K (real fleet) x S (synthetic multiplier); {N_SEED} seeds; sanity K=17,S=0 ~ record.\n")
    rows = []
    for m in (2, 3):
        eb = 0.2 if m == 2 else 0.0
        print(f"=== m={m} (record {RECORD[m]}, EB={'on' if eb else 'off'}) ===")
        print(f"{'K\\\\S':>5} " + " ".join(f"S={s:<5}" for s in S_LIST))
        grid = {}
        for K in K_LIST:
            for S in S_LIST:
                maes = []
                for seed in range(N_SEED):
                    rng = np.random.default_rng(1000 * seed + K * 7 + S)
                    per = []
                    for tt in tools:
                        pool = [t for t in tools if t != tt]
                        sub = list(rng.choice(pool, size=min(K, len(pool)), replace=False))
                        real = curve_list(d, sub)
                        if len(real) < 2:
                            continue
                        p0 = fit_pooled_p(real)
                        law = fleet_law(real, p0)
                        aug = real + (synth_tools(real, p0, law, S * K, rng) if S > 0 else [])
                        p_star = fit_pooled_p(aug)
                        a_pop = fleet_law(aug, p_star)[5]
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
        for K in K_LIST:
            base = grid[(K, 0)]
            cells = []
            for S in S_LIST:
                v = grid[(K, S)]
                mark = "" if S == 0 else ("↓" if v < base - 0.05 else ("↑" if v > base + 0.05 else "="))
                cells.append(f"{v:5.2f}{mark:1}")
                rows.append(dict(m=m, K=K, S=S, MAE=round(v, 2)))
            print(f"{K:>5} " + " ".join(f"{c:<7}" for c in cells))
        print()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "r8_physics_augmentation.csv"), index=False)
    # frontier reading
    print("FRONTIER READING (per m, per K: does the best S beat S=0 by >0.1 um?):")
    for m in (2, 3):
        for K in K_LIST:
            g = {r["S"]: r["MAE"] for r in rows if r["m"] == m and r["K"] == K}
            best_s = min(g, key=g.get)
            gain = g[0] - g[best_s]
            verdict = f"AUGMENTATION HELPS (best S={best_s}, -{gain:.2f} um)" if gain > 0.1 else "neutral/harmful"
            print(f"  m={m} K={K:2}: S=0 {g[0]:.2f} -> best {g[best_s]:.2f}  [{verdict}]")
    print("\nwrote results/r8_physics_augmentation.csv")


if __name__ == "__main__":
    main()
