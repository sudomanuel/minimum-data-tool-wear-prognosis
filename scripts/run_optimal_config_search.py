"""run_optimal_config_search.py — JOINT search for the optimal few-shot configuration, instead of
testing levers one at a time. Grid over:
  m            in {2,3,4,5}                         (data budget)
  p_strategy   in {full, m_matched}                  (exponent fit uses full training trajectory,
                                                       or only the first m points of each training tool
                                                       -- matching what the model actually sees at deploy)
  fit_method   in {theil_sen, siegel, ols}           (few-shot slope/intercept estimator)
  lambda       in {0, .2, .4, .6, .8}                (empirical-Bayes shrinkage toward the population rate)
=> 2*4*3*5 = 120 configurations, each scored LOTO (leakage-safe) on MAE, RMSE, and the resulting
Mondrian conformal band (90% target coverage, mean width) -- the paper's THREE goals at once:
max accuracy, min data (m), min CI.

For each m, report the Pareto-best configuration (lowest MAE) and its Mondrian band.
Then report the global Pareto frontier across m (does more data always help, and by how much).
Pre-registered rule: adopt the discovered optimum only if it is not worse than the previously
reported per-m numbers (sanity) and report it plainly if it ties or is only marginally better --
no overclaiming.
"""
import os, sys, itertools
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
CENSOR = 300.0
ALPHA = 0.10
P_GRID = np.arange(0.20, 1.001, 0.05)
M_GRID = [2, 3, 4, 5]
LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.8]


def load():
    f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
    d = f[["tool_id", "within_tool_order", "vb_um"]].drop_duplicates()
    return (d.rename(columns={"within_tool_order": "order", "vb_um": "vb"})
            .sort_values(["tool_id", "order"]).reset_index(drop=True))


def tools_of(d):
    return sorted(d.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))


def fit_p(tr, m=None):
    """Global exponent on training tools. If m is given, use only each training tool's first m
    points (m_matched strategy: consistent with what the model sees at deploy time)."""
    best_p, best = 0.5, np.inf
    for p in P_GRID:
        tot = 0.0
        for _, g in tr.groupby("tool_id"):
            gg = g[g.vb <= CENSOR].sort_values("order")
            if m is not None:
                gg = gg.iloc[:m]
            if len(gg) < 2:
                continue
            A = np.column_stack([np.ones(len(gg)), gg.order.to_numpy(float) ** p])
            c, *_ = np.linalg.lstsq(A, gg.vb.to_numpy(float), rcond=None)
            tot += float(np.sum((A @ c - gg.vb.to_numpy(float)) ** 2))
        if tot < best:
            best, best_p = tot, p
    return best_p


def theil_sen(x, y):
    s = np.median([(y[j] - y[i]) / (x[j] - x[i])
                   for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]])
    return float(s), float(np.median(y - s * x))


def siegel(x, y):
    """Siegel repeated-median: per-point median slope, then median of those (more robust to a single
    leverage point than Sen's pooled-pairwise median, which matters at m=3-4)."""
    n = len(x)
    med_i = []
    for i in range(n):
        sl = [(y[j] - y[i]) / (x[j] - x[i]) for j in range(n) if j != i and x[j] != x[i]]
        if sl:
            med_i.append(np.median(sl))
    s = float(np.median(med_i)) if med_i else 0.0
    return s, float(np.median(y - s * x))


def ols(x, y):
    A = np.column_stack([np.ones(len(x)), x])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(c[1]), float(c[0])


FIT = {"theil_sen": theil_sen, "siegel": siegel, "ols": ols}


def full_rate_theilsen(g, p):
    gg = g[g.vb <= CENSOR].sort_values("order")
    if len(gg) < 2:
        return np.nan
    a, _ = theil_sen(gg.order.to_numpy(float) ** p, gg.vb.to_numpy(float))
    return a


def cq(a, al):
    a = np.sort(a)
    k = int(np.ceil((len(a) + 1) * (1 - al)))
    return float(a[min(k, len(a)) - 1])


def run_config(d, p_cache, p_strategy, m, fit_name, lam):
    """LOTO evaluation of one configuration. Returns per-tool residual list + horizon list (for
    conformal) and MAE/RMSE."""
    fit_fn = FIT[fit_name]
    aes, sqes, resid, horiz = [], [], [], []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = p_cache[(p_strategy, m, tt)]
        x = o[:m] ** p
        a_hat, _ = fit_fn(x, v[:m])
        if lam > 0:
            rates = {t: full_rate_theilsen(gt.sort_values("order"), p) for t, gt in tr.groupby("tool_id")}
            rates = [r for r in rates.values() if np.isfinite(r)]
            a_pop = float(np.median(rates)) if rates else a_hat
            a_use = (1 - lam) * a_hat + lam * a_pop
        else:
            a_use = a_hat
        b_use = float(np.median(v[:m] - a_use * x))
        pred = b_use + a_use * o[fut] ** p
        tru = v[fut]; ae = np.abs(pred - tru)
        aes.append(ae.mean()); sqes.append(np.mean(ae ** 2))
        resid.append(ae); horiz.append((fut - (m - 1)).astype(float))
    return dict(MAE=float(np.mean(aes)), RMSE=float(np.sqrt(np.mean(sqes))),
                n=len(aes), resid=resid, horiz=horiz)


def mondrian_band(resid_list, horiz_list):
    """LOTO-honest Mondrian coverage/width: calibrate on all OTHER tools' residuals, score this
    tool's own residuals -- exactly as in run_cqr_cv_test.py."""
    R = [np.concatenate(r) if len(r) else np.array([]) for r in [resid_list]]
    n_tools = len(resid_list)
    cov, wid = [], []
    all_r = resid_list; all_h = horiz_list
    for i in range(n_tools):
        cal_r = np.concatenate([all_r[j] for j in range(n_tools) if j != i])
        cal_h = np.concatenate([all_h[j] for j in range(n_tools) if j != i])
        r, h = all_r[i], all_h[i]
        if len(r) == 0 or len(cal_r) < 5:
            continue
        def qof(hh):
            sel = (cal_h <= 1) if hh <= 1 else ((cal_h >= 2) & (cal_h <= 3) if hh <= 3 else cal_h >= 4)
            return cq(cal_r[sel], ALPHA) if sel.sum() >= 5 else cq(cal_r, ALPHA)
        qq = np.array([qof(hh) for hh in h])
        inside = r <= qq
        cov.append(inside.mean()); wid.append((2 * qq).mean())
    return float(np.mean(cov)) * 100 if cov else np.nan, float(np.mean(wid)) if wid else np.nan


def main():
    d = load()
    print("Precomputing global exponent p per LOTO fold x (p_strategy, m) ...")
    p_cache = {}
    for p_strategy in ("full", "m_matched"):
        for m in M_GRID:
            for tt in tools_of(d):
                tr = d[d.tool_id != tt]
                p_cache[(p_strategy, m, tt)] = fit_p(tr, m=(m if p_strategy == "m_matched" else None))
    print(f"  cached {len(p_cache)} exponent fits\n")

    COV_MIN = 88.0  # pre-stated validity floor for a 90%-target conformal band (project convention)
    configs = list(itertools.product(("full", "m_matched"), M_GRID, FIT.keys(), LAMBDAS))
    print(f"Joint grid: {len(configs)} configurations (p_strategy x m x fit_method x lambda)\n")
    rows = []
    for p_strategy, m, fit_name, lam in configs:
        r = run_config(d, p_cache, p_strategy, m, fit_name, lam)
        cov, wid = mondrian_band(r["resid"], r["horiz"])
        rows.append(dict(p_strategy=p_strategy, m=m, fit=fit_name, lam=lam,
                         MAE=round(r["MAE"], 3), RMSE=round(r["RMSE"], 3), n=r["n"],
                         mondrian_cov=round(cov, 1), mondrian_width=round(wid, 1)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "optimal_config_full_grid.csv"), index=False)
    print(f"wrote results/optimal_config_full_grid.csv ({len(df)} configs, incl. Mondrian cov/width)\n")

    print(f"=== Best VALID configuration PER DATA BUDGET m (lowest MAE subject to coverage >= {COV_MIN}%) ===")
    best_per_m = []
    for m in M_GRID:
        sub = df[df.m == m]
        valid = sub[sub.mondrian_cov >= COV_MIN].sort_values("MAE")
        if len(valid) == 0:
            top = sub.sort_values("MAE").iloc[0]
            flag = "  [NO CONFIG AT THIS m REACHES VALID COVERAGE -- reporting best-MAE anyway, flagged invalid]"
        else:
            top = valid.iloc[0]; flag = ""
        best_per_m.append(dict(m=m, p_strategy=top.p_strategy, fit=top.fit, lam=top.lam,
                               MAE=top.MAE, RMSE=top.RMSE, mondrian_cov=top.mondrian_cov,
                               mondrian_width=top.mondrian_width, n=top.n,
                               valid=bool(top.mondrian_cov >= COV_MIN)))
        print(f"  m={m}: p_strategy={top.p_strategy:9s} fit={top.fit:10s} lambda={top.lam:.1f}  "
              f"-> MAE={top.MAE:5.2f} RMSE={top.RMSE:5.2f}  |  Mondrian@90%: cov={top.mondrian_cov:4.1f}%  "
              f"width={top.mondrian_width:5.1f} um  n={top.n}{flag}")

    bpm = pd.DataFrame(best_per_m)
    bpm.to_csv(os.path.join(ROOT, "results", "optimal_config_per_m.csv"), index=False)

    print("\n=== Reference: previously reported plain few-shot (fit=theil_sen, lam=0, p_strategy=full) ===")
    ref = []
    for m in M_GRID:
        row = df[(df.m == m) & (df.fit == "theil_sen") & (df.lam == 0.0) & (df.p_strategy == "full")].iloc[0]
        ref.append(dict(m=m, MAE=row.MAE))
        print(f"  m={m}: MAE={row.MAE:5.2f}")

    print("\n=== Gain from joint optimum vs the reference (per m) ===")
    for b, r in zip(best_per_m, ref):
        gain = r["MAE"] - b["MAE"]
        print(f"  m={b['m']}: reference {r['MAE']:.2f} -> optimum {b['MAE']:.2f}  "
              f"(gain {gain:+.2f} um{'  <-- adopt' if gain > 0.05 else '  (no material gain, keep reference for simplicity)'})")

    print(f"\nwrote results/optimal_config_per_m.csv ({len(bpm)} rows)")

    # global recommendation: knee of the Pareto frontier (m vs MAE) among VALID (coverage>=COV_MIN)
    # configs only -- smallest m within 10% of the best achievable MAE across all valid m
    valid_bpm = bpm[bpm.valid]
    if len(valid_bpm) == 0:
        print("\n=== No m reaches valid Mondrian coverage -- cannot recommend an operating point ===")
    else:
        best_overall = valid_bpm.MAE.min()
        knee = valid_bpm[valid_bpm.MAE <= best_overall * 1.10].iloc[0]
        print(f"\n=== Recommended operating point (minimum data within 10% of best VALID accuracy) ===")
        print(f"  m={int(knee.m)}  p_strategy={knee.p_strategy}  fit={knee.fit}  lambda={knee.lam}  "
              f"-> MAE={knee.MAE} um, Mondrian width={knee.mondrian_width} um @ {knee.mondrian_cov}% coverage")


if __name__ == "__main__":
    main()
