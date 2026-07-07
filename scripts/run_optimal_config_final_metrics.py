"""run_optimal_config_final_metrics.py — full honest metrics (MAE/RMSE/MAPE/maxAE/pooled-R2 +
per-tool R2 distribution) for the THREE configurations adopted from the joint search
(run_optimal_config_search.py): m=2 EB-shrunk, m=3 Siegel repeated-median, m=4 m-matched exponent.
No numbers are invented here -- everything is recomputed LOTO, leakage-safe, for the paper table."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_optimal_config_search import load, tools_of, fit_p, FIT, CENSOR

ADOPTED = [
    dict(name="Ours -- conservative (optimal, m=2)", m=2, p_strategy="full", fit="theil_sen", lam=0.2),
    dict(name="Ours -- conservative (optimal, m=3)", m=3, p_strategy="full", fit="siegel", lam=0.0),
    dict(name="Ours -- precise (optimal, m=4)", m=4, p_strategy="m_matched", fit="theil_sen", lam=0.0),
]


def full_rate_theilsen(g, p):
    gg = g[g.vb <= CENSOR].sort_values("order")
    if len(gg) < 2:
        return np.nan
    a, _ = FIT["theil_sen"](gg.order.to_numpy(float) ** p, gg.vb.to_numpy(float))
    return a


def eval_full(d, m, p_strategy, fit_name, lam):
    fit_fn = FIT[fit_name]
    P, Y, per_mae, per_rmse, per_mape, per_max, per_r2, per_tool_id = [], [], [], [], [], [], [], []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = fit_p(tr, m=(m if p_strategy == "m_matched" else None))
        x = o[:m] ** p
        a_hat, _ = fit_fn(x, v[:m])
        if lam > 0:
            rates = [full_rate_theilsen(gt.sort_values("order"), p) for _, gt in tr.groupby("tool_id")]
            rates = [r for r in rates if np.isfinite(r)]
            a_pop = float(np.median(rates)) if rates else a_hat
            a_use = (1 - lam) * a_hat + lam * a_pop
        else:
            a_use = a_hat
        b_use = float(np.median(v[:m] - a_use * x))
        pred = b_use + a_use * o[fut] ** p
        tru = v[fut]; ae = np.abs(pred - tru)
        P.append(pred); Y.append(tru)
        per_mae.append(ae.mean()); per_rmse.append(np.sqrt((ae ** 2).mean()))
        per_mape.append((ae / tru).mean() * 100); per_max.append(ae.max())
        ss_res = float(np.sum((tru - pred) ** 2))
        ss_tot = float(np.sum((tru - tru.mean()) ** 2)) if len(tru) > 1 else np.nan
        per_r2.append((1 - ss_res / ss_tot) if (ss_tot and ss_tot > 0) else np.nan)
        per_tool_id.append(tt)
    P = np.concatenate(P); Y = np.concatenate(Y)
    pooled_r2 = 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)
    r2arr = np.array(per_r2, dtype=float)
    r2def = r2arr[~np.isnan(r2arr)]
    pm = np.array(per_mae)
    return dict(MAE=np.mean(per_mae), RMSE=np.mean(per_rmse), MAPE=np.mean(per_mape),
                MaxAE=np.mean(per_max), R2_pooled=pooled_r2, n_tools=len(per_mae),
                mae_med=np.median(pm), mae_min=pm.min(), mae_max=pm.max(),
                r2_med=np.median(r2def) if len(r2def) else np.nan,
                r2_pos=int((r2def > 0).sum()), r2_def=len(r2def))


def main():
    d = load()
    recs = []
    print("Adopted (joint-search-optimal) configurations -- full honest metrics:\n")
    print(f"{'name':38} {'m':>2} {'MAE':>6} {'MAEmed':>7} {'RMSE':>6} {'MAPE%':>6} {'MaxAE':>6} "
          f"{'R2pool':>7} {'R2med':>7} {'R2>0':>6}")
    for cfg in ADOPTED:
        r = eval_full(d, cfg["m"], cfg["p_strategy"], cfg["fit"], cfg["lam"])
        print(f"{cfg['name']:38} {cfg['m']:>2} {r['MAE']:6.1f} {r['mae_med']:7.1f} {r['RMSE']:6.1f} "
              f"{r['MAPE']:6.1f} {r['MaxAE']:6.1f} {r['R2_pooled']:7.2f} {r['r2_med']:7.2f} "
              f"{r['r2_pos']}/{r['r2_def']:>3}")
        recs.append(dict(name=cfg["name"], m=cfg["m"], p_strategy=cfg["p_strategy"], fit=cfg["fit"],
                         lam=cfg["lam"], MAE=round(r["MAE"], 1), MAE_median=round(r["mae_med"], 1),
                         MAE_min=round(r["mae_min"], 1), MAE_max=round(r["mae_max"], 1),
                         RMSE=round(r["RMSE"], 1), MAPE_pct=round(r["MAPE"], 1),
                         MaxAE=round(r["MaxAE"], 1), R2_pooled=round(r["R2_pooled"], 2),
                         R2_median=round(r["r2_med"], 2), R2_pos=r["r2_pos"], R2_def=r["r2_def"],
                         n_tools=r["n_tools"]))
    pd.DataFrame(recs).to_csv(os.path.join(ROOT, "results", "optimal_config_final_metrics.csv"), index=False)
    print("\nwrote results/optimal_config_final_metrics.csv")

    # two-line m-curve figure: base few-shot vs joint-optimal config, both LOTO
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        base = pd.read_csv(os.path.join(ROOT, "results", "mcurve.csv")).set_index("m")["MAE_um"]
        opt = {r["m"]: r["MAE"] for r in recs}
        ms = [2, 3, 4]
        bvals = [float(base.get(m, np.nan)) for m in ms]; ovals = [opt[m] for m in ms]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(ms, bvals, "o--", color="#8a8a8a", lw=2, ms=8, label="base few-shot (Theil–Sen)")
        ax.plot(ms, ovals, "o-", color="#1f5fa8", lw=2.4, ms=9, label="joint-optimal (validity-constrained)")
        ax.axhline(18.7, ls=":", color="#b03030", lw=1.8, label="average-wear-curve baseline (18.7)")
        for m, v in zip(ms, ovals):
            ax.annotate(f"{v:.1f}", (m, v), textcoords="offset points", xytext=(0, -14), ha="center",
                        color="#1f5fa8", fontsize=9)
        for m, v in zip(ms, bvals):
            ax.annotate(f"{v:.1f}", (m, v), textcoords="offset points", xytext=(0, 8), ha="center",
                        color="#6a6a6a", fontsize=8)
        ax.set_xlabel("m = number of early VB measurements"); ax.set_ylabel("future-VB MAE (µm)")
        ax.set_title("Minimum-data m-curve: base vs joint-optimal configuration (LOTO)")
        ax.set_xticks(ms); ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "outputs", "figures", "mcurve.png"), dpi=220); plt.close(fig)
        print("regenerated outputs/figures/mcurve.png (base vs optimal)")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
