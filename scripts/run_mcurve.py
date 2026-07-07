"""
run_mcurve.py — the minimum-data m-curve (the lever that works) with full error metrics.

Deployed model: physics-integrated wear law VB = b + a*order^p with FEW-SHOT SELF-ADAPTATION — the
slope a and level b are fit (robust Theil-Sen) to the held-out tool's OWN first m points; the global
exponent p is fit on TRAINING tools only (leakage-safe). Scored on the sealed future (order >= m,
wear regime VB <= 300) under Leave-One-Tool-Out.

Reports two operating points and the average-wear-curve baseline, each with MAE, RMSE, MAPE, R^2 and
max error:
    conservative = m=3 early measurements   |   precise = m=4 early measurements
Plus the m-curve (m=2,3,4 on the same 18 tools) and a figure.

Outputs: results/mcurve_metrics.csv, results/mcurve.csv, outputs/figures/mcurve.png.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from phm.prognostic_system import fit_population, fewshot_offset, predict_vb

CENSOR = 300.0


def load():
    f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
    d = f[["tool_id", "within_tool_order", "vb_um"]].drop_duplicates()
    return (d.rename(columns={"within_tool_order": "order", "vb_um": "vb"})
            .sort_values(["tool_id", "order"]).reset_index(drop=True))


def fit_global_p(tr):
    best_p, best = 0.5, np.inf
    for p in np.arange(0.2, 1.001, 0.05):
        tot = 0.0
        for _, g in tr.groupby("tool_id"):
            gg = g[g.vb <= CENSOR]
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


def tools_of(d):
    return sorted(d.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))


def eval_model(d, m, model):
    """Return per-point (pred,true) arrays + per-tool error lists for the given model at horizon m.
    model in {'self','baseline'}."""
    P, Y, perMAE, perRMSE, perMAPE, perMAX = [], [], [], [], [], []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = fit_global_p(tr)
        if model == "self":
            x = o[:m] ** p; a, b = theil_sen(x, v[:m]); pred = b + a * o[fut] ** p
        else:  # baseline average-wear-curve + few-shot offset
            pop = fit_population(tr); off = fewshot_offset(pop, o, v, m); pred = predict_vb(pop, o[fut], off)
        tru = v[fut]; ae = np.abs(pred - tru)
        P.append(pred); Y.append(tru)
        perMAE.append(ae.mean()); perRMSE.append(np.sqrt((ae ** 2).mean()))
        perMAPE.append((ae / tru).mean() * 100); perMAX.append(ae.max())
    P = np.concatenate(P); Y = np.concatenate(Y)
    r2 = 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)        # pooled R^2
    return dict(MAE=np.mean(perMAE), RMSE=np.mean(perRMSE), MAPE=np.mean(perMAPE),
                MaxAE=np.mean(perMAX), R2=r2, n_tools=len(perMAE))


def main():
    d = load()
    rows = [("Average-wear-curve (baseline)", "m=3", eval_model(d, 3, "baseline")),
            ("Our model: physics + few-shot self-adapt (CONSERVATIVE)", "m=3", eval_model(d, 3, "self")),
            ("Our model: physics + few-shot self-adapt (PRECISE)", "m=4", eval_model(d, 4, "self"))]
    recs = []
    print("Deployed model = physics wear law + few-shot self-adaptation (robust). LOTO, wear regime.\n")
    print(f"{'model':54} {'m':>3} {'MAE':>6} {'RMSE':>6} {'MAPE%':>6} {'MaxAE':>6} {'R2':>6}")
    for name, mm, r in rows:
        print(f"{name:54} {mm:>3} {r['MAE']:6.1f} {r['RMSE']:6.1f} {r['MAPE']:6.1f} "
              f"{r['MaxAE']:6.1f} {r['R2']:6.2f}")
        recs.append(dict(model=name, m=mm, MAE_um=round(r['MAE'], 1), RMSE_um=round(r['RMSE'], 1),
                         MAPE_pct=round(r['MAPE'], 1), MaxAE_um=round(r['MaxAE'], 1),
                         R2=round(r['R2'], 2), n_tools=r['n_tools']))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    pd.DataFrame(recs).to_csv(os.path.join(ROOT, "results", "mcurve_metrics.csv"), index=False)

    # m-curve (same model, m=2..4, all 18 tools)
    mc = [(m, eval_model(d, m, "self")["MAE"]) for m in [2, 3, 4]]
    pd.DataFrame(mc, columns=["m", "MAE_um"]).round(2).to_csv(
        os.path.join(ROOT, "results", "mcurve.csv"), index=False)
    print("\nm-curve (MAE vs # early measurements, all 18 tools):")
    for m, mae in mc:
        print(f"   m={m}: MAE={mae:.2f} um")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ms = [m for m, _ in mc]; vals = [v for _, v in mc]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ms, vals, "o-", color="#1f5fa8", lw=2, ms=8)
        ax.axhline(recs[0]["MAE_um"], ls="--", color="#b03030",
                   label=f"average-wear-curve baseline ({recs[0]['MAE_um']} um)")
        for m, v in mc:
            ax.annotate(f"{v:.1f}", (m, v), textcoords="offset points", xytext=(0, 8), ha="center")
        ax.set_xlabel("m = number of early VB measurements"); ax.set_ylabel("future-VB MAE (um)")
        ax.set_title("Minimum-data m-curve (LOTO, few-shot self-adaptation)")
        ax.set_xticks(ms); ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        os.makedirs(os.path.join(ROOT, "outputs", "figures"), exist_ok=True)
        fig.savefig(os.path.join(ROOT, "outputs", "figures", "mcurve.png"), dpi=220)
        print("\nwrote outputs/figures/mcurve.png + results/mcurve_metrics.csv + results/mcurve.csv")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
