"""run_per_tool_metrics.py — per-tool MAE and R² for the deployed few-shot model (LOTO, m=3 & m=4),
so the paper can report the per-tool distribution (not only pooled R²). Honest, reproducible."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load, fit_global_p, theil_sen, tools_of  # reuse identical model
CENSOR = 300.0


def per_tool(d, m):
    rows = []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = fit_global_p(tr)
        x = o[:m] ** p; a, b = theil_sen(x, v[:m]); pred = b + a * o[fut] ** p
        tru = v[fut]; ae = np.abs(pred - tru)
        ss_res = float(np.sum((tru - pred) ** 2))
        ss_tot = float(np.sum((tru - tru.mean()) ** 2)) if len(tru) > 1 else np.nan
        r2 = (1 - ss_res / ss_tot) if (ss_tot and ss_tot > 0) else np.nan
        rows.append(dict(tool=tt, n_future=len(fut), MAE=ae.mean(), R2=r2))
    return pd.DataFrame(rows)


def summarize(df, m):
    mae = df.MAE.to_numpy()
    r2 = df.R2.dropna().to_numpy()
    print(f"\n=== m={m}  ({len(df)} tools; R² defined for {len(r2)}) ===")
    print(f"  MAE per-tool: median {np.median(mae):.1f}, range [{mae.min():.1f}, {mae.max():.1f}] µm")
    print(f"  R²  per-tool: median {np.median(r2):.2f}, range [{r2.min():.2f}, {r2.max():.2f}]; "
          f"{int((r2 > 0).sum())}/{len(r2)} tools with R² > 0")
    return dict(m=m, n=len(df), mae_med=np.median(mae), mae_min=mae.min(), mae_max=mae.max(),
                r2_med=np.median(r2), r2_min=r2.min(), r2_max=r2.max(),
                r2_pos=int((r2 > 0).sum()), r2_def=len(r2))


def main():
    d = load()
    out = []
    for m in (3, 4):
        df = per_tool(d, m)
        df.round(2).to_csv(os.path.join(ROOT, "results", f"per_tool_metrics_m{m}.csv"), index=False)
        out.append(summarize(df, m))
    pd.DataFrame(out).round(2).to_csv(os.path.join(ROOT, "results", "per_tool_summary.csv"), index=False)
    print("\nwrote results/per_tool_metrics_m3.csv, _m4.csv, per_tool_summary.csv")


if __name__ == "__main__":
    main()
