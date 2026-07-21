# -*- coding: utf-8 -*-
"""run_p3_breakdown.py — the per-horizon and per-tool detail behind the adopted result.

The nested audit says the horizon-adaptive weighting wins; it does not say WHERE it wins. The
manuscript claims a mechanism ("the trade-off Paper 1 stopped at is an artefact of one global
gamma"), and a mechanism claim has to be shown at the level where it operates: per horizon.

This script re-runs the adopted configuration (m = 4, gamma(h) = 8/h) and the Paper-1 record
(m = 4, gamma = 3) under the identical leave-one-tool-out protocol and writes, for every single
prediction, the tool, the horizon, the truth and both errors. Everything downstream — figures,
tables, the per-horizon rows of the manuscript — reads these two files, so no number in the paper
can drift away from the experiment that produced it.

Outputs
  results/p3_breakdown_predictions.csv   one row per (tool, horizon): truth and both errors
  results/p3_breakdown_horizon.csv       MAE per horizon, both schemes, plus the gamma actually used
  results/p3_breakdown_tool.csv          MAE per tool, both schemes
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from run_mcurve import load, tools_of
from run_optimal_config_search import fit_p
from run_record_attempts2 import wls_gamma, local_p_grid

CENSOR = 300.0
M = 4
REC = 5.63
SCHEMES = {"record": lambda h: 3.0, "adopted": lambda h: 8.0 / max(h, 1)}


def run_scheme(d, gamma_fn):
    """Leave-one-tool-out forecast of the sealed continuation of every tool."""
    out = []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]
        g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= M:
            continue
        fut = np.arange(M, len(o))
        fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = fit_p(tr, m=M)                                   # fleet exponent, training tools only
        p = local_p_grid(o, v, M, p, lambda t, y: wls_gamma(t, y, 1.0))
        tau = o[:M] ** p
        for j in fut:
            h = int(j - (M - 1))
            a, b = wls_gamma(tau, v[:M], gamma_fn(h))
            out.append(dict(tool=tt, horizon=h, order=float(o[j]), truth=float(v[j]),
                            pred=float(b + a * o[j] ** p), gamma=float(gamma_fn(h))))
    return pd.DataFrame(out)


def main():
    d = load()
    rec = run_scheme(d, SCHEMES["record"]).rename(
        columns=dict(pred="pred_record", gamma="gamma_record"))
    ado = run_scheme(d, SCHEMES["adopted"]).rename(
        columns=dict(pred="pred_adopted", gamma="gamma_adopted"))
    df = rec.merge(ado, on=["tool", "horizon", "order", "truth"])
    df["err_record"] = df.pred_record - df.truth
    df["err_adopted"] = df.pred_adopted - df.truth
    df.to_csv(os.path.join(ROOT, "results", "p3_breakdown_predictions.csv"), index=False)

    # ---- per horizon: where the gain actually is ----
    hz = df.groupby("horizon").apply(lambda g: pd.Series(dict(
        n=len(g),
        gamma_record=g.gamma_record.iloc[0],
        gamma_adopted=round(float(g.gamma_adopted.iloc[0]), 3),
        MAE_record=round(float(g.err_record.abs().mean()), 2),
        MAE_adopted=round(float(g.err_adopted.abs().mean()), 2),
    ))).reset_index()
    hz["gain_um"] = (hz.MAE_record - hz.MAE_adopted).round(2)
    hz["gain_pct"] = (100 * (hz.MAE_record - hz.MAE_adopted) / hz.MAE_record).round(1)
    hz.to_csv(os.path.join(ROOT, "results", "p3_breakdown_horizon.csv"), index=False)

    # ---- per tool: is the gain a fleet property or one lucky tool? ----
    tl = df.groupby("tool").apply(lambda g: pd.Series(dict(
        n=len(g),
        MAE_record=round(float(g.err_record.abs().mean()), 2),
        MAE_adopted=round(float(g.err_adopted.abs().mean()), 2),
    ))).reset_index()
    tl["gain_um"] = (tl.MAE_record - tl.MAE_adopted).round(2)
    tl["improved"] = tl.gain_um > 0
    tl.to_csv(os.path.join(ROOT, "results", "p3_breakdown_tool.csv"), index=False)

    mr = float(tl.MAE_record.mean())
    ma = float(tl.MAE_adopted.mean())
    print(f"per-tool mean MAE   record {mr:.2f} um   adopted {ma:.2f} um   "
          f"(record target {REC})", flush=True)
    print(f"tools improved: {int(tl.improved.sum())}/{len(tl)}", flush=True)
    print("\nper horizon:")
    print(hz.to_string(index=False))
    print("\nwrote results/p3_breakdown_{predictions,horizon,tool}.csv")


if __name__ == "__main__":
    main()
