"""
package_release.py — build a professional, exportable, audit-ready package of the model + findings.

Creates release/phm_tool_wear_release/ with:
  model/        model_params.json, predict.py (standalone, pure-numpy inference), model_card.md
  results/      key metric CSVs
  figures/      the 8 management/result figures
  docs/         protocol+limitations, data request, presentation script, folder map
  presentations/ the two current decks
  INDEX.html    highly visual landing page (objective, figures, metrics, model card, downloads)
  MANIFEST.json inventory + sha256 + provenance + headline metrics

Self-tests the packaged predict.py at the end.
"""
import os, sys, json, shutil, hashlib, datetime
sys.dont_write_bytecode = True          # keep the release free of __pycache__
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
REL = os.path.join(ROOT, "release", "phm_tool_wear_release")
CENSOR = 300.0


def load():
    f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
    d = f[["tool_id", "within_tool_order", "vb_um"]].drop_duplicates()
    return (d.rename(columns={"within_tool_order": "order", "vb_um": "vb"})
            .sort_values(["tool_id", "order"]).reset_index(drop=True))


def fit_global_p(df):
    best_p, best = 0.5, np.inf
    for p in np.arange(0.2, 1.001, 0.05):
        tot = 0.0
        for _, g in df.groupby("tool_id"):
            gg = g[g.vb <= CENSOR]
            if len(gg) < 2:
                continue
            A = np.column_stack([np.ones(len(gg)), gg.order.to_numpy(float) ** p])
            c, *_ = np.linalg.lstsq(A, gg.vb.to_numpy(float), rcond=None)
            tot += float(np.sum((A @ c - gg.vb.to_numpy(float)) ** 2))
        if tot < best:
            best, best_p = tot, p
    return float(best_p)


def theil_sen(x, y):
    s = np.median([(y[j] - y[i]) / (x[j] - x[i])
                   for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]])
    return float(s), float(np.median(y - s * x))


def conformal_q(df, p, m=3):
    res = []
    tools = sorted(df.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))
    for tt in tools:
        g = df[df.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        x = o[:m] ** p; a, b = theil_sen(x, v[:m])
        res += list(np.abs((b + a * o[fut] ** p) - v[fut]))
    res = np.sort(res)
    def q(al):
        k = int(np.ceil((len(res) + 1) * (1 - al))); return float(res[min(k, len(res)) - 1])
    return q(0.2), q(0.1)


def _resid_h(df, p, m=3):
    R, Hh = [], []
    for tt in sorted(df.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0)):
        g = df[df.tool_id == tt].sort_values("order"); o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        x = o[:m] ** p; a, b = theil_sen(x, v[:m]); pred = b + a * o[fut] ** p
        R += list(np.abs(pred - v[fut])); Hh += list((fut - (m - 1)).astype(int))
    return np.array(R), np.array(Hh)


def _bin(h):
    return "near" if h <= 1 else ("mid" if h <= 3 else "far")


def mondrian_q(df, p, m=3):
    """Per-horizon-bin conformal quantiles (Mondrian) — tighter, still valid coverage."""
    R, Hh = _resid_h(df, p, m); out = {}
    for al, lvl in [(0.2, "80"), (0.1, "90")]:
        gk = int(np.ceil((len(R) + 1) * (1 - al))); gq = float(np.sort(R)[min(gk, len(R)) - 1])
        dd = {}
        for bn in ["near", "mid", "far"]:
            rr = np.sort(R[np.array([_bin(h) == bn for h in Hh])])
            if len(rr) >= 5:
                k = int(np.ceil((len(rr) + 1) * (1 - al))); dd[bn] = round(float(rr[min(k, len(rr)) - 1]), 1)
            else:
                dd[bn] = round(gq, 1)
        out[lvl] = dd
    return out


def kalman_pop(df, p):
    """Population params for the online Kalman monitor: measurement var, mean drift, drift var."""
    slopes, resid = [], []
    for _, g in df.groupby("tool_id"):
        gg = g[g.vb <= CENSOR].sort_values("order")
        if len(gg) < 2:
            continue
        tau = gg.order.to_numpy(float) ** p; v = gg.vb.to_numpy(float)
        A = np.column_stack([np.ones(len(tau)), tau]); c, *_ = np.linalg.lstsq(A, v, rcond=None)
        slopes.append(c[1]); resid += list(v - A @ c)
    return (round(max(float(np.var(resid)), 1.0), 2), round(float(np.mean(slopes)), 3),
            round(max(float(np.var(slopes, ddof=1)), 1e-6), 4))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


PREDICT_PY = '''"""
predict.py — standalone inference for the PHM cutting-tool wear model (pure numpy).
Usage:
    from predict import predict_wear
    out = predict_wear(orders_early=[0,1,2], vb_early=[103,108,121], horizon=12)
Given a tool's first few flank-wear (VB, µm) measurements, returns the predicted future VB curve,
a guaranteed-coverage band, and a remaining-useful-life (RUL) window.
"""
import os, json
import numpy as np

_P = json.load(open(os.path.join(os.path.dirname(__file__), "model_params.json")))


def _theil_sen(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    s = np.median([(y[j] - y[i]) / (x[j] - x[i])
                   for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]])
    return float(s), float(np.median(y - s * x))


def predict_wear(orders_early, vb_early, horizon=12, vb_fail=None, level=90):
    """orders_early/vb_early: the tool's first m measurements. Returns the future curve, a Mondrian
    (horizon-adaptive) guaranteed-coverage band, and the RUL window."""
    p = _P["exponent_p"]
    vb_fail = _P["vb_fail_um"] if vb_fail is None else vb_fail
    o = np.asarray(orders_early, float); v = np.asarray(vb_early, float)
    a, b = _theil_sen(o ** p, v)
    grid = np.arange(int(o.min()), int(o.min()) + horizon + 1)
    pred = b + a * grid ** p
    last = float(o.max())
    qm = _P.get("conformal_q_mondrian", {}).get(str(level))     # tighter, horizon-adaptive

    def qof(order_pt):
        if qm is None:
            return _P["conformal_q"][str(level)]
        h = int(order_pt - last)
        return qm["near"] if h <= 1 else (qm["mid"] if h <= 3 else qm["far"])
    qs = np.array([qof(x) for x in grid])
    lo, hi = pred - qs, pred + qs

    def cross(curve):
        idx = np.where(curve >= vb_fail)[0]
        return int(grid[idx[0]]) if len(idx) else None
    t_fail = cross(pred); t_early = cross(hi); t_late = cross(lo)
    return {
        "order": grid.tolist(),
        "vb_pred": [round(x, 1) for x in pred],
        "vb_lo": [round(x, 1) for x in lo], "vb_hi": [round(x, 1) for x in hi],
        "confidence_level_pct": level, "band_halfwidth_um": round(float(qs[-1]), 1),
        "band_type": "mondrian_horizon_adaptive" if qm else "global",
        "vb_fail_um": vb_fail, "t_fail_estimate": t_fail,
        "rul_window": {"earliest": (None if t_early is None else t_early - last),
                       "latest": (None if t_late is None else t_late - last)},
        "exponent_p": p,
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(predict_wear([0, 1, 2], [103, 108, 121], horizon=12))
'''


MONITOR_PY = '''"""
monitor.py — ONLINE next-cut monitoring via a Kalman filter (pure numpy, ~4 um one-step-ahead).
Feed the measurements seen so far -> predicted VB of the NEXT cut. Complements predict.py (early RUL).
"""
import os, json
import numpy as np
_P = json.load(open(os.path.join(os.path.dirname(__file__), "model_params.json")))


def predict_next(orders, vb):
    p = _P["exponent_p"]; K = _P["kalman"]; R = K["R"]; pod = K["pop_drift"]; dv = K["drift_var"]; sa2 = dv
    o = np.asarray(orders, float); v = np.asarray(vb, float); tau = o ** p
    H = np.array([[1.0, 0.0]]); x = np.array([v[0], pod]); P = np.array([[R, 0.0], [0.0, dv]])
    for k in range(1, len(o)):
        dt = tau[k] - tau[k - 1]
        F = np.array([[1.0, dt], [0.0, 1.0]]); Q = sa2 * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
        xp = F @ x; Pp = F @ P @ F.T + Q
        S = (H @ Pp @ H.T)[0, 0] + R; Kg = (Pp @ H.T / S).ravel()
        x = xp + Kg * (v[k] - (H @ xp)[0]); P = (np.eye(2) - np.outer(Kg, H)) @ Pp
    nxt = o.max() + 1; dt = nxt ** p - tau[-1]
    return {"next_order": int(nxt), "vb_next_pred": round(float(x[0] + x[1] * dt), 1)}


if __name__ == "__main__":
    print(predict_next([0, 1, 2, 3], [103, 108, 121, 136]))
'''


EXAMPLE_PY = '''"""
integrate_example.py — how to plug the production model into another model / a pipeline.
The model is a pure-numpy function with a stable contract:
    predict_wear(orders_early, vb_early, horizon, vb_fail=None, level=90) -> dict
Run:  python examples/integrate_example.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from predict import predict_wear


def maintenance_decision(orders, vb, vb_fail=200.0, safety_level=90):
    out = predict_wear(orders, vb, horizon=15, vb_fail=vb_fail, level=safety_level)
    rul = out["rul_window"]; earliest = rul["earliest"]
    return {"change_now": bool(earliest is not None and earliest <= 1),
            "rul_window_cuts": (rul["earliest"], rul["latest"]),
            "band_halfwidth_um": out["band_halfwidth_um"], "vb_next": out["vb_pred"][len(orders)]}


if __name__ == "__main__":
    o = predict_wear([0, 1, 2], [150, 170, 190], horizon=8)
    print("direct predict_wear -> next VB:", o["vb_pred"][3:], "| band +/-", o["band_halfwidth_um"])
    for name, t in {"toolA": ([0, 1, 2], [100, 112, 121]), "toolB": ([0, 1, 2], [150, 170, 190])}.items():
        print(name, maintenance_decision(*t))
'''


def main():
    if os.path.exists(REL):
        shutil.rmtree(REL, ignore_errors=True)
    for sub in ["model", "results", "figures", "docs", "presentations", "examples"]:
        os.makedirs(os.path.join(REL, sub), exist_ok=True)

    df = load(); p = fit_global_p(df); q80, q90 = conformal_q(df, p)
    mq = mondrian_q(df, p); kR, kdrift, kdv = kalman_pop(df, p)
    metrics = pd.read_csv(os.path.join(ROOT, "results", "mcurve_metrics.csv")).to_dict("records")
    params = {
        "name": "PHM cutting-tool wear — physics power + few-shot self-adaptation",
        "version": "1.1", "built": datetime.date.today().isoformat(),
        "model": "VB(t) = b + a * t^p ; (a,b) robust Theil-Sen on the tool's first m points; p global",
        "exponent_p": round(p, 3), "wear_regime_um": CENSOR, "vb_fail_um": 200.0,
        "few_shot_m": {"conservative": 3, "precise": 4},
        "conformal_q": {"80": round(q80, 1), "90": round(q90, 1)},
        "conformal_q_mondrian": mq,   # horizon-adaptive band (near h<=1, mid 2-3, far>=4): tighter, valid
        "kalman": {"R": kR, "pop_drift": kdrift, "drift_var": kdv,
                   "use": "online next-cut monitoring (monitor.py); one-step-ahead MAE ~4 um"},
        "metrics_LOTO": metrics,
        "notes": "Leakage-safe leave-one-tool-out. Condition/sensors carry no learnable signal (R2<0). "
                 "Band = Mondrian horizon-adaptive conformal (90% coverage, mean ~52um vs 92um global).",
    }
    json.dump(params, open(os.path.join(REL, "model", "model_params.json"), "w"), indent=2)
    open(os.path.join(REL, "model", "predict.py"), "w", encoding="utf-8").write(PREDICT_PY)
    open(os.path.join(REL, "model", "monitor.py"), "w", encoding="utf-8").write(MONITOR_PY)
    open(os.path.join(REL, "examples", "integrate_example.py"), "w", encoding="utf-8").write(EXAMPLE_PY)

    # copy results / figures / docs / decks
    for fn in ["model_comparison.csv", "mcurve_metrics.csv", "mcurve.csv", "final_eval_models.csv",
               "final_eval_bootstrap.csv", "final_eval_conformal.csv", "base_model.csv",
               "online_monitor.csv", "normalized_conformal.csv"]:
        sp = os.path.join(ROOT, "results", fn)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(REL, "results", fn))
    figs = ["context_wear", "pipeline_flow", "pipeline_flow_full", "equations", "models_overview",
            "metrics_bars", "mcurve", "positioning_bars", "conformal_demo", "base_vs_ours",
            "all_models_comparison", "kalman_online"]
    for fn in figs:
        sp = os.path.join(ROOT, "outputs", "figures", fn + ".png")
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(REL, "figures", fn + ".png"))
    for fn in ["evaluation_protocol_and_limitations.md", "lab_data_request.md",
               "supervisor_presentation_script_en.md", "executive_script_en.md", "executive_script_es.md",
               "framework_reinforcement_candidates.md", "publishability_assessment.md", "legacy_audit.md"]:
        sp = os.path.join(ROOT, "reports", fn)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(REL, "docs", fn))
    shutil.copy2(os.path.join(ROOT, "FOLDER_MAP.md"), os.path.join(REL, "docs", "FOLDER_MAP.md"))
    for fn in ["executive_deck_en.pptx", "executive_deck_es.pptx",
               "executive_deck_en.pdf", "executive_deck_es.pdf"]:
        sp = os.path.join(ROOT, "outputs", "presentations", fn)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(REL, "presentations", fn))

    # model card
    mc = f"""# Model card — {params['name']}  (v{params['version']}, {params['built']})

## What it does
Predicts a cutting tool's future flank wear (VB, µm) and remaining useful life (RUL) from its first few
measurements, with a guaranteed-coverage uncertainty band.

## Model
`{params['model']}`  ·  global exponent p = {params['exponent_p']}  ·  failure threshold {params['vb_fail_um']:.0f} µm.
Few-shot: conservative m=3, precise m=4. **Band = Mondrian horizon-adaptive conformal** (90% coverage,
mean ~52 µm vs 92 µm global): near-horizon q={params['conformal_q_mondrian']['90']['near']},
mid={params['conformal_q_mondrian']['90']['mid']}, far={params['conformal_q_mondrian']['90']['far']} µm.
**Online monitor** (`monitor.py`, Kalman): next-cut one-step-ahead MAE ~4 µm — for live monitoring.

## Performance (leave-one-tool-out, 18 tools)
| model | m | MAE µm | RMSE µm | MAPE | maxAE µm | R² |
|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {m['model']} | {m['m']} | {m['MAE_um']} | {m['RMSE_um']} | {m['MAPE_pct']}% | {m['MaxAE_um']} | {m['R2']} |"
        for m in metrics) + """

## Intended use & limits
- Use the **band**, not only the point estimate, for maintenance decisions.
- Cutting condition / sensors are NOT predictive here (1 tool per condition) — do not rely on them.
- The accuracy gain over the average-wear-curve is large (~38%) but borderline-significant at n=18;
  replication (≥2 tools/condition) is required to confirm and to validate RUL.
- Inputs treated in the tool's own measurement units; wear regime VB ≤ 300 µm (above = breakage).
"""
    open(os.path.join(REL, "model", "model_card.md"), "w", encoding="utf-8").write(mc)

    # INDEX.html
    rows = "".join(
        f"<tr><td>{m['model']}</td><td>{m['m']}</td><td>{m['MAE_um']}</td><td>{m['RMSE_um']}</td>"
        f"<td>{m['MAPE_pct']}%</td><td>{m['MaxAE_um']}</td><td>{m['R2']}</td></tr>" for m in metrics)
    cards = [("context_wear", "Objective", "Predict the rest from a few early measurements"),
             ("pipeline_flow_full", "Complete flow (graph)", "Both branches: adopted (R²=0.67) vs documented null"),
             ("all_models_comparison", "Baseline vs all", "Our model beats every alternative (LOTO MAE)"),
             ("equations", "Equations", "Monotone physics law, self-adapt, conformal, HI/RUL"),
             ("models_overview", "Models", "Baseline vs ours vs conformal layer"),
             ("base_vs_ours", "Naive vs ours", "Naive ML/NN fails (R²<0); ours works (R²=0.67)"),
             ("metrics_bars", "Error reduction", "18.7 → 9.7 µm; R² −1.31 → 0.67"),
             ("mcurve", "m-curve", "More early points → less error"),
             ("positioning_bars", "Why physics wins", "Beats linear / quadratic / average"),
             ("conformal_demo", "Guaranteed band", "Real values fall inside the 90% band"),
             ("kalman_online", "Online monitor (Kalman)", "Next-cut one-step-ahead MAE ~4 um")]
    figs_html = "".join(
        f'<div class="card"><h3>{t}</h3><p>{d}</p><img src="figures/{n}.png" alt="{t}"></div>'
        for n, t, d in cards)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>PHM Tool-Wear — Model & Findings</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;margin:0;color:#1b2a3a;background:#f4f7fb}}
 header{{background:#102a43;color:#fff;padding:30px 40px}}
 header h1{{margin:0;font-size:28px}} header p{{margin:6px 0 0;color:#bfd2e6}}
 .wrap{{max-width:1100px;margin:0 auto;padding:24px 40px}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
 .card{{background:#fff;border:1px solid #d7e0ea;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
 .card h3{{margin:0 0 4px;color:#1f5fa8}} .card p{{margin:0 0 10px;color:#5a6b7b;font-size:14px}}
 .card img{{width:100%;border-radius:6px;border:1px solid #eef2f7}}
 table{{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;overflow:hidden}}
 th,td{{padding:9px 12px;border:1px solid #e1e8f0;text-align:center;font-size:14px}}
 th{{background:#102a43;color:#fff}} tr:nth-child(even){{background:#eef3f9}}
 h2{{color:#102a43;border-bottom:2px solid #1f5fa8;padding-bottom:6px;margin-top:34px}}
 .pills a{{display:inline-block;margin:4px 8px 4px 0;padding:8px 14px;background:#1f5fa8;color:#fff;
   text-decoration:none;border-radius:20px;font-size:14px}}
 .note{{background:#fff;border-left:4px solid #c0392b;padding:12px 16px;border-radius:6px;margin:10px 0}}
</style></head><body>
<header><h1>PHM Cutting-Tool Wear — Model & Findings</h1>
<p>Minimum-data physics-integrated prognosis · v{params['version']} · {params['built']}</p></header>
<div class="wrap">
 <h2>Results & method (visual)</h2><div class="grid">{figs_html}</div>
 <h2>Performance — leave-one-tool-out (18 tools)</h2>
 <table><tr><th>model</th><th>m</th><th>MAE µm</th><th>RMSE µm</th><th>MAPE</th><th>maxAE µm</th><th>R²</th></tr>{rows}</table>
 <h2>Model</h2>
 <p><b>{params['model']}</b><br>global exponent p = {params['exponent_p']} · failure threshold
 {params['vb_fail_um']:.0f} µm · conformal half-width 80%={params['conformal_q']['80']} µm,
 90%={params['conformal_q']['90']} µm.</p>
 <div class="note"><b>Honest limit:</b> the ~38% gain is borderline-significant at n=18; cutting condition
 is not predictive (1 tool/condition). <b>The ask:</b> ≥2 tools per condition (replication).</div>
 <h2>Package contents</h2>
 <div class="pills">
  <a href="model/model_card.md">Model card</a><a href="model/model_params.json">model_params.json</a>
  <a href="model/predict.py">predict.py</a><a href="results/">results/</a><a href="figures/">figures/</a>
  <a href="docs/evaluation_protocol_and_limitations.md">Protocol & limits</a>
  <a href="docs/lab_data_request.md">Data request</a>
  <a href="docs/supervisor_presentation_script_en.md">Presentation script</a>
  <a href="docs/publishability_assessment.md">Publishability</a>
  <a href="presentations/executive_deck_en.pptx">Exec deck (EN)</a>
  <a href="presentations/executive_deck_es.pptx">Exec deck (ES)</a>
  <a href="presentations/executive_deck_en.pdf">Exec PDF (EN)</a>
  <a href="presentations/executive_deck_es.pdf">Exec PDF (ES)</a>
  <a href="docs/executive_script_en.md">Speaker script (EN)</a>
  <a href="MANIFEST.json">MANIFEST</a>
 </div>
</div></body></html>"""
    open(os.path.join(REL, "INDEX.html"), "w", encoding="utf-8").write(html)

    # clean any bytecode caches, then manifest with checksums
    for r, ds, _ in os.walk(REL):
        for dd in list(ds):
            if dd == "__pycache__":
                shutil.rmtree(os.path.join(r, dd), ignore_errors=True)
    files = []
    for r, _, fs in os.walk(REL):
        for fn in fs:
            fp = os.path.join(r, fn)
            files.append(dict(path=os.path.relpath(fp, REL).replace("\\", "/"),
                              bytes=os.path.getsize(fp), sha256_16=sha(fp)))
    manifest = dict(package=params["name"], version=params["version"], built=params["built"],
                    headline={"baseline_MAE_um": 18.7, "our_MAE_um_m3": 11.6, "our_MAE_um_m4": 9.7,
                              "R2_m4": 0.67, "conformal_q90_um": round(q90, 1)},
                    n_files=len(files), files=sorted(files, key=lambda x: x["path"]))
    json.dump(manifest, open(os.path.join(REL, "MANIFEST.json"), "w"), indent=2)

    # self-test the packaged predict.py
    sys.path.insert(0, os.path.join(REL, "model"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict", os.path.join(REL, "model", "predict.py"))
    pred = importlib.util.module_from_spec(spec); spec.loader.exec_module(pred)
    out = pred.predict_wear([0, 1, 2], [103, 108, 121], horizon=12)
    ok = (len(out["vb_pred"]) == 13 and out["vb_hi"][-1] > out["vb_pred"][-1] > out["vb_lo"][-1]
          and out["band_type"] == "mondrian_horizon_adaptive")
    spec2 = importlib.util.spec_from_file_location("monitor", os.path.join(REL, "model", "monitor.py"))
    mon = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(mon)
    nx = mon.predict_next([0, 1, 2, 3], [103, 108, 121, 136])
    ok2 = isinstance(nx.get("vb_next_pred"), float)
    print(f"global exponent p = {p:.3f} | Mondrian q90 near/mid/far = "
          f"{mq['90']['near']}/{mq['90']['mid']}/{mq['90']['far']} um")
    print(f"packaged predict.py self-test: {'PASS' if ok else 'FAIL'} | monitor.py self-test: "
          f"{'PASS' if ok2 else 'FAIL'} (next={nx})")
    print(f"release built at {REL} | {len(files)} files")


if __name__ == "__main__":
    main()
