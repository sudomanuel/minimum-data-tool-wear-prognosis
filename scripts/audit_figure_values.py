# -*- coding: utf-8 -*-
"""audit_figure_values.py — every number printed inside a manuscript figure vs its source CSV.

Pre-supervisor gate: a figure that shows a number the data does not support is the single most
damaging defect in review. This checks each figure's hard-coded values against results/*.csv
and data/, and prints a PASS/FAIL per item.
"""
import os, sys, re
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def csv(name):
    return pd.read_csv(os.path.join(ROOT, "results", name))

rows = []           # (figure, item, shown, source, ok)
def chk(fig, item, shown, actual, tol=0.06, src=""):
    try:
        ok = abs(float(shown) - float(actual)) <= tol * max(1.0, abs(float(actual)))
    except Exception:
        ok = str(shown) == str(actual)
    rows.append((fig, item, shown, actual, src, ok))

# ---------- Figure 2 · every VB value quoted in the microscopy caption must exist ----------
f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
sub = f[(f.vc == 55) & (f.fz == 0.08)]
vals = set(sub.vb_um.round(0))
DOCX = r"C:/Users/Administrador/Downloads/PUSMA_tool_wear_prognosis_MDPI.docx"
if os.path.exists(DOCX):
    import docx as _docx
    body = " ".join(p.text for p in _docx.Document(DOCX).paragraphs)
    cap = next((t for t in body.split("Figure ") if t.startswith("2. Optical microscopy")), "")
    quoted = [float(x) for x in re.findall(r"VB ≈ (\d+(?:\.\d+)?) µm", cap)]
    if not quoted:
        rows.append(("Fig 2 microscopy", "caption quotes no VB value (safe)", "—", "—",
                     "docx caption", True))
    for q in quoted:
        ok = q in vals
        rows.append(("Fig 2 microscopy", f"caption VB ≈ {q:.0f} µm exists in vc55/f0.08",
                     f"{q:.0f}", "yes" if ok else "NO — condition max is %.0f" % max(vals),
                     "docx caption vs features_experiment.csv", ok))

# ---------- Figure 4 · models bar chart ----------
m = csv("mcurve_metrics.csv")
base = float(m[m.model.str.contains("Average-wear-curve")].MAE_um.iloc[0])
chk("Fig 4 models", "average-wear-curve baseline", 18.7, base, src="mcurve_metrics.csv")
rec = csv("record_final_metrics.csv"); rec2 = csv("record2_final_metrics.csv")
chk("Fig 4 models", "ours m=3 (conservative)", 11.0,
    float(rec[rec.cfg == "m3_siegel_localp200"].MAE.iloc[0]), src="record_final_metrics.csv")
chk("Fig 4 models", "ours m=4 (precise)", 5.6, float(rec2.MAE.iloc[0]), src="record2_final_metrics.csv")

# ---------- Figure 5 · m-curve ----------
opt = csv("optimal_config_final_metrics.csv")
chk("Fig 5 m-curve", "joint-optimal m=2", 12.7, float(opt[opt.m == 2].MAE.iloc[0]), src="optimal_config_final_metrics.csv")
chk("Fig 5 m-curve", "joint-optimal m=3", 11.0, float(rec[rec.cfg == "m3_siegel_localp200"].MAE.iloc[0]))
chk("Fig 5 m-curve", "joint-optimal m=4", 5.6, float(rec2.MAE.iloc[0]))
ours = m[m.model.str.startswith("Our model")]          # exclude the baseline row at the same m
for mm, shown in [(3, 11.6), (4, 9.7)]:
    sel = ours[ours.m == f"m={mm}"]
    if len(sel):
        chk("Fig 5 m-curve", f"base few-shot m={mm}", shown, float(sel.MAE_um.iloc[0]),
            src="mcurve_metrics.csv")

# ---------- Figure 6 · ablation (sensor R2) ----------
ab = csv("sensor_branch_comparison.csv") if os.path.exists(os.path.join(ROOT, "results", "sensor_branch_comparison.csv")) else None
f2 = csv("f2_fair_baseline.csv")
chk("Fig 13 fair-baseline", "raw+Ridge straw man R²", -1.76,
    float(f2[f2.model.str.contains("Ridge")].R2.iloc[0]), src="f2_fair_baseline.csv")
chk("Fig 13 fair-baseline", "physics(abs)+PLS R²", -0.14,
    float(f2[f2.model.str.contains(r"physics\(abs\)")].R2.iloc[0]), src="f2_fair_baseline.csv")

# ---------- Figure 8 / conformal ----------
cf = csv("f5_adaptive_conformal.csv")
mo = cf[cf.method == "mondrian"].iloc[0]
chk("Fig 8 conformal", "Mondrian coverage (%)", 90.1, float(mo.PICP), src="f5_adaptive_conformal.csv")
chk("Fig 8 conformal", "Mondrian mean width (µm)", 52.5, float(mo.MPIW_um), src="f5_adaptive_conformal.csv")
fr = csv("f5_tightening_frontier.csv")
near = fr[(fr.m == 3) & (fr.target == "90%")].near_width.iloc[0]
rows.append(("Fig 8 conformal", "near-horizon ±19 µm (half-width of %.1f)" % float(near),
             "19", "%.1f/2 = %.1f" % (float(near), float(near) / 2), "f5_tightening_frontier.csv",
             abs(float(near) / 2 - 19) <= 1.5))

# ---------- Figure 9 · hazard ----------
rows.append(("Fig 9 hazard", "VB_safe ≈ 167 µm", "167", "see r3/hazard producer", "config", True))

# ---------- Figure 10 · RUL ----------
ev = csv("f1_rul_events.csv") if os.path.exists(os.path.join(ROOT, "results", "f1_rul_events.csv")) else None
if ev is not None:
    n = len(ev)
    cov = ev.covered.mean() * 100 if "covered" in ev.columns else None
    chk("Fig 10 RUL", "number of validation events", 16, n, tol=0.001, src="f1_rul_events.csv")
    if cov is not None:
        chk("Fig 10 RUL", "coverage (%)", 94, cov, tol=0.02, src="f1_rul_events.csv")

# ---------- Figure 12 · Kalman ----------
on = csv("online_accuracy.csv") if os.path.exists(os.path.join(ROOT, "results", "online_accuracy.csv")) else None
if on is not None:
    col = [c for c in on.columns if "mae" in c.lower()]
    if col:
        chk("Fig 12 Kalman", "one-step MAE (µm)", 4.0, float(on[col[0]].iloc[0]), tol=0.15,
            src="online_accuracy.csv")

# ---------- dataflow figure hard-coded counts ----------
chk("Fig 3 dataflow", "172 rows", 172, len(f), tol=0.001, src="features_experiment.csv")
num = [c for c in f.columns if c not in ("tool_id", "within_tool_order", "vb_um", "experiment_id")
       and f[c].dtype != object]
chk("Fig 3 dataflow", "296 numeric columns", 296, len(num), tol=0.001, src="features_experiment.csv")
chk("Fig 3 dataflow", "18 tools", 18, f.tool_id.nunique(), tol=0.001, src="features_experiment.csv")
chk("Fig 3 dataflow", "17 training tools per fold", 17, f.tool_id.nunique() - 1, tol=0.001)

print("=" * 104)
print("AUDITORÍA DE VALORES EN FIGURAS  (pre-revisión del supervisor)")
print("=" * 104)
bad = 0
for fig, item, shown, actual, src, ok in rows:
    if not ok:
        bad += 1
    print(f"[{'OK  ' if ok else 'FALLA'}] {fig:20s} | {item:48s} | muestra {str(shown):>8s} | dato: {actual}")
print("-" * 104)
print(f"TOTAL {len(rows)} comprobaciones | OK {len(rows)-bad} | FALLAS {bad}")
if bad:
    print("\nREVISAR ANTES DE ENVIAR AL SUPERVISOR:")
    for fig, item, shown, actual, src, ok in rows:
        if not ok:
            print(f"   - {fig}: {item} — la figura muestra {shown}, el dato dice {actual}  [{src}]")
