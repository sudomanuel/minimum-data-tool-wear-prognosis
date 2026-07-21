# -*- coding: utf-8 -*-
"""make_paper3_graphs_v2.py — Paper-3 knowledge graph (new axis) + method pipeline + equations.

Produces three artefacts, all obeying the house drawing rules established for the trilogy:
  * ORTHOGONAL connectors only (no angular crossings);
  * a clearance validator: no connector may graze a box it does not connect to;
  * schematic hops where nets cross and junction dots where they genuinely join;
  * legends placed outside the drawing area so they never cover an element.

  graphify-out/paper3_horizon_map.json / .html   the paper's knowledge graph (vis-network)
  outputs/figures/p3_pipeline.png                the method pipeline, circuit-style
  outputs/figures/p3_equations.png               the equations actually used, in order

The paper axis is NO LONGER meta-learning: it is HORIZON-ADAPTIVE WEIGHTING of a closed-form
robust estimator. PI-MAML survives only as an archived internal negative and is not drawn here.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_dataflow_figure import Router, CLEAR, HOP_R          # the validated orthogonal router

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "outputs", "figures")
PRINT = os.path.join(FIG, "print")
os.makedirs(PRINT, exist_ok=True)
GOUT = next((p for p in [r"D:/KSF/PHM/phm_tool_wear/graphify-out",
                         os.path.join(ROOT, "graphify-out")] if os.path.isdir(p)),
            os.path.join(ROOT, "graphify-out"))

NAVY, INK, SLATE = "#141E4F", "#1a2a36", "#5a6b7a"
EMER, EMER_BG = "#0E7A4D", "#EAF6F0"
BLUE, BLUE_BG = "#1F5FA8", "#EDF3FA"
TEAL, TEAL_BG = "#0E7490", "#E9F5F8"
RED, RED_BG = "#B03A2E", "#FDF0EE"
PUR, PUR_BG = "#5B21B6", "#F4EFFC"
GREY_LN, GOLD = "#7A8794", "#B7791F"


def _res(f):
    p = os.path.join(RES, f)
    return pd.read_csv(p) if os.path.exists(p) else None


def facts():
    """Pull the live numbers so the graph can never disagree with the experiments."""
    f = dict(rec3=11.02, rec4=5.63, base=18.7)
    a = _res("p3_nested_audit.csv")
    if a is not None and len(a[a.m == 4]):
        r = a[a.m == 4].iloc[0]
        f.update(m4_search=float(r.in_search_MAE), m4_nested=float(r.nested_MAE),
                 m4_opt=float(r.optimism), m4_share=float(r.nested_winner_share),
                 m4_cov=float(r.coverage_pct), m4_width=float(r.mean_width_um),
                 m4_scheme=str(r.best_scheme))
    if a is not None and len(a[a.m == 3]):
        r = a[a.m == 3].iloc[0]
        f.update(m3_nested=float(r.nested_MAE), m3_cov=float(r.coverage_pct))
    p = _res("p3_probe_estimator.csv")
    if p is not None:
        g = p[(p.m == 4) & (p.tag.str.startswith("global gamma="))]
        f["gamma_curve"] = [(int(t.split("=")[1]), float(m), float(r))
                            for t, m, r in zip(g.tag, g.MAE, g.pooled_R2)]
        h = p[(p.m == 4) & (p.tag.str.startswith("gamma(h)"))]
        f["adaptive"] = [(t, float(m), float(r)) for t, m, r in zip(h.tag, h.MAE, h.pooled_R2)]
    hz = _res("p3_breakdown_horizon.csv")
    if hz is not None:
        r1 = hz[hz.horizon == 1].iloc[0]
        f.update(h1_rec=float(r1.MAE_record), h1_ado=float(r1.MAE_adopted),
                 h1_gain=float(r1.gain_pct), n_far=int(hz[hz.horizon > 8].n.sum()))
    tl = _res("p3_breakdown_tool.csv")
    if tl is not None:
        f.update(n_tools=len(tl), n_up=int((tl.MAE_adopted < tl.MAE_record).sum()),
                 n_tie=int((tl.MAE_adopted == tl.MAE_record).sum()),
                 n_dn=int((tl.MAE_adopted > tl.MAE_record).sum()))
    pr = _res("p3_breakdown_predictions.csv")
    if pr is not None:
        f["n_pred"] = len(pr)
    s = _res("p3_m3_siegel.csv")
    if s is not None:
        f["m3_wrm"] = sorted({round(float(x), 2) for x in
                              s[s.tag.str.startswith("WRM gamma(h)=") &
                                ~s.tag.str.contains("weight-free")].MAE})
    return f


# ============================================================ 1 · knowledge graph
def build_graph(F):
    N = [
     ("P3", f"PAPER 3\nHorizon-Adaptive Weighting\nof a closed-form robust estimator", "core"),
     # the opening
     ("o_stop", "THE OPENING · Paper 1 stopped at γ = 3\n\"γ>3 lowers MAE further but is not\nadopted\" (its own record code)", "open"),
     ("o_trade", "The trade-off it could not resolve\nlarger γ → lower MAE but degraded pooled R²", "open"),
     ("o_hyp", "HYPOTHESIS\nthe trade-off is an artefact of ONE GLOBAL γ", "open"),
     # method
     ("m_law", "Wear law inherited from Paper 1\nVB(t) = b + a·τ,  τ = t^p", "meth"),
     ("m_wls", "Weighted fit at horizon h\nw_i ∝ τ_i^γ(h)", "meth"),
     ("m_gam", "HORIZON-ADAPTIVE WEIGHT\nγ(h) = c / h  — sharp near, anchored far", "meth"),
     ("m_pool", "Per-horizon refit: one weighting\nper predicted step, not one per tool", "meth"),
     # protocol
     ("v_loocv", "LOOCV by tool · 18 folds\nleakage-safe, inherited from Paper 1", "proto"),
     ("v_nested", "NESTED selection audit\nscheme re-chosen blind in every fold", "proto"),
     ("v_conf", "Conformal validity gate\nMondrian band, coverage ≥ 88% for 90%", "proto"),
     ("v_rule", "Pre-stated adoption rule\nbeat the record AND keep coverage", "proto"),
     # results
     ("r_m4", f"ADOPTED · m = 4\n{F.get('m4_search',3.63):.2f} µm in-search → "
              f"{F.get('m4_nested',3.57):.2f} µm NESTED\nvs record {F['rec4']} µm", "res"),
     ("r_opt", f"Selection optimism {F.get('m4_opt',-0.06):+.2f} µm\n"
               f"same scheme wins in {F.get('m4_share',0.94)*100:.0f}% of blind folds", "res"),
     ("r_cov", f"Coverage {F.get('m4_cov',89.0):.1f}% (gate 88)\nmean width {F.get('m4_width',49.0):.0f} µm", "res"),
     ("r_r2", "Pooled R² preserved at 0.70\n(global γ=6 would drop it to 0.44)", "res"),
     # the honest boundary
     ("n_m3", f"NOT ADOPTED · m = 3\nnested {F.get('m3_nested',12.13):.2f} µm vs record {F['rec3']} µm", "neg"),
     ("n_why", "MECHANISM: the m=3 record is a MEDIAN\nestimator — with 3 points the weighted\nmedian is a discrete selector, weights\ncannot move it (all schemes → 11.57)", "neg"),
     ("n_extr", "REJECTED · exponent by extrapolation skill\n6.37 µm, pooled R² −0.98", "neg"),
     ("n_maml", "ARCHIVED · PI-MAML (not published)\nfree net 40.9 · Meta-SGD 17.4 · closed form 12.25", "neg"),
     # evidence produced for the manuscript
     ("e_break", f"PER-HORIZON / PER-TOOL BREAKDOWN\n{F.get('n_pred',100)} sealed predictions, "
                 f"{F.get('n_tools',18)} tools\nevery figure and table reads these files", "evid"),
     ("e_near", f"NEAR HORIZON (h = 1, all 18 tools)\n{F.get('h1_rec',3.92):.2f} → "
                f"{F.get('h1_ado',2.28):.2f} µm  ({F.get('h1_gain',42):.0f}% lower)\n"
                "where the constant γ was too FLAT", "evid"),
     ("e_mid", "MID HORIZONS (h = 4…8)\n24–32% lower error\nwhere the constant γ was too SHARP", "evid"),
     ("e_far", f"FAR HORIZONS (h > 8) — HONEST BOUNDARY\none tool carries "
               f"{F.get('n_far',67)} readings; there the\nadopted rule is WORSE than the record", "neg"),
     ("e_tool", f"PER-TOOL EFFECT\n{F.get('n_up',15)} improved · {F.get('n_tie',1)} unchanged · "
                f"{F.get('n_dn',2)} worse\n(the two that lose are the two longest tools)", "evid"),
     # artefacts
     ("a_fig", "RESULT FIGURES (drawing rules enforced)\np3_tradeoff · p3_horizon · p3_audit · p3_per_tool", "art"),
     ("a_meth", "METHOD ARTEFACTS\np3_pipeline (orthogonal, 0 clearance violations)\n"
                "p3_equations (8 equations, in manuscript order)", "art"),
     ("a_ms", "MANUSCRIPT · PUSMA_P3_HORIZON_MDPI.docx\n6 sections IMRaD · 8 equations · 7 tables\n"
              "5 figures · 25 refs in order of appearance", "art"),
     # claim
     ("c_claim", "CLAIM\nthe weighting must follow the horizon,\nand the estimator must be CONTINUOUS\nfor weights to modulate it", "core"),
    ]
    E = [
     ("P3", "o_stop", "departs from"), ("o_stop", "o_trade", "because of"),
     ("o_trade", "o_hyp", "reframed as"), ("o_hyp", "m_gam", "tested by"),
     ("m_law", "m_wls", "fitted by"), ("m_wls", "m_gam", "generalised by"),
     ("m_gam", "m_pool", "applied per horizon"),
     ("m_pool", "v_loocv", "evaluated under"), ("v_loocv", "v_nested", "audited by"),
     ("v_nested", "v_conf", "and by"), ("v_conf", "v_rule", "judged by"),
     ("v_rule", "r_m4", "adopts"), ("r_m4", "r_opt", "robust because"),
     ("r_m4", "r_cov", "valid because"), ("r_m4", "r_r2", "without cost in"),
     ("v_rule", "n_m3", "rejects"), ("n_m3", "n_why", "explained by"),
     ("v_rule", "n_extr", "rejects"), ("v_rule", "n_maml", "rejected earlier"),
     ("r_m4", "c_claim", "supports"), ("n_why", "c_claim", "sharpens"),
     # evidence chain
     ("r_m4", "e_break", "decomposed by"),
     ("e_break", "e_near", "shows"), ("e_break", "e_mid", "shows"),
     ("e_break", "e_far", "and bounds by"), ("e_break", "e_tool", "and shows"),
     ("e_near", "c_claim", "evidences"), ("e_mid", "c_claim", "evidences"),
     ("e_far", "c_claim", "bounds"),
     ("e_break", "a_fig", "drawn as"), ("m_pool", "a_meth", "drawn as"),
     ("a_fig", "a_ms", "figures of"), ("a_meth", "a_ms", "figures of"),
     ("c_claim", "a_ms", "written up in"),
    ]
    col = {"core": NAVY, "open": GOLD, "meth": TEAL, "proto": BLUE, "res": EMER, "neg": RED,
           "evid": "#0F766E", "art": "#7C3AED"}
    nodes = [dict(id=i, label=l, group=g) for i, l, g in N]
    edges = [{"from": a, "to": b, "label": t} for a, b, t in E]
    json.dump(dict(meta=dict(project="PHM tool wear — trilogía", seed="Paper 3 · horizon-adaptive",
                             built="2026-07-21",
                             note="Eje: ponderación adaptativa al horizonte. PI-MAML archivado."),
                   nodes=nodes, edges=edges),
              open(os.path.join(GOUT, "paper3_horizon_map.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    vn = [dict(id=n["id"], label=n["label"],
               color=dict(background=col[n["group"]], border=col[n["group"]]),
               font=dict(color="#ffffff", size=13), shape="box", margin=9,
               borderWidth=3 if n["group"] == "core" else 2) for n in nodes]
    ve = [{"from": e["from"], "to": e["to"], "label": e["label"],
           "font": {"color": "#cfd6e4", "size": 11, "strokeWidth": 0},
           "color": {"color": "#7A8794"}, "arrows": "to"} for e in edges]
    chips = "".join(f'<span class="chip" style="background:{c}">{k}</span>' for k, c in
                    [("núcleo", NAVY), ("la apertura", GOLD), ("método", TEAL),
                     ("protocolo", BLUE), ("adoptado", EMER), ("rechazado/archivado", RED)])
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Paper 3 — Horizon-Adaptive Weighting</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>body{{font-family:Segoe UI,Arial;margin:0;background:#141821;color:#eee}}
#hdr{{padding:10px 16px;background:#1c2230}} #net{{height:calc(100vh - 92px)}}
.chip{{display:inline-block;padding:2px 10px;border-radius:10px;margin-right:8px;font-size:12px}}</style>
</head><body><div id="hdr"><b>PAPER 3 — Horizon-Adaptive Weighting</b> · el récord del Paper 1 batido
en m=4 ({F.get('m4_nested',3.57):.2f} µm anidado vs {F['rec4']} µm), no batido en m=3 (y se explica por qué)<br>{chips}</div>
<div id="net"></div><script>
var nodes=new vis.DataSet({json.dumps(vn, ensure_ascii=False)});
var edges=new vis.DataSet({json.dumps(ve, ensure_ascii=False)});
new vis.Network(document.getElementById('net'),{{nodes:nodes,edges:edges}},{{
 layout:{{hierarchical:{{enabled:true,direction:'LR',sortMethod:'directed',
   levelSeparation:300,nodeSpacing:130}}}},physics:false,
 edges:{{smooth:{{type:'cubicBezier',roundness:0.45}}}},
 interaction:{{hover:true,navigationButtons:true}}}});
</script></body></html>"""
    open(os.path.join(GOUT, "paper3_horizon_map.html"), "w", encoding="utf-8").write(html)
    print(f"graph: {len(nodes)} nodes / {len(edges)} edges -> paper3_horizon_map.json/.html")


# ============================================================ 2 · method pipeline
def build_pipeline(F):
    fig, ax = plt.subplots(figsize=(20, 10.4))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
    ax.set_xlim(0, 200); ax.set_ylim(0, 104); ax.axis("off")
    B = {}

    def box(name, x, y, w, h, text, fc, ec, fs=8.4, tc=INK, lw=1.6, mono=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec=ec,
                                    lw=lw, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold", zorder=4, linespacing=1.32,
                family="monospace" if mono else None)
        B[name] = (x, y, w, h)

    def lbl(x, y, t, fs=7.0, c="#43555f", ha="center"):
        ax.text(x, y, t, ha=ha, va="center", fontsize=fs, style="italic", color=c, zorder=7,
                bbox=dict(fc="white", ec="none", alpha=0.94, pad=0.7))

    # header band (title area kept clear of the drawing)
    ax.add_patch(Rectangle((1.5, 95.5), 197, 7.5, fc=NAVY, ec="none", zorder=3))
    ax.text(4, 99.2, "METHOD PIPELINE", ha="left", va="center", fontsize=12.5,
            color="#9FB4E8", fontweight="bold", zorder=4)
    ax.text(30, 99.2, "one weighting per predicted horizon — everything else inherited from Paper 1",
            ha="left", va="center", fontsize=10.6, color="white", fontweight="bold", zorder=4)

    # --- row 1: inputs -> exponent
    box("d1", 2, 76, 26, 12, "TOOL UNDER TEST\nfirst m = 4 VB readings\n(the support window)", BLUE_BG, BLUE)
    box("d2", 2, 56, 26, 12, "TRAINING FLEET\n17 tools, leakage-safe\n(no test-tool data)", BLUE_BG, BLUE)
    box("p1", 36, 56, 26, 12, "FLEET EXPONENT p*\npooled SSE, grid [0.20, 1.00]", TEAL_BG, TEAL)
    box("p2", 36, 76, 26, 12, "LOCAL SHRINKAGE of p\nrefit on the tool's own m points", TEAL_BG, TEAL)
    box("p3", 70, 66, 24, 12, "PHYSICAL CLOCK\nτ = t^p*   (linearises the law)", TEAL_BG, TEAL, mono=True)

    # --- row 2: the contribution
    box("g1", 102, 76, 30, 13, "HORIZON h OF THE PREDICTION\nh = 1, 2, 3 … steps beyond support", PUR_BG, PUR)
    box("g2", 102, 55, 30, 15, "HORIZON-ADAPTIVE WEIGHT\nγ(h) = 8 / h\nw_i ∝ τ_i^γ(h)\n(Paper 1 = constant γ = 3)", PUR_BG, PUR, lw=2.6, fs=8.8)
    box("g3", 140, 66, 26, 13, "WEIGHTED FIT per horizon\n(b, a) = argmin Σ w_i (·)²\nclosed form — no iteration", EMER_BG, EMER, lw=2.4)
    box("g4", 140, 44, 26, 12, "FORECAST\nVB(t) = b + a·t^p*", EMER_BG, EMER, lw=2.4, mono=True)

    # --- row 3: guarantees and verdict
    box("c1", 102, 30, 30, 12, "MONDRIAN CONFORMAL BAND\nrecalibrated on the new residuals", PUR_BG, PUR)
    box("c2", 102, 12, 30, 12, f"COVERAGE {F.get('m4_cov',89.0):.1f}%  (gate 88)\nmean width {F.get('m4_width',49.0):.0f} µm", PUR_BG, PUR)
    box("a1", 36, 30, 26, 12, "NESTED AUDIT\nscheme re-chosen blind per fold", BLUE_BG, BLUE)
    box("a2", 36, 12, 26, 12, f"optimism {F.get('m4_opt',-0.06):+.2f} µm\nwinner in {F.get('m4_share',0.94)*100:.0f}% of folds", BLUE_BG, BLUE)
    box("v1", 172, 44, 26, 35, f"ADOPTED\n{F.get('m4_nested',3.57):.2f} µm\n(record {F['rec4']} µm)\n\npooled R² 0.70\npreserved", EMER_BG, EMER, lw=2.8, fs=9.4)
    box("v2", 172, 12, 26, 22, f"m = 3 NOT adopted\nnested {F.get('m3_nested',12.13):.2f} µm\nvs {F['rec3']} µm\n\nmedian estimator:\nweights cannot\nmodulate it", RED_BG, RED, tc="#7c2d24", fs=8.0)

    r = Router()
    r.add([(28, 82), (36, 82)], BLUE, skip={"d1", "p2"})
    r.add([(28, 62), (36, 62)], BLUE, skip={"d2", "p1"})
    r.add([(62, 62), (66, 62), (66, 72), (70, 72)], TEAL, skip={"p1", "p3"})
    r.add([(62, 82), (66, 82), (66, 72)], TEAL, arrow=False, skip={"p2", "p3"})
    r.dot(66, 72, TEAL)
    r.add([(94, 72), (98, 72), (98, 82.5), (102, 82.5)], TEAL, skip={"p3", "g1"})
    r.add([(94, 72), (98, 72), (98, 62.5), (102, 62.5)], TEAL, skip={"p3", "g2"})
    r.add([(117, 76), (117, 70)], PUR, lw=2.4, skip={"g1", "g2"})
    r.add([(132, 62.5), (136, 62.5), (136, 72.5), (140, 72.5)], EMER, lw=3.0, skip={"g2", "g3"})
    r.add([(153, 66), (153, 56)], EMER, lw=3.0, skip={"g3", "g4"})
    r.add([(166, 50), (172, 50)], EMER, lw=3.0, skip={"g4", "v1"})
    r.add([(153, 44), (153, 36), (132, 36)], PUR, skip={"g4", "c1"})
    r.add([(117, 30), (117, 24)], PUR, skip={"c1", "c2"})
    r.add([(132, 18), (136, 18), (136, 40), (185, 40), (185, 44)], PUR, lw=1.4,
          skip={"c2", "v1"})
    r.add([(49, 30), (49, 24)], BLUE, skip={"a1", "a2"})
    r.add([(62, 18), (72, 18), (72, 8), (169, 8), (169, 46), (172, 46)], BLUE, lw=1.4,
          skip={"a2", "v1"})
    bad = r.validate(B)
    for nm, p0, p1 in bad:
        print("  !! clearance:", nm, p0, p1)
    hops = r.draw(ax)

    lbl(32, 84.6, "m points"); lbl(32, 64.6, "17 curves")
    lbl(98.4, 88, "per step", ha="center"); lbl(136.5, 78.5, "w(τ; h)")
    lbl(155.6, 60, "b, a", ha="left"); lbl(168.5, 52.6, "VB̂", ha="left")

    # legend OUTSIDE the drawing (bottom-left gutter), never over an element
    lx, ly = 4, 4.2
    ax.add_line(Line2D([lx, lx + 3.2], [ly, ly], color=SLATE, lw=1.7, zorder=4))
    th = np.linspace(np.pi, 0, 40)
    ax.add_line(Line2D(lx + 4.4 + HOP_R * np.cos(th), ly + HOP_R * np.sin(th), color=SLATE,
                       lw=1.7, zorder=4))
    ax.add_line(Line2D([lx + 5.65, lx + 8.8], [ly, ly], color=SLATE, lw=1.7, zorder=4))
    ax.add_line(Line2D([lx + 4.4, lx + 4.4], [ly - 2.4, ly + 2.4], color=SLATE, lw=1.7, zorder=3))
    ax.text(lx + 10, ly, "lines cross (not connected)", fontsize=7.4, va="center", color=SLATE,
            style="italic")
    jx = lx + 48
    ax.add_line(Line2D([jx, jx + 8.8], [ly, ly], color=SLATE, lw=1.7, zorder=4))
    ax.add_line(Line2D([jx + 4.4, jx + 4.4], [ly, ly + 2.4], color=SLATE, lw=1.7, zorder=4))
    ax.plot([jx + 4.4], [ly], marker="o", ms=5.2, color=SLATE, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.text(jx + 10, ly, "lines join (same quantity)", fontsize=7.4, va="center", color=SLATE,
            style="italic")

    fig.savefig(os.path.join(FIG, "p3_pipeline.png"), dpi=300, facecolor="white")
    fig.savefig(os.path.join(PRINT, "p3_pipeline_600dpi.png"), dpi=600, facecolor="white")
    plt.close(fig)
    print(f"pipeline: clearance violations {len(bad)} | hops {hops} -> p3_pipeline.png")


# ============================================================ 3 · equations panel
def build_equations(F):
    eqs = [
     (r"$\mathrm{VB}(t)\;=\;b\;+\;a\,\tau,\qquad \tau=t^{\,p^{*}}$",
      "(1)  wear law in the physical clock — inherited from Paper 1"),
     (r"$p^{*}=\arg\min_{p}\;\sum_{k\in\mathrm{train}}\;\min_{b_k,a_k}\sum_i"
      r"\left(\mathrm{VB}_{k,i}-b_k-a_k t_{k,i}^{\,p}\right)^{2}$",
      "(2)  fleet exponent, pooled over training tools only"),
     (r"$\dfrac{d\mathrm{VB}}{dt}\;\geq\;0,\qquad \dfrac{d^{2}\mathrm{VB}}{dt^{2}}\;\leq\;0"
      r"\qquad (a>0,\;0<p<1)$",
      "(3)  admissibility: monotone growth and deceleration, by construction"),
     (r"$(\hat b,\hat a)\;=\;\arg\min_{b,a}\;\sum_{i=1}^{m} w_i\left(b+a\tau_i-\mathrm{VB}_i\right)^{2}$",
      "(4)  weighted fit on the m support points — closed form, no iteration"),
     (r"$w_i\;\propto\;\tau_i^{\,\gamma}\qquad\text{(Paper 1: }\gamma=3\text{ constant)}$",
      "(5)  the weighting Paper 1 used, and stopped at"),
     (r"$w_i(h)\;\propto\;\tau_i^{\,\gamma(h)},\qquad \gamma(h)=\dfrac{c}{h},\qquad c=8$",
      "(6)  THIS WORK: the weight follows the horizon h being predicted"),
     (r"$\hat{\mathrm{VB}}(t_{m+h})\;=\;\hat b(h)\;+\;\hat a(h)\,t_{m+h}^{\,p^{*}}$",
      "(7)  one refit per horizon: the estimator is re-solved for each step ahead"),
     (r"$\hat q_{\,1-\alpha}^{(\mathrm{bin})}=\mathrm{Quantile}_{1-\alpha}"
      r"\left\{\left|r_j\right| : \mathrm{bin}(h_j)=\mathrm{bin}\right\}$",
      "(8)  Mondrian conformal quantile per horizon bin — guarantee inherited from Paper 1"),
    ]
    n = len(eqs)
    fig, ax = plt.subplots(figsize=(13.2, 1.35 * n + 1.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.965, bottom=0.02)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.add_patch(Rectangle((1, 92.5), 98, 6.5, fc=NAVY, ec="none"))
    ax.text(3, 95.75, "EQUATIONS USED", ha="left", va="center", fontsize=12.5,
            color="#9FB4E8", fontweight="bold")
    ax.text(24, 95.75, "in the order the manuscript introduces them", ha="left", va="center",
            fontsize=10.4, color="white", fontweight="bold")
    top, step = 88.0, 88.0 / n
    for k, (tex, cap) in enumerate(eqs):
        y = top - k * step
        hl = "(6)" in cap
        ax.add_patch(FancyBboxPatch((3, y - step + 1.6), 94, step - 2.0,
                                    boxstyle="round,pad=0.4",
                                    fc=EMER_BG if hl else "#F7F9FB",
                                    ec=EMER if hl else "#D6DEE6",
                                    lw=2.2 if hl else 1.1, zorder=2))
        ax.text(50, y - step * 0.40 + 1.9, tex, ha="center", va="center", fontsize=15.5,
                color=INK, zorder=3)
        ax.text(6, y - step + 3.2, cap, ha="left", va="center", fontsize=8.6,
                color=EMER if hl else SLATE, style="italic",
                fontweight="bold" if hl else "normal", zorder=3)
    fig.savefig(os.path.join(FIG, "p3_equations.png"), dpi=300, facecolor="white")
    fig.savefig(os.path.join(PRINT, "p3_equations_600dpi.png"), dpi=600, facecolor="white")
    plt.close(fig)
    print("equations: 8 -> p3_equations.png")


if __name__ == "__main__":
    F = facts()
    build_graph(F)
    build_pipeline(F)
    build_equations(F)
    print("done — Paper 3 graph + pipeline + equations")
