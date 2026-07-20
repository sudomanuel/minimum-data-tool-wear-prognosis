# -*- coding: utf-8 -*-
"""make_paper3_graph.py — Paper 3 knowledge graph, seeded from the trilogy master map.

Input : graphify-out/paper1_trilogy_map.json  (master; carries the P1 sections, results and the
        five P3 bridge nodes the user curated)
Output: graphify-out/paper3_pimaml_map.json + .html

The master supplies the INHERITED layer (what Paper 3 stands on: P1 sections, records, and the
bridges b3_*). This script adds the OWN layer of Paper 3: the meta-task distribution, the
support/query interface, the two optimisation loops, the validation protocols, the comparators
and the result slots — so the graph is the skeleton the manuscript is written from.
"""
import json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
GOUT_CANDIDATES = [r"D:/KSF/PHM/phm_tool_wear/graphify-out",
                   os.path.join(ROOT, "graphify-out")]
GOUT = next((p for p in GOUT_CANDIDATES if os.path.isdir(p)), GOUT_CANDIDATES[0])
MASTER = os.path.join(GOUT, "paper1_trilogy_map.json")

COL = {"core": "#141E4F", "inherit": "#1F5FA8", "task": "#0E7490", "loop": "#6B21A8",
       "proto": "#B7791F", "cmp": "#7A8794", "result": "#0E7A4D", "risk": "#B03A2E"}

# ---------------- own layer of Paper 3 ----------------
NODES = [
 ("P3", "PAPER 3 · PI-MAML\nPhysics-Informed Meta-Learning\npara adaptación rápida cross-condition", "core"),

 # --- §3.1 task distribution -------------------------------------------------
 ("t_def", "§3.1 Tarea = (útil i, ventana w, horizonte h)\nNO 'un útil = una tarea'", "task"),
 ("t_real", "𝒯_real · sub-ventanas de los 18 útiles\ntodas las triples admisibles", "task"),
 ("t_phys", "𝒯_phys · generador físico\n(b,a,p,σ) ~ posterior de flota → ec.(2)", "task"),
 ("t_mix", "𝒯 = λ·𝒯_real + (1−λ)·𝒯_phys\nλ fijado solo en meta-entrenamiento", "task"),
 ("t_epi", "DISTINCIÓN EPISTÉMICA\nsintético NO informa de un útil concreto\npero SÍ enseña a adaptarse", "risk"),

 # --- §3.2 interface ---------------------------------------------------------
 ("q_sq", "§3.2 Support S_i (m=3,4) vs Query Q_i\nmisma interfaz de despliegue que P1", "task"),

 # --- §3.3–3.4 loops ---------------------------------------------------------
 ("l_in", "§3.3 Bucle interno · K≤3 pasos\nL_in = MSE + η₁·monotonía + η₂·desaceleración", "loop"),
 ("l_phys", "Penalizaciones = forma diferencial\nde las restricciones de P1 (ec. 3)", "loop"),
 ("l_out", "§3.4 Bucle externo · meta-update\nL_out = error en Q_i tras adaptar", "loop"),
 ("l_init", "Inicialización meta-aprendida θ\n(+ tasas α, β)", "loop"),
 ("l_alg", "§3.5 Algoritmo PI-MAML + cabeza RUL\nconformal y hazard heredados de P1", "loop"),

 # --- §4 protocols -----------------------------------------------------------
 ("v_loocv", "§4.2a LOOCV por útil · 18 pliegues\ntodas las tareas del útil excluido fuera", "proto"),
 ("v_lolo", "§4.2b LOLO · dejar-un-NIVEL-fuera\nvc / f / refrigeración — 8 pliegues", "proto"),
 ("v_degen", "LOCO ≡ LOOCV (degenerado)\n1 útil por condición ⇒ LOLO es el test real", "risk"),
 ("v_rule", "Regla de adopción preestablecida\nbatir 11.0/5.6 µm con cobertura válida", "proto"),

 # --- §4.3 comparators -------------------------------------------------------
 ("c_p1", "Récord P1 · 11.0 µm (m=3) · 5.6 µm (m=4)", "cmp"),
 ("c_maml", "MAML estándar (sin física, η=0)", "cmp"),
 ("c_ft", "Fine-tuning de red agrupada", "cmp"),
 ("c_fleet", "Baseline de curva promedio · 18.7 µm", "cmp"),
 ("c_abl", "Ablaciones · λ=1 / λ=0 / η=0 / K=0..3", "cmp"),

 # --- §5 results (slots filled by the experiment) ----------------------------
 ("r_adapt", "Curva de adaptación\nerror vs. pasos K (K=0 aísla la inicialización)", "result"),
 ("r_loocv", "Resultado LOOCV\n[llenado por run_p3_pimaml.py]", "result"),
 ("r_lolo", "Resultado LOLO cross-condition\n[llenado por run_p3_pimaml.py]", "result"),
 ("r_verdict", "VEREDICTO bajo la regla\nadoptar / no adoptar (ambos publicables)", "result"),
]

EDGES = [
 ("P3", "t_def", "formula"), ("t_def", "t_real", "instancia"), ("t_def", "t_phys", "instancia"),
 ("t_real", "t_mix", "mezcla"), ("t_phys", "t_mix", "mezcla"), ("t_phys", "t_epi", "se defiende con"),
 ("t_mix", "q_sq", "materializa"), ("q_sq", "l_in", "alimenta"),
 ("l_in", "l_phys", "restringida por"), ("l_in", "l_out", "anidado en"),
 ("l_out", "l_init", "produce"), ("l_init", "l_alg", "desplegada en"),
 ("l_alg", "v_loocv", "evaluado por"), ("l_alg", "v_lolo", "evaluado por"),
 ("v_lolo", "v_degen", "justificado por"), ("v_loocv", "v_rule", "juzgado por"),
 ("v_lolo", "v_rule", "juzgado por"),
 ("v_rule", "c_p1", "contra"), ("v_rule", "c_maml", "contra"), ("v_rule", "c_ft", "contra"),
 ("v_rule", "c_fleet", "contra"), ("v_rule", "c_abl", "contra"),
 ("v_loocv", "r_loocv", "produce"), ("v_lolo", "r_lolo", "produce"),
 ("l_in", "r_adapt", "produce"), ("r_loocv", "r_verdict", "decide"),
 ("r_lolo", "r_verdict", "decide"), ("r_adapt", "r_verdict", "decide"),
]

# inherited anchors: (P3 own node) -> (master node id), label
INHERIT = [
 ("t_def", "b3_gran", "resuelve la limitación"),
 ("t_real", "b3_sub", "realiza el puente"),
 ("t_phys", "b3_gen", "realiza el puente"),
 ("l_phys", "b3_gen", "restricciones de"),
 ("l_init", "b3_prior", "realiza el puente"),
 ("v_rule", "r_neg", "hereda la regla de"),
 ("c_p1", "r_rec", "es el récord"),
 ("t_epi", "r_neg", "acotado por (augmentación ≠ replicación)"),
 ("l_alg", "r_band", "hereda la banda"),
 ("l_alg", "r_safe", "hereda el umbral"),
 ("r_verdict", "b3_zero", "avanza hacia"),
]


def build():
    master = json.load(open(MASTER, encoding="utf-8"))
    mnodes = {n["id"]: n for n in master["nodes"]}

    nodes, edges = [], []
    for nid, label, grp in NODES:
        nodes.append(dict(id=nid, label=label, group=grp))
    for a, b, lab in EDGES:
        edges.append(dict({"from": a, "to": b, "label": lab}))

    # pull the inherited master nodes actually referenced, marked as inherited
    used = sorted({m for _, m, _ in INHERIT})
    for mid in used:
        m = mnodes.get(mid)
        if not m:
            print("  !! master node missing:", mid); continue
        nodes.append(dict(id="M_" + mid, label="↰ P1 · " + m["label"], group="inherit"))
    for own, mid, lab in INHERIT:
        edges.append({"from": own, "to": "M_" + mid, "label": lab})

    out_json = os.path.join(GOUT, "paper3_pimaml_map.json")
    json.dump(dict(meta=dict(project="PHM tool wear — trilogía", seed="Paper 3 (from Paper 1 master)",
                             built="2026-07-20",
                             note="Grafo del Paper 3; la capa 'inherit' son anclas al mapa maestro"),
                   nodes=nodes, edges=edges),
              open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    vn = [dict(id=n["id"], label=n["label"],
               color=dict(background=COL[n["group"]], border=COL[n["group"]]),
               font=dict(color="#ffffff", size=13), shape="box", margin=8,
               borderWidth=3 if n["group"] == "core" else 2) for n in nodes]
    ve = [dict({"from": e["from"], "to": e["to"], "label": e.get("label", ""),
                "font": {"color": "#cfd6e4", "size": 11, "strokeWidth": 0},
                "color": {"color": "#7A8794"}, "arrows": "to"}) for e in edges]
    chips = "".join(
        f'<span class="chip" style="background:{c}">{k}</span>' for k, c in
        [("núcleo", COL["core"]), ("heredado de P1", COL["inherit"]), ("meta-tareas", COL["task"]),
         ("bucles", COL["loop"]), ("protocolos", COL["proto"]), ("comparadores", COL["cmp"]),
         ("resultados", COL["result"]), ("riesgo/defensa", COL["risk"])])
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Paper 3 — PI-MAML (grafo derivado del maestro)</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>body{{font-family:Segoe UI,Arial;margin:0;background:#141821;color:#eee}}
#hdr{{padding:10px 16px;background:#1c2230}} #net{{height:calc(100vh - 92px)}}
.chip{{display:inline-block;padding:2px 10px;border-radius:10px;margin-right:8px;font-size:12px}}</style>
</head><body>
<div id="hdr"><b>PAPER 3 — PI-MAML · grafo derivado del mapa maestro del Paper 1</b><br>{chips}</div>
<div id="net"></div>
<script>
var nodes = new vis.DataSet({json.dumps(vn, ensure_ascii=False)});
var edges = new vis.DataSet({json.dumps(ve, ensure_ascii=False)});
new vis.Network(document.getElementById('net'), {{nodes:nodes, edges:edges}}, {{
  layout:{{hierarchical:{{enabled:true, direction:'LR', sortMethod:'directed',
           levelSeparation:260, nodeSpacing:110}}}},
  physics:false, edges:{{smooth:{{type:'cubicBezier', roundness:0.45}}}},
  interaction:{{hover:true, navigationButtons:true}}}});
</script></body></html>"""
    out_html = os.path.join(GOUT, "paper3_pimaml_map.html")
    open(out_html, "w", encoding="utf-8").write(html)
    print(f"nodes {len(nodes)} (own {len(NODES)} + inherited {len(used)}) | edges {len(edges)}")
    print("wrote", out_json)
    print("wrote", out_html)


if __name__ == "__main__":
    build()
