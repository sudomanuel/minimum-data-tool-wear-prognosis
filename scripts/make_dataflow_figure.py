# -*- coding: utf-8 -*-
"""make_dataflow_figure.py — v3 CIRCUIT-STYLE, highly detailed end-to-end data-flow map.

v3 changes (supervisor/user review):
  * ORTHOGONAL (Manhattan) connectors only — no curved arcs.
  * NO connector may touch or graze a box it does not connect to: routes run in dedicated
    corridors and a validator asserts >= CLEAR units of clearance against every other box.
  * Crossings are disambiguated like an electrical schematic:
      - a HOP (semicircular jump) where two nets merely cross without connecting;
      - a filled JUNCTION DOT where lines of the same net genuinely join.

Content source of truth: graphify pipeline skeleton + real artifacts/numbers of the study.
EN + ES editions, 300 dpi, full-bleed axes.
"""
import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "outputs", "figures")

NAVY = "#141E4F"
INK = "#1a2a36"; SLATE = "#5a6b7a"
EMER = "#0E7A4D"; EMER_BG = "#EAF6F0"
GREY_BG = "#F2F4F6"; GREY_LN = "#7A8794"
BLUE = "#1F5FA8"; BLUE_BG = "#EDF3FA"
TEAL = "#0E7490"; TEAL_BG = "#E9F5F8"
RED = "#B03A2E"; RED_BG = "#FDF0EE"
PUR = "#5B21B6"; PUR_BG = "#F4EFFC"
GOLD = "#B7791F"

CLEAR = 1.4      # minimum clearance (canvas units) between a connector and a foreign box
HOP_R = 1.25     # hop radius

T = {
 "EN": dict(
   objective_tag="OBJECTIVE",
   objective=("From no more than 4 microscope readings of a brand-new tool: its full wear curve, a "
              "guaranteed 90% band and a chipping-safe\nstop rule — then the rest of the tool's life "
              "runs measurement-free."),
   phases=[("01", "ACQUIRE"), ("02", "ENGINEER"), ("03", "PERSONALISE"),
           ("04", "GUARANTEE"), ("05", "DECIDE")],
   payoff_tag="WHY IT PAYS",
   kpis=[("5.6 µm", "future-wear error from 4 readings\n= 2.8% of the 200 µm budget"),
         ("90%", "guaranteed band · ±19 µm\nfor the next cycle"),
         ("167 µm", "chipping-safe stop\none-cycle risk ≤ 10%"),
         ("94%", "of 16 real RUL events\ninside the calibrated window"),
         ("95%", "of tool life runs with\nzero further measurements")],
   a1="Hanwha XD26II-V Swiss-type centre\n18 tools = full 3×3×2 DOE\nvc 55/70/80 · f 0.08/0.20/0.30\ndry vs. coolant · ONE tool per recipe",
   a2="Axial (A) + radial (R) accelerometers\none burst per contact, 6 per cycle\n[AE recorded, set aside]",
   a3="Keyence VHX-7000 microscope\nVB (µm) at cycle stops, ex situ\n≈5 inspections per tool",
   b1="RAW SIGNAL STORE\nper-contact TXT bursts\nA + R channels",
   b2="VB TARGET TABLE\n172 rows (tool, cycle, VB)\n+ 18 chipping values 127–291 µm",
   c1="Signal QA + segmentation\nburst isolation per contact",
   c2="FEATURE ENGINEERING\n294 features per cut:\nRMS · signal/band/wavelet energies\nkurtosis · crest factor\nRMS & dominant freq × A, R",
   c3="features_experiment.csv\n172 × 296 numeric",
   wall="L O O C V   W A L L",
   wallsub="18 folds · one complete tool held out · only its first m readings visible · future sealed\nscaling / selection / tuning / calibration fitted INSIDE the training folds only",
   cur="Curation 294→181",
   s1="Consensus selection\n(in-fold)",
   s2="Fold-safe\naugmentation",
   s3="Nested\ntuning",
   s4="A/R fusion\n(6 strategies)",
   s5="Regressors: ridge · LASSO · elastic net · random forest · XGBoost · MLP · PLS\n+ physics-equation hybrids (eq. 11): fleet-law residuals · τ-regressor",
   sout="CROSS-TOOL VB READING FAILS\nbest R² ≈ 0 (radial + PLS), all ≤ 0\nsignal lives WITHIN a tool\n(prognosability 0.04)",
   p1="FLEET WEAR SHAPE\ngrid p ∈ [0.05, 0.95] · pooled SSE\n17 training tools → p* ≈ 0.20",
   p2="PERSONALISE, m = 3–4\nτ = t^p* · Theil–Sen / Siegel\nweights w ∝ τ^γ, γ = 3\nguard fires only > 5·σ_meas",
   p3="PERSONAL WEAR CURVE\nVB(t) = b + a·t^p*\nmonotone + decelerating (eq. 3)\nfrom VB readings ONLY",
   k1="CONFORMAL GUARANTEE\n≈118 OOF residuals → Mondrian\nbins ≤1 · 2–3 · ≥4 → ±19 µm near\nmean 52.5 µm @ 90.1%",
   k2="CHIPPING HAZARD\nlogit h(VB), 172 cycles + 18 events\nh ≤ 0.10 → VB_safe ≈ 167 µm\n(P10 cross-check: 135 µm)",
   doe="DOE inference (Lenth · ANOVA)\ncooling moves LEVELS — context",
   d3="ONLINE MONITOR\nKalman, τ-clock, event-triggered\nnext cut 3.7–4.0 µm → alarm",
   d1="RUL WINDOW\nladder {120,150,175,200} µm\n16 events · 94% covered",
   d2="SAFE-STOP DECISION\nstop when UPPER band edge\ncrosses VB_safe — planned change,\nno surprise chipping",
   lane_s="SENSOR BRANCH — the obvious route, given every chance",
   lane_p="PHYSICS BRANCH — the deployed route",
   e_sig="vibration", e_vb="VB values", e_burst="6 bursts/cycle", e_feats="294 feats/cut",
   e_rows="172 × 296", e_curves="17 training VB curves", e_m="first m = 3–4 readings",
   e_ba="(b, a)", e_curve="VB(t)", e_band="curve + band", e_risk="h(VB)",
   e_events="18 chipping events", e_null="no transferable signal", e_alarm="alarm",
   e_kurt="radial spectral kurtosis →\nexploratory risk covariate (p = 0.010)",
   legend_hop="lines cross (not connected)", legend_dot="lines join (same data)",
 ),
 "ES": dict(
   objective_tag="OBJETIVO",
   objective=("Con no más de 4 lecturas de microscopio de una herramienta nueva: su curva completa de "
              "desgaste, una banda garantizada al 90%\ny una regla de parada segura ante astillado — y "
              "el resto de la vida corre sin mediciones."),
   phases=[("01", "ADQUIRIR"), ("02", "INGENIERÍA"), ("03", "PERSONALIZAR"),
           ("04", "GARANTIZAR"), ("05", "DECIDIR")],
   payoff_tag="POR QUÉ PAGA",
   kpis=[("5.6 µm", "error de desgaste futuro con 4 lecturas\n= 2.8% del presupuesto de 200 µm"),
         ("90%", "banda garantizada · ±19 µm\npara el próximo ciclo"),
         ("167 µm", "parada segura ante astillado\nriesgo por ciclo ≤ 10%"),
         ("94%", "de 16 eventos RUL reales\ndentro de la ventana calibrada"),
         ("95%", "de la vida corre con\ncero mediciones adicionales")],
   a1="Centro suizo Hanwha XD26II-V\n18 útiles = DOE completo 3×3×2\nvc 55/70/80 · f 0.08/0.20/0.30\nseco vs. refrigerante · 1 útil por receta",
   a2="Acelerómetros axial (A) + radial (R)\nuna ráfaga por contacto, 6 por ciclo\n[EA registrada, apartada]",
   a3="Microscopio Keyence VHX-7000\nVB (µm) en paradas de ciclo, ex situ\n≈5 inspecciones por útil",
   b1="ALMACÉN DE SEÑAL CRUDA\nráfagas TXT por contacto\ncanales A + R",
   b2="TABLA DE TARGETS VB\n172 filas (útil, ciclo, VB)\n+ 18 astillados 127–291 µm",
   c1="QA de señal + segmentación\naislamiento de ráfaga por contacto",
   c2="INGENIERÍA DE CARACTERÍSTICAS\n294 por corte:\nRMS · energías señal/banda/wavelet\ncurtosis · factor de cresta\nfrec. RMS y dominante × A, R",
   c3="features_experiment.csv\n172 × 296 numéricas",
   wall="M U R O   L O O C V",
   wallsub="18 pliegues · un útil completo fuera · solo sus primeras m lecturas visibles · futuro sellado\nescalado / selección / tuning / calibración SOLO dentro del pliegue de entrenamiento",
   cur="Curación 294→181",
   s1="Selección por consenso\n(en-pliegue)",
   s2="Augmentación\nfold-safe",
   s3="Tuning\nanidado",
   s4="Fusión A/R\n(6 estrategias)",
   s5="Regresores: ridge · LASSO · elastic net · bosque aleatorio · XGBoost · MLP · PLS\n+ híbridos con la ecuación física (ec. 11): residuos de flota · regresor τ",
   sout="LEER VB ENTRE ÚTILES FALLA\nmejor R² ≈ 0 (radial + PLS), todo ≤ 0\nla señal vive DENTRO de cada útil\n(prognosabilidad 0.04)",
   p1="FORMA DE DESGASTE DE FLOTA\ngrilla p ∈ [0.05, 0.95] · SSE agrupado\n17 útiles de entrenamiento → p* ≈ 0.20",
   p2="PERSONALIZAR, m = 3–4\nτ = t^p* · Theil–Sen / Siegel\npesos w ∝ τ^γ, γ = 3\nguarda solo si > 5·σ_medición",
   p3="CURVA DE DESGASTE PERSONAL\nVB(t) = b + a·t^p*\nmonótona + desacelerante (ec. 3)\nSOLO con lecturas VB",
   k1="GARANTÍA CONFORME\n≈118 residuos OOF → Mondrian\nceldas ≤1 · 2–3 · ≥4 → ±19 µm cerca\nmedia 52.5 µm @ 90.1%",
   k2="HAZARD DE ASTILLADO\nlogit h(VB), 172 ciclos + 18 eventos\nh ≤ 0.10 → VB_safe ≈ 167 µm\n(verificación P10: 135 µm)",
   doe="Inferencia DOE (Lenth · ANOVA)\nrefrigeración mueve NIVELES — contexto",
   d3="MONITOR EN LÍNEA\nKalman, reloj τ, por eventos\npróximo corte 3.7–4.0 µm → alarma",
   d1="VENTANA RUL\nescalera {120,150,175,200} µm\n16 eventos · 94% cubiertos",
   d2="DECISIÓN DE PARADA SEGURA\nparar cuando el borde SUPERIOR\ncruza VB_safe — cambio planificado,\nsin astillado sorpresa",
   lane_s="RAMA DE SENSORES — la ruta obvia, con todas las oportunidades",
   lane_p="RAMA FÍSICA — la ruta desplegada",
   e_sig="vibración", e_vb="valores VB", e_burst="6 ráfagas/ciclo", e_feats="294 caract./corte",
   e_rows="172 × 296", e_curves="17 curvas VB de entren.", e_m="primeras m = 3–4 lecturas",
   e_ba="(b, a)", e_curve="VB(t)", e_band="curva + banda", e_risk="h(VB)",
   e_events="18 eventos de astillado", e_null="sin señal transferible", e_alarm="alarma",
   e_kurt="curtosis espectral radial →\ncovariable exploratoria (p = 0.010)",
   legend_hop="las líneas se cruzan (no conectan)", legend_dot="las líneas se unen (mismo dato)",
 ),
}


# ============================ orthogonal router ============================
class Router:
    """Manhattan connectors with schematic hops at crossings and junction dots."""

    def __init__(self):
        self.nets = []      # dict(pts, color, lw, dashed, arrow, skip)
        self.dots = []      # (x, y, color)
        self._hop_pts = []

    def add(self, pts, color=GREY_LN, lw=1.7, dashed=False, arrow=True, skip=()):
        # enforce strict orthogonality
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            assert abs(x1 - x0) < 1e-9 or abs(y1 - y0) < 1e-9, f"non-orthogonal segment {pts}"
        self.nets.append(dict(pts=[tuple(map(float, p)) for p in pts], color=color, lw=lw,
                              dashed=dashed, arrow=arrow, skip=set(skip)))

    def dot(self, x, y, color=GREY_LN):
        self.dots.append((x, y, color))

    # ---------- validation: no connector may graze a foreign box ----------
    def validate(self, boxes):
        bad = []
        for net in self.nets:
            for (x0, y0), (x1, y1) in zip(net["pts"], net["pts"][1:]):
                sx0, sx1 = sorted((x0, x1)); sy0, sy1 = sorted((y0, y1))
                for name, (bx, by, bw, bh) in boxes.items():
                    if name in net["skip"]:
                        continue
                    # inflate the box by CLEAR and test overlap with the segment's bbox
                    ix0, ix1 = bx - CLEAR, bx + bw + CLEAR
                    iy0, iy1 = by - CLEAR, by + bh + CLEAR
                    if sx0 < ix1 and sx1 > ix0 and sy0 < iy1 and sy1 > iy0:
                        bad.append((name, (x0, y0), (x1, y1)))
        return bad

    # ---------- crossing detection ----------
    def _crossings(self):
        segs = []
        for ni, net in enumerate(self.nets):
            for p0, p1 in zip(net["pts"], net["pts"][1:]):
                segs.append((ni, p0, p1, abs(p1[1] - p0[1]) < 1e-9))
        hops = defaultdict(list)          # (net, seg_idx_within_net) -> [x of crossing]
        idx_in_net = defaultdict(int)
        keyed = []
        for ni, p0, p1, horiz in segs:
            keyed.append((ni, idx_in_net[ni], p0, p1, horiz))
            idx_in_net[ni] += 1
        for ni, si, p0, p1, horiz in keyed:
            if not horiz:
                continue
            hy = p0[1]; hx0, hx1 = sorted((p0[0], p1[0]))
            for nj, sj, q0, q1, vhoriz in keyed:
                if vhoriz or nj == ni:
                    continue
                vx = q0[0]; vy0, vy1 = sorted((q0[1], q1[1]))
                if hx0 + HOP_R < vx < hx1 - HOP_R and vy0 + 0.4 < hy < vy1 - 0.4:
                    hops[(ni, si)].append(vx)
                    self._hop_pts.append((vx, hy))
        return hops

    # ---------- drawing ----------
    def draw(self, ax):
        hops = self._crossings()
        n_hops = 0
        for ni, net in enumerate(self.nets):
            ls = (0, (5, 3)) if net["dashed"] else "solid"
            pts = net["pts"]
            for si, (p0, p1) in enumerate(zip(pts, pts[1:])):
                horiz = abs(p1[1] - p0[1]) < 1e-9
                xs = hops.get((ni, si), [])
                last = si == len(pts) - 2
                if horiz and xs:
                    n_hops += len(xs)
                    y = p0[1]
                    going_right = p1[0] > p0[0]
                    order = sorted(xs) if going_right else sorted(xs, reverse=True)
                    cur = p0[0]
                    for cx in order:
                        a = cx - HOP_R if going_right else cx + HOP_R
                        ax.add_line(Line2D([cur, a], [y, y], color=net["color"],
                                           lw=net["lw"], ls=ls, zorder=2,
                                           solid_capstyle="round"))
                        th = np.linspace(np.pi, 0, 40) if going_right else np.linspace(0, np.pi, 40)
                        ax.add_line(Line2D(cx + HOP_R * np.cos(th), y + HOP_R * np.sin(th),
                                           color=net["color"], lw=net["lw"], zorder=2))
                        cur = cx + HOP_R if going_right else cx - HOP_R
                    ax.add_line(Line2D([cur, p1[0]], [y, y], color=net["color"], lw=net["lw"],
                                       ls=ls, zorder=2, solid_capstyle="round"))
                else:
                    ax.add_line(Line2D([p0[0], p1[0]], [p0[1], p1[1]], color=net["color"],
                                       lw=net["lw"], ls=ls, zorder=2, solid_capstyle="round"))
                if last and net["arrow"]:
                    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                    n = max(abs(dx), abs(dy))
                    ux, uy = (dx / n, dy / n) if n else (1, 0)
                    ax.annotate("", xy=p1, xytext=(p1[0] - ux * 1.6, p1[1] - uy * 1.6),
                                arrowprops=dict(arrowstyle="-|>", color=net["color"],
                                                lw=net["lw"], shrinkA=0, shrinkB=0,
                                                mutation_scale=13), zorder=5)
        for x, y, c in self.dots:
            ax.plot([x], [y], marker="o", ms=5.4, color=c, zorder=6,
                    markeredgecolor="white", markeredgewidth=0.8)
        return n_hops


def make(lang):
    L = T[lang]
    fig, ax = plt.subplots(figsize=(24, 12.8))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
    ax.set_xlim(0, 240); ax.set_ylim(0, 128); ax.axis("off")

    B = {}   # name -> (x, y, w, h)

    def box(name, x, y, w, h, text, fc, ec, fs=8.2, tc=INK, lw=1.5, mono=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", zorder=4, linespacing=1.32,
                family="monospace" if mono else None)
        B[name] = (x, y, w, h)

    def lbl(x, y, text, fs=7.0, color="#43555f", ha="center"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=fs, style="italic", color=color,
                zorder=7, bbox=dict(fc="white", ec="none", alpha=0.94, pad=0.7))

    # ---------------- banner / rail / lanes ----------------
    ax.add_patch(Rectangle((1.5, 118.5), 237, 8.4, fc=NAVY, ec="none", zorder=3))
    ax.text(4.5, 122.7, L["objective_tag"], ha="left", va="center", fontsize=13,
            color="#9FB4E8", fontweight="bold", zorder=4)
    ax.text(29, 122.7, L["objective"], ha="left", va="center", fontsize=10.6, color="white",
            fontweight="bold", zorder=4, linespacing=1.5)
    for (x0, x1), (num, name) in zip([(1.5, 56), (60, 84), (92, 152), (156, 190), (194, 238.5)],
                                     L["phases"]):
        ax.plot([x0, x1], [114.6, 114.6], color=NAVY, lw=1.1, zorder=3)
        ax.text(x0, 116.2, num, fontsize=10, color=GOLD, fontweight="bold", zorder=4)
        ax.text(x0 + 4.5, 116.2, name, fontsize=9, color=NAVY, fontweight="bold", zorder=4)
    ax.add_patch(Rectangle((92, 76), 146.5, 32, fc=GREY_BG, ec="none", alpha=0.5, zorder=0))
    ax.add_patch(Rectangle((92, 16), 146.5, 56, fc=EMER_BG, ec="none", alpha=0.32, zorder=0))
    ax.text(237, 110.4, L["lane_s"], ha="right", fontsize=8.6, color=SLATE, fontweight="bold")
    ax.text(237, 17.6, L["lane_p"], ha="right", fontsize=8.6, color=EMER, fontweight="bold")

    # ---------------- boxes ----------------
    box("a1", 2, 88, 26, 16, L["a1"], BLUE_BG, BLUE, fs=7.4)
    box("a2", 2, 66, 26, 13, L["a2"], BLUE_BG, BLUE, fs=7.4)
    box("a3", 2, 24.5, 26, 13, L["a3"], BLUE_BG, BLUE, fs=7.4)
    box("b1", 36, 70, 20, 13, L["b1"], TEAL_BG, TEAL, fs=7.4)
    box("b2", 36, 24, 20, 14, L["b2"], TEAL_BG, TEAL, fs=7.3)
    box("c1", 62, 92, 21, 10, L["c1"], TEAL_BG, TEAL, fs=7.3)
    box("c2", 62, 62, 21, 22, L["c2"], TEAL_BG, TEAL, fs=7.3)
    box("c3", 62, 36, 21, 11, L["c3"], "white", TEAL, fs=7.4, mono=True)

    box("cur", 94, 96, 18, 9, L["cur"], "white", GREY_LN, fs=7.2)
    box("s1", 116, 96, 18, 9, L["s1"], "white", GREY_LN, fs=7.2)
    box("s2", 138, 96, 15, 9, L["s2"], "white", GREY_LN, fs=7.2)
    box("s3", 157, 96, 13, 9, L["s3"], "white", GREY_LN, fs=7.2)
    box("s4", 174, 96, 16, 9, L["s4"], "white", GREY_LN, fs=7.2)
    box("s5", 94, 80, 96, 11, L["s5"], "white", GREY_LN, fs=7.8)
    box("sout", 196, 80, 42, 25, L["sout"], RED_BG, RED, fs=8.2, tc="#7c2d24", lw=2.0)

    box("p1", 94, 52, 26, 13, L["p1"], EMER_BG, EMER, fs=7.5)
    box("p2", 94, 31, 26, 16, L["p2"], EMER_BG, EMER, fs=7.5, lw=2.4)
    box("p3", 126, 38, 26, 15, L["p3"], EMER_BG, EMER, fs=7.5, lw=2.4)
    box("k1", 158, 50, 28, 16, L["k1"], PUR_BG, PUR, fs=7.4, lw=2.2)
    box("k2", 158, 22, 28, 16, L["k2"], PUR_BG, PUR, fs=7.4)
    box("doe", 126, 20, 26, 9, L["doe"], "white", GREY_LN, fs=6.9)
    box("d3", 196, 60, 42, 11, L["d3"], EMER_BG, EMER, fs=7.7)
    box("d1", 196, 44, 42, 12, L["d1"], BLUE_BG, NAVY, fs=7.7, lw=2.2)
    box("d2", 196, 22, 42, 16, L["d2"], NAVY, NAVY, fs=8.4, tc="white", lw=2.4)

    # ---------------- LOOCV wall ----------------
    ax.plot([88, 88], [16, 110], ls=(0, (6, 3)), color=RED, lw=2.8, zorder=1)
    ax.text(86.2, 80, L["wall"], ha="center", va="center", fontsize=10, color=RED,
            fontweight="bold", zorder=6, rotation=90,
            bbox=dict(fc="white", ec=RED, lw=1.0, pad=2.0))
    ax.text(88, 113.4, L["wallsub"], ha="center", va="top", fontsize=6.6, color=RED,
            style="italic", linespacing=1.4, zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.94, pad=1.2))

    # ---------------- connectors ----------------
    r = Router()
    # acquisition -> raw stores (a1 + a2 join at a junction dot)
    r.add([(28, 96), (32, 96), (32, 76.5), (36, 76.5)], BLUE, skip={"a1", "b1"})
    r.add([(28, 72.5), (32, 72.5), (32, 76.5)], BLUE, arrow=False, skip={"a2", "b1"})
    r.dot(32, 76.5, BLUE)
    r.add([(28, 31), (36, 31)], BLUE, skip={"a3", "b2"})
    # raw -> preparation
    r.add([(56, 76.5), (59, 76.5), (59, 97), (62, 97)], TEAL, skip={"b1", "c1"})
    r.add([(72.5, 92), (72.5, 84)], TEAL, skip={"c1", "c2"})
    r.add([(72.5, 62), (72.5, 47)], TEAL, skip={"c2", "c3"})
    r.add([(56, 31), (59, 31), (59, 41.5), (62, 41.5)], TEAL, skip={"b2", "c3"})
    # across the LOOCV wall: three pins out of the feature table
    r.add([(83, 46), (91.5, 46), (91.5, 100.5), (94, 100.5)], GREY_LN, skip={"c3", "cur"})
    r.add([(83, 44), (89.5, 44), (89.5, 58.5), (94, 58.5)], EMER, lw=2.4, skip={"c3", "p1"})
    r.add([(83, 39), (94, 39)], EMER, lw=3.0, skip={"c3", "p2"})
    # sensor lane chain
    r.add([(112, 100.5), (116, 100.5)], GREY_LN, skip={"cur", "s1"})
    r.add([(134, 100.5), (138, 100.5)], GREY_LN, skip={"s1", "s2"})
    r.add([(153, 100.5), (157, 100.5)], GREY_LN, skip={"s2", "s3"})
    r.add([(170, 100.5), (174, 100.5)], GREY_LN, skip={"s3", "s4"})
    r.add([(182, 96), (182, 91)], GREY_LN, skip={"s4", "s5"})
    r.add([(190, 85.5), (196, 85.5)], GREY_LN, skip={"s5", "sout"})
    # physics golden path
    r.add([(107, 52), (107, 47)], EMER, lw=2.4, skip={"p1", "p2"})
    r.add([(120, 39), (123, 39), (123, 45.5), (126, 45.5)], EMER, lw=3.0, skip={"p2", "p3"})
    r.add([(152, 45.5), (155, 45.5), (155, 58), (158, 58)], EMER, lw=3.0, skip={"p3", "k1"})
    # conformal fan-out (junction dot) -> online monitor + RUL window
    r.add([(186, 58), (189, 58), (189, 65.5), (196, 65.5)], EMER, lw=2.2, skip={"k1", "d3"})
    r.add([(189, 58), (189, 50), (196, 50)], EMER, lw=3.0, skip={"k1", "d1"})
    r.dot(189, 58, EMER)
    # chipping events trunk (runs in the free corridor below everything)
    r.add([(46, 24), (46, 18), (172, 18), (172, 22)], PUR, skip={"b2", "k2"})
    r.add([(152, 24.5), (158, 24.5)], GREY_LN, lw=1.2, skip={"doe", "k2"})
    r.add([(186, 30), (196, 30)], PUR, lw=2.0, skip={"k2", "d2"})
    # decision column
    r.add([(228, 60), (228, 56)], EMER, lw=1.8, skip={"d3", "d1"})
    r.add([(216, 44), (216, 38)], NAVY, lw=2.8, skip={"d1", "d2"})
    # exploratory covariate (dashed) — crosses the two conformal branches -> hops
    r.add([(205, 80), (205, 75), (192, 75), (192, 34), (186, 34)], RED, lw=1.5, dashed=True,
          skip={"sout", "k2"})

    bad = r.validate(B)
    if bad:
        for name, p0, p1 in bad:
            print(f"  !! clearance violation: box {name} vs segment {p0}->{p1}")
    n_hops = r.draw(ax)

    # ---------------- edge labels (placed in free corridors) ----------------
    lbl(31.6, 84.5, L["e_sig"], ha="left")
    lbl(32, 28.4, L["e_vb"])
    lbl(59.4, 87, L["e_burst"], ha="left")
    lbl(74.6, 55, L["e_feats"], ha="left")
    lbl(59.6, 36.4, L["e_vb"], ha="left")
    lbl(94.5, 93.5, L["e_rows"], ha="left")
    lbl(96.5, 69, L["e_curves"], ha="left")
    lbl(96.5, 25.5, L["e_m"], ha="left")
    lbl(121.6, 42.6, L["e_ba"], ha="left")
    lbl(154, 41.4, L["e_curve"])
    lbl(191.4, 53.4, L["e_band"], ha="left")
    lbl(190, 30, L["e_risk"])
    lbl(105, 19.6, L["e_events"], ha="left")
    lbl(193, 88, L["e_null"])
    lbl(229.6, 58, L["e_alarm"], ha="left")
    ax.text(190.5, 76.5, L["e_kurt"], ha="right", fontsize=6.6, style="italic", color=RED,
            zorder=7, bbox=dict(fc="white", ec=RED, lw=0.5, alpha=0.95, pad=1.0))

    # ---------------- schematic legend ----------------
    lx, ly = 3, 109.5
    ax.add_line(Line2D([lx, lx + 3.4], [ly, ly], color=SLATE, lw=1.7, zorder=4))
    th = np.linspace(np.pi, 0, 40)
    ax.add_line(Line2D(lx + 4.6 + HOP_R * np.cos(th), ly + HOP_R * np.sin(th), color=SLATE,
                       lw=1.7, zorder=4))
    ax.add_line(Line2D([lx + 5.85, lx + 9.2], [ly, ly], color=SLATE, lw=1.7, zorder=4))
    ax.add_line(Line2D([lx + 4.6, lx + 4.6], [ly - 2.6, ly + 2.6], color=SLATE, lw=1.7, zorder=3))
    ax.text(lx + 10.4, ly, L["legend_hop"], fontsize=7.2, va="center", color=SLATE, style="italic")
    jx = lx + 46
    ax.add_line(Line2D([jx, jx + 9.2], [ly, ly], color=SLATE, lw=1.7, zorder=4))
    ax.add_line(Line2D([jx + 4.6, jx + 4.6], [ly, ly + 2.6], color=SLATE, lw=1.7, zorder=4))
    ax.plot([jx + 4.6], [ly], marker="o", ms=5.4, color=SLATE, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.text(jx + 10.4, ly, L["legend_dot"], fontsize=7.2, va="center", color=SLATE, style="italic")

    # ---------------- payoff ribbon ----------------
    ax.add_patch(Rectangle((1.5, 1.5), 237, 12, fc="white", ec=NAVY, lw=1.4, zorder=3))
    ax.text(4.5, 7.5, L["payoff_tag"], ha="left", va="center", fontsize=11, color=NAVY,
            fontweight="bold", zorder=4)
    x0 = 30
    for big, small in L["kpis"]:
        ax.text(x0, 9.6, big, ha="left", va="center", fontsize=14, color=EMER,
                fontweight="bold", zorder=4)
        ax.text(x0, 4.3, small, ha="left", va="center", fontsize=6.8, color=SLATE, zorder=4,
                linespacing=1.3)
        x0 += 42

    fig.savefig(os.path.join(FIG, f"dataflow_detailed_{lang}.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote dataflow_detailed_{lang}.png | clearance violations: {len(bad)} | hops: {n_hops}")


if __name__ == "__main__":
    make("EN")
    make("ES")
