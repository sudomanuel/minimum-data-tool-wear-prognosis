# -*- coding: utf-8 -*-
"""make_dataflow_figure.py — v2 BUSINESS-GRADE, highly detailed end-to-end data-flow map.

Structure (per the graphify pipeline skeleton, enriched with every real artifact/number):
  OBJECTIVE banner -> numbered phase rail -> swimlaned flow with per-edge data payloads and
  the LOOCV wall -> PAYOFF KPI ribbon.  The deployed route is the emphasized "golden path".
EN + ES editions, 300 dpi, full-bleed axes (matplotlib default margins were the v2.0 bug).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

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
   e_rows="172 × 296", e_curves="17 training\nVB curves", e_m="first m = 3–4\nreadings",
   e_ba="(b, a)", e_curve="VB(t)", e_band="curve + band", e_risk="h(VB)",
   e_events="18 chipping events", e_null="no transferable signal",
   e_kurt="radial spectral kurtosis →\nexploratory risk covariate (p = 0.010)",
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
   e_rows="172 × 296", e_curves="17 curvas VB de\nentrenamiento", e_m="primeras m = 3–4\nlecturas",
   e_ba="(b, a)", e_curve="VB(t)", e_band="curva + banda", e_risk="h(VB)",
   e_events="18 eventos de astillado", e_null="sin señal transferible",
   e_kurt="curtosis espectral radial →\ncovariable exploratoria (p = 0.010)",
 ),
}


def make(lang):
    L = T[lang]
    fig, ax = plt.subplots(figsize=(24, 12.8))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 240); ax.set_ylim(0, 128); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=8.2, tc=INK, lw=1.5, bold=True, mono=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", zorder=4, linespacing=1.32,
                family="monospace" if mono else None)
        return (x, y, w, h)

    def arrow(a, b, color=GREY_LN, label=None, lx=0, ly=0, lw=1.6, con="arc3,rad=0.0",
              fs=7.0, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=lw,
                                     color=color, zorder=2, connectionstyle=con,
                                     linestyle=(0, (5, 3)) if dashed else "solid"))
        if label:
            ax.text((a[0]+b[0])/2 + lx, (a[1]+b[1])/2 + ly, label, ha="center", va="center",
                    fontsize=fs, style="italic", color="#43555f", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.6))

    def R(b):  return (b[0]+b[2]+0.5, b[1]+b[3]/2)
    def Lm(b): return (b[0]-0.5, b[1]+b[3]/2)
    def Tm(b): return (b[0]+b[2]/2, b[1]+b[3]+0.5)
    def Bm(b): return (b[0]+b[2]/2, b[1]-0.5)

    # ======== OBJECTIVE banner ========
    ax.add_patch(Rectangle((1.5, 118.5), 237, 8.4, fc=NAVY, ec="none", zorder=3))
    ax.text(4.5, 122.7, L["objective_tag"], ha="left", va="center", fontsize=13,
            color="#9FB4E8", fontweight="bold", zorder=4)
    ax.text(29, 122.7, L["objective"], ha="left", va="center", fontsize=10.6, color="white",
            fontweight="bold", zorder=4, linespacing=1.5)

    # ======== phase rail ========
    spans = [(1.5, 55), (58.5, 81.5), (91.5, 152), (157, 187), (191.5, 238.5)]
    for (x0, x1), (num, name) in zip(spans, L["phases"]):
        ax.plot([x0, x1], [114.6, 114.6], color=NAVY, lw=1.1, zorder=3)
        ax.text(x0, 116.2, num, fontsize=10, color=GOLD, fontweight="bold", zorder=4)
        ax.text(x0 + 4.5, 116.2, name, fontsize=9, color=NAVY, fontweight="bold", zorder=4)

    # ======== lane tints + titles ========
    ax.add_patch(Rectangle((89, 74.5), 150.5, 34, fc=GREY_BG, ec="none", alpha=0.55, zorder=0))
    ax.add_patch(Rectangle((89, 15.5), 150.5, 55, fc=EMER_BG, ec="none", alpha=0.35, zorder=0))
    ax.text(238, 110.5, L["lane_s"], ha="right", fontsize=8.6, color=SLATE,
            fontweight="bold", zorder=4)
    ax.text(238, 16.8, L["lane_p"], ha="right", fontsize=8.6, color=EMER,
            fontweight="bold", zorder=4)

    # ======== left half ========
    a1 = box(1.5, 92, 27, 17, L["a1"], BLUE_BG, BLUE, fs=7.4)
    a2 = box(1.5, 66, 27, 13, L["a2"], BLUE_BG, BLUE, fs=7.4)
    a3 = box(1.5, 27, 27, 12, L["a3"], BLUE_BG, BLUE, fs=7.4)
    b1 = box(33.5, 70, 21, 13, L["b1"], TEAL_BG, TEAL, fs=7.4)
    b2 = box(33.5, 25, 21, 14, L["b2"], TEAL_BG, TEAL, fs=7.3)
    c1 = box(59.5, 94, 22, 10, L["c1"], TEAL_BG, TEAL, fs=7.3)
    c2 = box(59.5, 62, 22, 24, L["c2"], TEAL_BG, TEAL, fs=7.3)
    c3 = box(59.5, 38, 22, 11, L["c3"], "white", TEAL, fs=7.4, mono=True)

    # ======== the wall ========
    ax.plot([86.5, 86.5], [4, 108], ls=(0, (6, 3)), color=RED, lw=2.8, zorder=1)
    ax.text(84.6, 22, L["wall"], ha="center", va="center", fontsize=10, color=RED,
            fontweight="bold", zorder=6, rotation=90,
            bbox=dict(fc="white", ec=RED, lw=1.0, pad=2.0))
    ax.text(86.5, 113.4, L["wallsub"], ha="center", va="top", fontsize=6.6, color=RED,
            style="italic", linespacing=1.4, zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.94, pad=1.2))

    # ======== sensor lane ========
    cur = box(91.5, 98, 17, 7.5, L["cur"], "white", GREY_LN, fs=7.2)
    s1 = box(112, 98, 19, 7.5, L["s1"], "white", GREY_LN, fs=7.2)
    s2 = box(134.5, 98, 15.5, 7.5, L["s2"], "white", GREY_LN, fs=7.2)
    s3 = box(153.5, 98, 12.5, 7.5, L["s3"], "white", GREY_LN, fs=7.2)
    s4 = box(169.5, 98, 17.5, 7.5, L["s4"], "white", GREY_LN, fs=7.2)
    s5 = box(91.5, 80, 95.5, 12, L["s5"], "white", GREY_LN, fs=7.8)
    so = box(191.5, 80, 47, 25.5, L["sout"], RED_BG, RED, fs=8.2, tc="#7c2d24", lw=2.0)
    arrow(R(cur), Lm(s1)); arrow(R(s1), Lm(s2)); arrow(R(s2), Lm(s3)); arrow(R(s3), Lm(s4))
    arrow(Bm(s4), (178.25, 92.5))
    arrow(R(s5), Lm(so), GREY_LN, L["e_null"], lx=-5, ly=3.2, con="arc3,rad=-0.15", fs=6.8)

    # ======== physics lane (golden path) ========
    p1 = box(91.5, 52, 27, 13, L["p1"], EMER_BG, EMER, fs=7.5)
    p2 = box(91.5, 27, 27, 16, L["p2"], EMER_BG, EMER, fs=7.5, lw=2.4)
    p3 = box(124, 37, 27, 15, L["p3"], EMER_BG, EMER, fs=7.5, lw=2.4)
    k1 = box(157, 49, 28, 16, L["k1"], PUR_BG, PUR, fs=7.4, lw=2.2)
    k2 = box(157, 21, 28, 16, L["k2"], PUR_BG, PUR, fs=7.4)
    doe = box(124, 21, 27, 9, L["doe"], "white", GREY_LN, fs=6.9)
    d3 = box(191.5, 59, 47, 11, L["d3"], EMER_BG, EMER, fs=7.7)
    d1 = box(191.5, 42, 47, 12, L["d1"], BLUE_BG, NAVY, fs=7.7, lw=2.2)
    d2 = box(191.5, 20, 47, 17, L["d2"], NAVY, NAVY, fs=8.4, tc="white", lw=2.4)

    # ======== edges: left half ========
    arrow(R(a1), Lm(b1), BLUE)
    arrow(R(a2), Lm(b1), BLUE, L["e_sig"], lx=-0.6, ly=-2.2)
    arrow(R(a3), Lm(b2), BLUE, L["e_vb"], ly=2.0)
    arrow(R(b1), Lm(c1), TEAL, L["e_burst"], lx=0.8, ly=2.8)
    arrow(Bm(c1), Tm(c2), TEAL)
    arrow(Bm(c2), Tm(c3), TEAL, L["e_feats"], lx=9.2, ly=0.3)
    arrow(R(b2), (58.9, 41.5), TEAL, L["e_vb"], ly=-2.4)

    # ======== across the wall ========
    arrow(R(c3), Lm(cur), GREY_LN, L["e_rows"], lx=-4.6, ly=4.2, con="arc3,rad=0.42")
    arrow((R(c3)[0], 42.5), Lm(p1), EMER, L["e_curves"], lx=-1.4, ly=-3.6,
          con="arc3,rad=-0.12", lw=2.4)
    arrow((R(c3)[0], 40.0), Lm(p2), EMER, L["e_m"], lx=-3.4, ly=-3.8,
          con="arc3,rad=-0.2", lw=3.0)
    arrow((86.5, 8.5), (156.4, 26.0), PUR, L["e_events"], lx=-16, ly=3.2, con="arc3,rad=0.1")

    # ======== physics golden path ========
    arrow(Bm(p1), Tm(p2), EMER, lw=2.4)
    arrow(R(p2), Lm(p3), EMER, L["e_ba"], ly=2.2, lw=3.0)
    arrow(R(p3), Lm(k1), EMER, L["e_curve"], lx=0.4, ly=-3.0, con="arc3,rad=-0.1", lw=3.0)
    arrow(R(k1), Lm(d1), EMER, L["e_band"], lx=0.4, ly=-2.8, con="arc3,rad=0.08", lw=3.0,
          fs=6.8)
    arrow(Tm(k1), Lm(d3), EMER, None, con="arc3,rad=-0.18", lw=1.7)
    arrow(R(k2), Lm(d2), PUR, L["e_risk"], lx=-1.2, ly=-2.6, fs=6.8, lw=2.0)
    arrow(R(doe), Lm(k2), GREY_LN, None, con="arc3,rad=0.15", lw=1.2)
    arrow(Bm(d1), Tm(d2), NAVY, lw=2.8)
    arrow(Bm(d3), (216.0, 54.5), EMER, lw=1.6)

    # exploratory covariate (dashed) through the inter-column corridor
    arrow((191.0, 79.4), (185.6, 33.5), RED, None, con="arc3,rad=-0.12", dashed=True, lw=1.5)
    ax.text(188.8, 74.0, L["e_kurt"], ha="right", fontsize=6.6, style="italic", color=RED,
            zorder=6, bbox=dict(fc="white", ec=RED, lw=0.5, alpha=0.94, pad=1.0))

    # ======== PAYOFF ribbon ========
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

    fig.savefig(os.path.join(FIG, f"dataflow_detailed_{lang}.png"), dpi=300,
                facecolor="white")
    plt.close(fig)
    print("wrote", f"dataflow_detailed_{lang}.png")


if __name__ == "__main__":
    make("EN")
    make("ES")
