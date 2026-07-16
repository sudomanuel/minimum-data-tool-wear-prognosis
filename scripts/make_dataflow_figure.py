# -*- coding: utf-8 -*-
"""make_dataflow_figure.py — HIGHLY DETAILED data-flow diagram of the whole pipeline.

Source of truth: the graphify pipeline-flow skeleton (20 nodes / 23 edges, graphify-out/
pipeline_flow.html of the archive project), enriched with the real data artifacts: file names,
row/column counts, per-edge data payloads, and the LOOCV boundary semantics. Two language
editions (EN/ES) at 300 dpi for the final-report deck (one dedicated slide each).

Lanes:  acquisition -> raw data -> preparation -> [LOOCV WALL] -> sensor lane (top, grey)
                                                              -> physics lane (bottom, green)
        -> calibration -> decision.  Every arrow is labelled with WHAT data moves across it.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "outputs", "figures")

NAVY, INK = "#1E2761", "#1a2a36"
GREEN, GF = "#1e7d4f", "#e9f5ef"
GREY, NF = "#5a6b7a", "#f1f4f6"
BLUE, BF = "#1F5FA8", "#eaf1fa"
TEAL, TF = "#0e7490", "#e6f4f7"
RED, RF = "#b03a2e", "#fdecea"
PUR, PF = "#6b21a8", "#f3e8ff"
EDGE = "#44525e"

T = {
 "EN": dict(
    a1="Hanwha XD26II-V\nSwiss-type turning centre\n(6-contact machining cycle)",
    a2="Axial (A) + radial (R)\naccelerometers\n[AE recorded, set aside]",
    a3="Keyence VHX-7000\nmicroscope — VB at\ncycle stops (ex situ)",
    b1="Raw signals per contact\nTXT, one energy burst per\ncontact · 6 per cycle",
    b2="VB targets\n172 inspections · 18 tools\n+ 18 wear-at-chipping levels",
    c1="Signal QA +\nper-burst segmentation",
    c2="Feature extraction\n294 features per cut\n(time · freq · energy · wavelet,\nper channel)",
    c3="Ingestion\nfeatures_experiment.csv\n172 rows × 296 numeric cols",
    c4="Curation 294 → 181\n(degenerate / redundant\nremoved, sensor branch)",
    wall="L O O C V   B O U N D A R Y",
    wallsub="18 folds · one full tool held out · only its first m readings visible ·\nfuture sealed for scoring · scaling, selection, tuning, calibration fitted INSIDE folds only",
    s1="Consensus feature\nselection (in-fold)",
    s2="Fold-safe\naugmentation",
    s3="Nested hyper-\nparameter tuning",
    s4="A/R fusion\n(6 strategies)",
    s5="Regressors: ridge · LASSO · elastic net · random forest ·\nXGBoost · MLP · PLS  (+ physics-equation hybrids, eq. 11)",
    sout="Cross-tool VB reading\nR² < 0 — documented null",
    p1="Fleet exponent p*\npooled fit on 17 training\ntools (eq. 4) · p* ≈ 0.20",
    p2="Few-shot personalisation\nTheil–Sen / Siegel + WLS γ = 3\n+ gross-error guard → (b, a)",
    p3="Personalised wear curve\nVB(t) = b + a·t^p*\nuses VB targets ONLY",
    k1="Out-of-fold residuals →\nMondrian horizon bins (≤1 · 2–3 · ≥4)\n→ 90% band (±19 µm near)",
    k2="Chipping hazard\n172 cycles + 18 real events\nlogit h(VB) → VB_safe ≈ 167 µm",
    doe="Unreplicated-DOE inference\n(half-normal · Lenth · ANOVA)",
    d1="Health index + multi-threshold\nRUL window (94% of 16 events)",
    d2="SAFE-STOP DECISION\nupper band edge\ncrosses VB_safe",
    d3="Online Kalman monitor\nτ-clock, event-triggered\nnext cut ≈ 4 µm",
    lane_s="DATA-DRIVEN SENSOR BRANCH", lane_p="PHYSICS-INTEGRATED FEW-SHOT BRANCH",
    e_sig="vibration\nsignals", e_vb="VB values", e_bursts="6 bursts/cycle",
    e_feats="294 features/cut", e_rows="172 × 296", e_curves="training-tool\nVB curves",
    e_m="first m = 3–4\nVB readings", e_par="(b, a)", e_curve="VB(t)",
    e_resid="out-of-fold\nresiduals", e_events="18 chipping\nevents", e_band="curve + 90% band",
    e_risk="one-cycle risk h(VB)", e_alarm="anomaly\nalarm", e_kurt="radial spectral kurtosis →\nexploratory risk covariate (p = 0.010)",
    e_ctx="condition acts on levels\n(directional context)", e_null="no transferable signal",
 ),
 "ES": dict(
    a1="Centro de torneado suizo\nHanwha XD26II-V\n(ciclo de 6 contactos)",
    a2="Acelerómetros axial (A)\n+ radial (R)\n[EA registrada, apartada]",
    a3="Microscopio Keyence\nVHX-7000 — VB en paradas\nde ciclo (ex situ)",
    b1="Señales crudas por contacto\nTXT, una ráfaga de energía por\ncontacto · 6 por ciclo",
    b2="Targets de VB\n172 inspecciones · 18 útiles\n+ 18 niveles de astillado",
    c1="QA de señal +\nsegmentación por ráfaga",
    c2="Extracción de características\n294 por corte\n(tiempo · frec. · energía · wavelet,\npor canal)",
    c3="Ingesta\nfeatures_experiment.csv\n172 filas × 296 cols numéricas",
    c4="Curación 294 → 181\n(degeneradas / redundantes\nfuera, rama de sensores)",
    wall="F R O N T E R A   L O O C V",
    wallsub="18 pliegues · un útil completo fuera · solo sus primeras m lecturas visibles ·\nfuturo sellado para calificar · escalado, selección, tuning y calibración SOLO dentro del pliegue",
    s1="Selección de características\npor consenso (en-pliegue)",
    s2="Augmentación\nfold-safe",
    s3="Tuning anidado de\nhiperparámetros",
    s4="Fusión A/R\n(6 estrategias)",
    s5="Regresores: ridge · LASSO · elastic net · bosque aleatorio ·\nXGBoost · MLP · PLS  (+ híbridos con ecuación física, ec. 11)",
    sout="Lectura de VB entre útiles\nR² < 0 — nulo documentado",
    p1="Exponente de flota p*\najuste agrupado en 17 útiles\nde entrenamiento (ec. 4) · p* ≈ 0.20",
    p2="Personalización few-shot\nTheil–Sen / Siegel + WLS γ = 3\n+ guarda anti-errores → (b, a)",
    p3="Curva de desgaste personalizada\nVB(t) = b + a·t^p*\nusa SOLO los targets de VB",
    k1="Residuos fuera-de-pliegue →\nceldas Mondrian por horizonte (≤1 · 2–3 · ≥4)\n→ banda 90% (±19 µm cerca)",
    k2="Hazard de astillado\n172 ciclos + 18 eventos reales\nlogit h(VB) → VB_safe ≈ 167 µm",
    doe="Inferencia DOE no replicado\n(semi-normal · Lenth · ANOVA)",
    d1="Índice de salud + ventana RUL\nmulti-umbral (94% de 16 eventos)",
    d2="DECISIÓN DE PARADA SEGURA\nel borde superior de la banda\ncruza VB_safe",
    d3="Monitor Kalman en línea\nreloj τ, por eventos\npróximo corte ≈ 4 µm",
    lane_s="RAMA DE SENSORES BASADA EN DATOS", lane_p="RAMA FÍSICA DE POCAS MUESTRAS",
    e_sig="señales de\nvibración", e_vb="valores VB", e_bursts="6 ráfagas/ciclo",
    e_feats="294 caract./corte", e_rows="172 × 296", e_curves="curvas VB de\nentrenamiento",
    e_m="primeras m = 3–4\nlecturas VB", e_par="(b, a)", e_curve="VB(t)",
    e_resid="residuos fuera-\nde-pliegue", e_events="18 eventos de\nastillado", e_band="curva + banda 90%",
    e_risk="riesgo por ciclo h(VB)", e_alarm="alarma de\nanomalía", e_kurt="curtosis espectral radial →\ncovariable exploratoria de riesgo (p = 0.010)",
    e_ctx="la condición actúa sobre niveles\n(contexto direccional)", e_null="sin señal transferible",
 ),
}


def make(lang):
    L = T[lang]
    fig, ax = plt.subplots(figsize=(20.6, 10.6))
    ax.set_xlim(0, 206); ax.set_ylim(0, 106); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=9.0, tc=INK, lw=1.6, bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.55",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", zorder=4, linespacing=1.25)
        return (x, y, w, h)

    def arrow(a, b, color=EDGE, label=None, lx=0, ly=0, lw=1.7, con="arc3,rad=0.0", fs=7.6,
              dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=lw,
                                     color=color, zorder=2, connectionstyle=con,
                                     linestyle=(0, (5, 3)) if dashed else "solid"))
        if label:
            mx, my = (a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly
            ax.text(mx, my, label, ha="center", va="center", fontsize=fs, style="italic",
                    color="#38505e", zorder=5,
                    bbox=dict(fc="white", ec="none", alpha=0.88, pad=0.7))

    def R(b):  return (b[0] + b[2] + 0.6, b[1] + b[3] / 2)
    def Lm(b): return (b[0] - 0.6, b[1] + b[3] / 2)
    def Tm(b): return (b[0] + b[2] / 2, b[1] + b[3] + 0.6)
    def Bm(b): return (b[0] + b[2] / 2, b[1] - 0.6)

    # ---------------- left half: acquisition -> raw -> preparation ----------------
    a1 = box(1.5, 78, 23, 13, L["a1"], BF, BLUE, fs=8.8)
    a2 = box(1.5, 56, 23, 13, L["a2"], BF, BLUE, fs=8.8)
    a3 = box(1.5, 14, 23, 13, L["a3"], BF, BLUE, fs=8.8)
    b1 = box(29.5, 64, 24, 14, L["b1"], TF, TEAL, fs=8.6)
    b2 = box(29.5, 14, 24, 13, L["b2"], TF, TEAL, fs=8.6)
    c1 = box(58, 79, 21, 11, L["c1"], TF, TEAL, fs=8.8)
    c2 = box(58, 57, 21, 15.5, L["c2"], TF, TEAL, fs=8.6)
    c3 = box(58, 33, 21, 14, L["c3"], TF, TEAL, fs=8.6)

    # ---------------- LOOCV wall ----------------
    ax.plot([84.5, 84.5], [2, 94], ls=(0, (6, 3)), color=RED, lw=2.6, zorder=1)
    ax.text(84.5, 103.6, L["wall"], ha="center", fontsize=12.5, color=RED, fontweight="bold")
    ax.text(84.5, 101.2, L["wallsub"], ha="center", va="top", fontsize=7.8, color=RED,
            style="italic", linespacing=1.3,
            bbox=dict(fc="white", ec=RED, lw=0.8, alpha=0.95, pad=2.2))

    # ---------------- sensor lane (top): serpentine ----------------
    ax.text(133, 94.3, L["lane_s"], ha="center", fontsize=10.5, color=GREY, fontweight="bold")
    cur = box(89, 81, 15.5, 10, L["c4"], NF, GREY, fs=7.6)
    s1 = box(107.5, 81, 16.5, 10, L["s1"], NF, GREY, fs=7.9)
    s2 = box(127, 81, 14.5, 10, L["s2"], NF, GREY, fs=8.2)
    s3 = box(144.5, 81, 15.5, 10, L["s3"], NF, GREY, fs=8.2)
    s4 = box(163, 81, 14.5, 10, L["s4"], NF, GREY, fs=8.2)
    s5 = box(89, 64.5, 88.5, 10, L["s5"], NF, GREY, fs=8.8)
    so = box(181, 81, 23, 10, L["sout"], RF, RED, fs=8.4, tc="#7c2d24")

    arrow(R(cur), Lm(s1), GREY); arrow(R(s1), Lm(s2), GREY)
    arrow(R(s2), Lm(s3), GREY); arrow(R(s3), Lm(s4), GREY)
    arrow(Bm(s4), (170.2, 75.1), GREY)
    arrow(R(s5), Bm(so), GREY, L["e_null"], lx=4.5, ly=-1.6, con="arc3,rad=-0.25", fs=7.4)

    # ---------------- physics lane (bottom) ----------------
    ax.text(113, 45.4, L["lane_p"], ha="center", fontsize=10.5, color=GREEN, fontweight="bold")
    p1 = box(89, 31, 24, 12.5, L["p1"], GF, GREEN, fs=8.4)
    p2 = box(89, 12, 24, 13.5, L["p2"], GF, GREEN, fs=8.4)
    p3 = box(119.5, 20, 24, 13.5, L["p3"], GF, GREEN, fs=8.6)
    arrow(Bm(p1), Tm(p2), GREEN)
    arrow(R(p2), Lm(p3), GREEN, L["e_par"], ly=2.0)

    # ---------------- calibration ----------------
    k1 = box(149.5, 28, 27, 13, L["k1"], PF, PUR, fs=8.0)
    k2 = box(149.5, 2, 27, 13, L["k2"], PF, PUR, fs=8.5)
    doe = box(119.5, 2, 24, 9.5, L["doe"], "white", GREY, fs=7.9)
    d3 = box(149.5, 47, 27, 11, L["d3"], GF, GREEN, fs=8.5)

    # ---------------- decision column ----------------
    d1 = box(181, 44, 23, 14.5, L["d1"], BF, NAVY, fs=7.9)
    d2 = box(181, 22.5, 23, 15, L["d2"], NAVY, NAVY, fs=8.7, tc="white")

    # ---------------- edges: left half ----------------
    arrow(R(a1), Lm(b1), BLUE)
    arrow(R(a2), Lm(b1), BLUE, L["e_sig"], lx=-1.0, ly=-2.4)
    arrow(R(a3), Lm(b2), BLUE, L["e_vb"], ly=2.0)
    arrow(R(b1), Lm(c1), TEAL, L["e_bursts"], lx=1.2, ly=2.6)
    arrow(Bm(c1), Tm(c2), TEAL)
    arrow(Bm(c2), Tm(c3), TEAL, L["e_feats"], lx=8.8, ly=0.4)
    arrow(R(b2), (57.4, 36.5), TEAL, L["e_vb"], ly=-2.4)

    # ---------------- edges across the wall ----------------
    arrow(R(c3), Lm(cur), GREY, None, con="arc3,rad=0.35")
    ax.text(86.8, 76.0, L["e_rows"], ha="center", fontsize=7.0, style="italic",
            color="#38505e", zorder=5, bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.6))
    arrow((R(c3)[0], R(c3)[1] - 1.5), Lm(p1), GREEN, L["e_curves"], lx=-0.6, ly=-3.2,
          con="arc3,rad=-0.10")
    arrow((R(c3)[0], R(c3)[1] - 3.0), Lm(p2), GREEN, L["e_m"], lx=-3.4, ly=-3.4,
          con="arc3,rad=-0.22")
    arrow((84.5, 6.0), (148.9, 6.5), PUR, L["e_events"], lx=-24, ly=2.6, fs=7.4)

    # ---------------- edges: physics -> calibration -> decision ----------------
    arrow(R(p3), Lm(k1), PUR, L["e_curve"], lx=0.4, ly=-3.2, fs=7.2, con="arc3,rad=-0.12")
    arrow(Tm(k1), Bm(d3), GREEN, L["e_band"], lx=5.2, ly=0.4, fs=7.3)
    arrow(R(k1), Lm(d1), NAVY, None, con="arc3,rad=0.22")
    arrow(R(k2), Lm(d2), NAVY, L["e_risk"], lx=-1.5, ly=-2.8, fs=7.4, con="arc3,rad=-0.1")
    arrow(R(doe), Lm(k2), GREY, L["e_ctx"], lx=0.5, ly=4.6, fs=7.0, con="arc3,rad=0.25")
    arrow(Bm(d1), Tm(d2), NAVY)
    arrow(Bm(d3), Tm(d2), GREEN, L["e_alarm"], lx=8.5, ly=0.0, fs=7.2, con="arc3,rad=-0.12")

    # exploratory covariate: sensor outcome -> hazard, dashed down the right margin
    arrow((199.5, 80.4), (176.5, 6.5), RED, None, con="arc3,rad=-0.32", dashed=True, lw=1.5)
    ax.text(197.5, 65.5, L["e_kurt"], ha="center", fontsize=7.3, style="italic", color=RED,
            bbox=dict(fc="white", ec=RED, lw=0.6, alpha=0.93, pad=1.2), zorder=5)

    fig.savefig(os.path.join(FIG, f"dataflow_detailed_{lang}.png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", f"dataflow_detailed_{lang}.png")


if __name__ == "__main__":
    make("EN")
    make("ES")
