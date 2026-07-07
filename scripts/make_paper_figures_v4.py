"""
make_paper_figures_v4.py — P6 Step 5: single-column lock + cero anotaciones.

Reglas del usuario (PDF QA v4):
  1. NINGUNA figura rompe el esquema de 2 columnas -> todas single-column
     (las 4 ex-figure* se rediseñan: paneles apilados en vertical).
  2. NINGUN texto/flecha/conector dentro del area de datos -> las
     anotaciones se eliminan; el mensaje vive en el caption.

Genera *_v4.png (no sobrescribe v3). benchmark_branches_v3 se mantiene
(sin anotaciones, ya single-column).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

MET = ROOT / "outputs" / "metrics"
FIGS = (ROOT / "paper" / "overleaf_extracted" / "phm_pinn_paper_scaffold_overleaf"
        / "phm_pinn_paper_skeleton" / "figures")
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
    "figure.dpi": 200, "savefig.dpi": 200, "axes.grid": True,
    "grid.alpha": 0.25,
})
C = {"tonly": "#D7263D", "xt": "#2E86AB", "sensor": "#1F4E79",
     "pinn": "#1B7F5A", "bad": "#A0521E", "dummy": "#7A7A7A"}


def _save(fig, name):
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


# 1 ------------------------------------------------ timeaware (sin anotacion)
def fig_timeaware_v4():
    rows = [("Poly2(t)", 4.96, C["tonly"]), ("Linear(t)", 9.93, C["tonly"]),
            ("ElasticNet(x,t)", 17.46, C["xt"]),
            ("PINN$_{mono}$ (initial)", 18.57, C["pinn"]),
            ("ElasticNet(x)", 19.07, C["sensor"]),
            ("Dummy mean", 56.36, C["dummy"])]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.invert_yaxis()
    for i, (_n, v, _c) in enumerate(rows):
        ax.text(v + 0.8, i, f"{v:.2f}", va="center", fontsize=10.5)
    ax.set_xlabel("LOEO MAE (µm) — lower is better")
    ax.set_xlim(0, 66)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (C["tonly"], C["xt"], C["pinn"], C["sensor"], C["dummy"])]
    ax.legend(handles, ["t-only", "(x,t)", "physics-informed",
                        "sensor-only", "floor"],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3,
              frameon=False, fontsize=10)
    _save(fig, "timeaware_comparison_v4.png")


# 2 ------------------------------------------- PINN ablation (sin anotacion)
def fig_pinn_ablation_v4():
    r = pd.read_csv(MET / "pinn_comparison_results.csv")
    g3 = r[r.group == "G3_physics"][["model", "MAE"]]
    order = ["PINN_mono", "PINN_no_physics", "PINN_mono_smooth", "PINN_smooth",
             "PINN_rate", "PINN_full_boundary_initial", "PINN_full"]
    labels = ["mono (best)", "no physics", "mono+smooth", "smooth",
              "rate", "full+bound.", "full"]
    vals = [float(g3.loc[g3.model == m, "MAE"].iloc[0]) for m in order]
    colors = [C["pinn"] if m == "PINN_mono" else
              ("#9CC7A5" if v < 30 else C["bad"]) for m, v in zip(order, vals)]
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    y = np.arange(len(order))
    ax.barh(y, vals, color=colors, alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=10.5)
    l1 = ax.axvline(4.96, color=C["tonly"], ls="--", lw=1.8,
                    label="Poly2(t) = 4.96 (temporal stress-test)")
    l2 = ax.axvline(19.07, color=C["sensor"], ls=":", lw=1.8,
                    label="ElasticNet(x) = 19.07 (sensor-only ref.)")
    ax.set_xlim(0, 128)
    ax.set_xlabel("LOEO MAE (µm) — lower is better")
    ax.legend(handles=[l1, l2], loc="upper center",
              bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False, fontsize=10)
    _save(fig, "pinn_comparison_mae_v4.png")


# 3 -------------------------------------------- OOF curve (sin anotacion)
def fig_vb_curve_v4():
    p = pd.read_csv(MET / "pinn_fold_predictions.csv")
    bp = p[p.model == "PINN_mono"].sort_values("experiment_order")
    pp = p[p.model == "Poly2(t)"].sort_values("experiment_order")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(bp.experiment_order, bp.VB_true, "ko-", lw=1.6, ms=7,
            label="observed VB")
    ax.plot(bp.experiment_order, bp.VB_pred, "s--", color=C["pinn"], ms=7,
            label="PINN$_{mono}$ OOF (MAE 18.6)")
    ax.plot(pp.experiment_order, pp.VB_pred, "^:", color=C["tonly"], ms=6,
            alpha=0.85, label="Poly2(t) OOF (MAE 5.0)")
    ax.set_xlabel("experiment order $t$ (temporal proxy)")
    ax.set_ylabel("VB (µm)")
    ax.set_ylim(58, 300)
    ax.legend(loc="upper left", fontsize=9.5)
    _save(fig, "pinn_vb_curve_v4.png")


# 4 ------------------------------------------ structural (sin textos in-plot)
def fig_structural_v4():
    s = pd.read_csv(MET / "structural_uncertainty_summary.csv")
    order = ["PINN_mono", "Linear(t)", "Poly2(t)"]
    s = s.set_index("model").loc[order].reset_index()
    cmap = {"Poly2(t)": C["tonly"], "Linear(t)": C["dummy"],
            "PINN_mono": C["pinn"]}
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for i, r in s.iterrows():
        c = cmap[r.model]
        ax.plot([r.epistemic_t_failure_q025, r.epistemic_t_failure_q975],
                [i, i], color=c, lw=9, alpha=0.35, solid_capstyle="round")
        ax.plot(r.t_failure_point, i, "X", color=c, ms=14, zorder=5)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(["PINN$_{mono}$", "Linear(t)", "Poly2(t)"], fontsize=11)
    ax.axvline(10, color="k", ls=":", lw=1.3)
    ax.set_xlabel("$t_{fail}$ (threshold crossing, order steps)")
    ax.set_xlim(9.8, 12.75)
    ax.set_ylim(-0.5, 2.5)
    _save(fig, "uncertainty_structural_v4.png")


# 5 --------------------------------- RUL single-column (sin texto in-plot)
def fig_rul_curves_v4():
    from phm.config import (PROCESSED_DATASET, EXP_ORDER_COL, TARGET_COLUMN,
                            RANDOM_SEED)
    from phm.pinn import (PINNRegressor, PINN_VARIANTS, resolve_driver_col,
                          select_minimal_physical_features)
    cfg = yaml.safe_load((ROOT / "config" / "physics.yaml").read_text(encoding="utf-8"))
    vb_fail = float(cfg["vb_failure_um"])
    df = pd.read_csv(PROCESSED_DATASET).sort_values(EXP_ORDER_COL)
    t_obs = df[EXP_ORDER_COL].values.astype(float)
    vb = df[TARGET_COLUMN].values.astype(float)
    grid = np.arange(1, 13.6, 0.01)
    p2, p1 = np.polyfit(t_obs, vb, 2), np.polyfit(t_obs, vb, 1)
    x_min = select_minimal_physical_features(df.columns)
    driver = resolve_driver_col(df.columns)
    pinn = PINNRegressor(hidden=(32, 32), epochs=3000, random_state=RANDOM_SEED,
                         **PINN_VARIANTS["PINN_mono"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pinn.fit(df[x_min].values, t_obs, vb, e_rot=df[driver].values)
    Xg = np.empty((len(grid), len(x_min)))
    for j in range(len(x_min)):
        Xg[:, j] = np.interp(grid, t_obs, df[x_min].values[:, j])
    curves = {"Poly2(t)": (np.polyval(p2, grid), C["tonly"], "-"),
              "PINN$_{mono}$": (pinn.predict(Xg, grid), C["pinn"], "-"),
              "Linear(t)": (np.polyval(p1, grid), C["dummy"], ":")}
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(t_obs, vb, "ko", ms=7, zorder=5, label="observed VB (max 280 µm)")
    for name, (vals, color, ls) in curves.items():
        idx = np.argmax(vals >= vb_fail)
        tf = grid[idx] if (vals >= vb_fail).any() else np.nan
        ax.plot(grid, vals, ls, color=color, lw=2.0,
                label=f"{name}: $t_{{fail}}$ = {tf:.2f}")
        ax.plot(tf, vb_fail, "X", color=color, ms=12, zorder=6)
    ax.axhline(vb_fail, color=C["bad"], ls="--", lw=1.6,
               label="VB$_{fail}$ = 300 µm (provisional)")
    ax.axvspan(10, 13.6, alpha=0.08, color="gray")
    ax.set_xlim(0.7, 13.5)
    ax.set_ylim(60, 365)
    ax.set_xlabel("experiment order $t$ (temporal proxy)")
    ax.set_ylabel("VB (µm)")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "rul_curves_v4.png")


# 6 ----------------------------- VB bands: 2 paneles APILADOS (single col)
def fig_vb_bands_v4():
    b = pd.read_csv(MET / "uncertainty_vb_bands.csv")
    from phm.config import PROCESSED_DATASET, EXP_ORDER_COL, TARGET_COLUMN
    df = pd.read_csv(PROCESSED_DATASET).sort_values(EXP_ORDER_COL)
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 6.6), sharex=True)
    for ax, m, color, sub in (
            (axes[0], "Poly2(t)", C["tonly"], "residual bootstrap (B=500)"),
            (axes[1], "PINN_mono", C["pinn"], "deep ensemble (K=10)")):
        d = b[b.model == m]
        ax.fill_between(d.t, d.VB_q025, d.VB_q975, alpha=0.18, color=color,
                        label="95% band (descriptive)")
        ax.fill_between(d.t, d.VB_q16, d.VB_q84, alpha=0.32, color=color,
                        label="68% band")
        ax.plot(d.t, d.VB_mean, color=color, lw=2, label="ensemble mean")
        ax.plot(df[EXP_ORDER_COL], df[TARGET_COLUMN], "ko", ms=5,
                label="observed VB")
        ax.axhline(300, color=C["bad"], ls="--", lw=1.4)
        ax.axvspan(10, 14, alpha=0.07, color="gray")
        ax.set_xlim(0.5, 14)
        ax.set_ylim(60, 380)
        ax.set_title(f"{'Poly2(t)' if m == 'Poly2(t)' else 'PINN$_{mono}$'} — {sub}",
                     fontsize=11.5)
        ax.set_ylabel("VB (µm)")
        ax.legend(fontsize=8.5, loc="upper left")
    axes[1].set_xlabel("experiment order $t$")
    _save(fig, "uncertainty_vb_bands_v4.png")


# 7 ------------------------- physical metrics: 3 paneles APILADOS (single col)
def fig_physical_metrics_v4():
    r = pd.read_csv(MET / "pinn_comparison_results.csv")
    models = ["Poly2(t)", "PINN_no_physics", "PINN_mono", "PINN_smooth",
              "PINN_rate", "PINN_full"]
    short = ["Poly2(t)", "no-phys", "mono", "smooth", "rate", "full"]
    sub = r.set_index("model").loc[models]
    cols = [("monotonicity_violations", "Monotonicity violations (OOF)", False),
            ("negative_rate_fraction", "Negative-rate fraction (OOF)", False),
            ("smoothness_penalty", "Smoothness penalty (µm², log)", True)]
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 7.6))
    for ax, (c, title, logy) in zip(axes, cols):
        v = sub[c].values.astype(float)
        colors = [C["tonly"]] + [C["pinn"]] * (len(models) - 1)
        ax.bar(range(len(models)), v, color=colors, alpha=0.92)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(short, fontsize=10.5)
        ax.set_title(title, fontsize=11.5)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(top=ax.get_ylim()[1] * 3)
        else:
            ax.set_ylim(top=max(v) * 1.22)
        for i, val in enumerate(v):
            ax.text(i, val, f"{val:.2g}", ha="center", va="bottom", fontsize=9.5)
    fig.tight_layout()
    _save(fig, "pinn_physical_metrics_v4.png")


# 8 -------------------------------- roadmap single-column (texto compactado)
def fig_roadmap_v4():
    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    ax.axis("off")
    boxes = [
        ("T01 single-tool stage (this paper)\nhonest baselines · frozen PINN scorecard\n· failure modes", "#EAF0F6", C["sensor"]),
        ("Multi-tool ingestion — NOT optional\nmanifest: tool id · cumulative cutting\ntime/length · failure events · metadata", "#FDF3E7", C["bad"]),
        ("Leave-One-Tool-Out validation\ntool-level splits · early-life prognosis\n· ≥ 1 tool reaching failure threshold", "#EAF0F6", C["sensor"]),
        ("Improved PINN configurations\npositive rate law g = softplus(a+bE)\n· monotone-by-construction VB · physical $t$", "#E8F4EE", C["pinn"]),
        ("CLAIM GATE: PINN claimed superior\nonly if it beats temporal AND classical\nbaselines under LOTO", "#FBEAEA", C["tonly"]),
    ]
    n = len(boxes)
    bh = 0.135
    ys = np.linspace(0.93, 0.07, n)
    for (text, fc, ec), yc in zip(boxes, ys):
        ax.add_patch(FancyBboxPatch((0.04, yc - bh / 2), 0.92, bh,
                     boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=1.8,
                     transform=ax.transAxes))
        ax.text(0.5, yc, text, ha="center", va="center", fontsize=10.8,
                transform=ax.transAxes)
    for y0, y1 in zip(ys[:-1] - bh / 2 - 0.004, ys[1:] + bh / 2 + 0.004):
        ax.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1),
                     transform=ax.transAxes, arrowstyle="-|>",
                     mutation_scale=20, lw=1.8, color="#333333"))
    _save(fig, "multitool_roadmap_v4.png")


if __name__ == "__main__":
    fig_timeaware_v4()
    fig_pinn_ablation_v4()
    fig_vb_curve_v4()
    fig_structural_v4()
    fig_vb_bands_v4()
    fig_physical_metrics_v4()
    fig_roadmap_v4()
    fig_rul_curves_v4()   # ultimo: reentrena PINN (~40 s)
    print("[done] 8 figures v4")
