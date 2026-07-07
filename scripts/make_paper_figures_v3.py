"""
make_paper_figures_v3.py — P6 Step 3: Visual QA fixes (layout lock).

Regenera las 6 figuras con defectos de layout detectados en la auditoria
visual, como *_v3.png (sin sobrescribir v2):

  timeaware_comparison_v3    legend fuera del plot (pisaba la barra Dummy)
  pinn_comparison_mae_v3     legend reubicada (pisaba la etiqueta 110.8)
  pinn_vb_curve_v3           anotacion movida a zona vacia (pisaba curvas)
  rul_curves_v3              etiqueta del umbral bajo la linea, sin colisiones
  benchmark_branches_v3      sizing single-column (>=7pt efectivos), titulo corto
  uncertainty_structural_v3  sizing single-column

Reglas: single-column -> figsize ~5.5 in de ancho con fuente 12
(7.5 pt efectivos a \\columnwidth); double-column (rul) -> 7.2 in.
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


# ------------------------------------------------- timeaware (legend below)
def fig_timeaware_v3():
    rows = [("Poly2(t)", 4.96, C["tonly"]), ("Linear(t)", 9.93, C["tonly"]),
            ("ElasticNet(x,t)", 17.46, C["xt"]),
            ("PINN$_{mono}$ (initial)", 18.57, C["pinn"]),
            ("ElasticNet(x)", 19.07, C["sensor"]),
            ("Dummy mean", 56.36, C["dummy"])]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.invert_yaxis()
    for i, (_n, v, _c) in enumerate(rows):
        ax.text(v + 0.8, i, f"{v:.2f}", va="center", fontsize=10.5)
    ax.set_xlabel("LOEO MAE (µm) — lower is better")
    ax.set_xlim(0, 66)
    ax.annotate("a quadratic in $t$ alone beats\nevery sensor-based model",
                xy=(3.6, 0.42), xytext=(24, 1.35), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color="black", lw=1.1,
                                connectionstyle="arc3,rad=0.15"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (C["tonly"], C["xt"], C["pinn"], C["sensor"], C["dummy"])]
    ax.legend(handles, ["t-only", "(x,t)", "physics-informed",
                        "sensor-only", "floor"],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              frameon=False, fontsize=10)
    _save(fig, "timeaware_comparison_v3.png")


# ------------------------------------------------- PINN ablation (no overlap)
def fig_pinn_ablation_v3():
    r = pd.read_csv(MET / "pinn_comparison_results.csv")
    g3 = r[r.group == "G3_physics"][["model", "MAE"]]
    order = ["PINN_mono", "PINN_no_physics", "PINN_mono_smooth", "PINN_smooth",
             "PINN_rate", "PINN_full_boundary_initial", "PINN_full"]
    labels = ["mono (best)", "no physics", "mono+smooth", "smooth",
              "rate", "full+bound.", "full"]
    vals = [float(g3.loc[g3.model == m, "MAE"].iloc[0]) for m in order]
    colors = [C["pinn"] if m == "PINN_mono" else
              ("#9CC7A5" if v < 30 else C["bad"]) for m, v in zip(order, vals)]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
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
    ax.annotate("rate-law misspecification\n(negative target rates)",
                xy=(76.7, 4.05), xytext=(48, 2.45), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C["bad"], lw=1.1))
    ax.set_xlim(0, 128)
    ax.set_xlabel("LOEO MAE (µm) — lower is better")
    ax.set_title("Initial PINN ablations: evolution baseline, not final verdict",
                 fontsize=11.5)
    ax.legend(handles=[l1, l2], loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=10)
    _save(fig, "pinn_comparison_mae_v3.png")


# ------------------------------------------------- OOF curve (annotation moved)
def fig_vb_curve_v3():
    p = pd.read_csv(MET / "pinn_fold_predictions.csv")
    bp = p[p.model == "PINN_mono"].sort_values("experiment_order")
    pp = p[p.model == "Poly2(t)"].sort_values("experiment_order")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(bp.experiment_order, bp.VB_true, "ko-", lw=1.6, ms=7,
            label="observed VB")
    ax.plot(bp.experiment_order, bp.VB_pred, "s--", color=C["pinn"], ms=7,
            label="PINN$_{mono}$ OOF (MAE 18.6)")
    ax.plot(pp.experiment_order, pp.VB_pred, "^:", color=C["tonly"], ms=6,
            alpha=0.85, label="Poly2(t) OOF (MAE 5.0)")
    # anotacion en la esquina inferior derecha (zona sin curvas)
    ax.annotate("end-of-life flattening:\nPINN underestimates final wear",
                xy=(9.95, 211), xytext=(5.6, 92), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=C["pinn"], lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_xlabel("experiment order $t$ (temporal proxy)")
    ax.set_ylabel("VB (µm)")
    ax.set_ylim(58, 300)
    ax.legend(loc="upper left", fontsize=9.5)
    _save(fig, "pinn_vb_curve_v3.png")


# ------------------------------------------------- RUL curves (double column)
def fig_rul_curves_v3():
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
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(t_obs, vb, "ko", ms=8, zorder=5, label="observed VB (max 280 µm)")
    for name, (vals, color, ls) in curves.items():
        idx = np.argmax(vals >= vb_fail)
        tf = grid[idx] if (vals >= vb_fail).any() else np.nan
        ax.plot(grid, vals, ls, color=color, lw=2.2,
                label=f"{name}: $t_{{fail}}$ = {tf:.2f}")
        ax.plot(tf, vb_fail, "X", color=color, ms=14, zorder=6)
    # umbral como entrada de leyenda (cero riesgo de colision con curvas)
    ax.axhline(vb_fail, color=C["bad"], ls="--", lw=1.8,
               label="VB$_{fail}$ = 300 µm (provisional)")
    ax.axvspan(10, 13.6, alpha=0.08, color="gray")
    ax.text(10.25, 92, "extrapolation\n(no data; conceptual RUL)",
            fontsize=10.5, color="gray")
    ax.set_xlim(0.7, 13.5)
    ax.set_ylim(60, 362)
    ax.set_xlabel("experiment order $t$ (temporal proxy)")
    ax.set_ylabel("VB (µm)")
    ax.legend(loc="upper left", fontsize=10)
    _save(fig, "rul_curves_v3.png")


# ------------------------------------------------- benchmark (single column)
def fig_benchmark_v3():
    m = pd.read_csv(MET / "layered_pipeline" / "09_all_metrics.csv")
    m = m[m.validation_type == "loeo"].dropna(subset=["MAE"])
    views = ["SOLO_A", "SOLO_R", "FUSION"]
    conds = [("defaults (ST, real)", lambda d: d[(d.tuning_method == "none") & (d.data_branch == "N")]),
             ("nested tuning (CT)", lambda d: d[d.tuning_method.isin(["random", "grid"]) & (d.data_branch == "N")]),
             ("augmented (best)", lambda d: d[(d.data_branch == "A")])]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    w = 0.26
    xs = np.arange(len(views))
    colors = [C["sensor"], C["bad"], "#5BA3D0"]
    for k, (lbl, sel) in enumerate(conds):
        vals = []
        for v in views:
            d = sel(m[m.feature_subset == v])
            vals.append(float(d.MAE.min()) if len(d) else np.nan)
        ax.bar(xs + (k - 1) * w, vals, w, label=lbl, color=colors[k], alpha=0.92)
        for x, val in zip(xs + (k - 1) * w, vals):
            ax.text(x, val + 0.6, f"{val:.1f}", ha="center", fontsize=9.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["SOLO_A\n(axial)", "SOLO_R\n(rotational)", "FUSION"],
                       fontsize=11)
    ax.set_ylabel("best LOEO MAE (µm)")
    ax.set_title("Sensor-only benchmark by view and condition", fontsize=11.5)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.set_ylim(0, 40)
    _save(fig, "benchmark_branches_v3.png")


# ------------------------------------------------- structural (single column)
def fig_structural_v3():
    s = pd.read_csv(MET / "structural_uncertainty_summary.csv")
    order = ["PINN_mono", "Linear(t)", "Poly2(t)"]
    s = s.set_index("model").loc[order].reset_index()
    cmap = {"Poly2(t)": C["tonly"], "Linear(t)": C["dummy"],
            "PINN_mono": C["pinn"]}
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for i, r in s.iterrows():
        c = cmap[r.model]
        ax.plot([r.epistemic_t_failure_q025, r.epistemic_t_failure_q975],
                [i, i], color=c, lw=9, alpha=0.35, solid_capstyle="round")
        ax.plot(r.t_failure_point, i, "X", color=c, ms=14, zorder=5)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(["PINN$_{mono}$", "Linear(t)", "Poly2(t)"], fontsize=11)
    ax.axvline(10, color="k", ls=":", lw=1.3)
    ax.text(10.04, 2.35, "last obs.\n(t = 10)", fontsize=9.5, va="center")
    ax.text(11.78, -0.42, "Poly2 and PINN bands do not\noverlap: structural "
            "uncertainty\ndominates the epistemic spread",
            fontsize=9.5, va="bottom")
    ax.set_xlabel("$t_{fail}$ (threshold crossing, order steps)")
    ax.set_xlim(9.8, 12.95)
    ax.set_ylim(-0.55, 2.85)
    _save(fig, "uncertainty_structural_v3.png")


if __name__ == "__main__":
    fig_timeaware_v3()
    fig_pinn_ablation_v3()
    fig_vb_curve_v3()
    fig_benchmark_v3()
    fig_structural_v3()
    fig_rul_curves_v3()   # ultimo: reentrena la PINN (~40 s)
    print("[done] 6 figures v3")
