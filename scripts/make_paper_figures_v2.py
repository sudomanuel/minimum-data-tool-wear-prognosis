"""
make_paper_figures_v2.py — paper-quality English figures (P6 stabilization).

Regenerates the figures embedded in the IEEE manuscript with:
  - English labels/titles/legends (originals were Spanish work-figures);
  - IEEE-column-friendly typography (large fonts, simple ink);
  - the exact message each figure must carry (specs A-E of the audit).

Outputs -> paper/overleaf_extracted/.../phm_pinn_paper_skeleton/figures/*_v2.png
Sources -> outputs/metrics/*.csv (numbers identical to the paper tables).
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
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
    "figure.dpi": 200, "savefig.dpi": 200, "axes.grid": True,
    "grid.alpha": 0.25,
})

C = {"tonly": "#D7263D", "xt": "#2E86AB", "sensor": "#1F4E79",
     "pinn": "#1B7F5A", "bad": "#A0521E", "dummy": "#7A7A7A"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


# ---------------------------------------------------------------- 1. time-aware
def fig_timeaware():
    rows = [("Poly2(t)", 4.96, C["tonly"]), ("Linear(t)", 9.93, C["tonly"]),
            ("ElasticNet(x,t)", 17.46, C["xt"]),
            ("PINN$_{mono}$ (initial)", 18.57, C["pinn"]),
            ("ElasticNet(x)", 19.07, C["sensor"]),
            ("Dummy mean", 56.36, C["dummy"])]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    for i, (_n, v, _c) in enumerate(rows):
        ax.text(v + 0.7, i, f"{v:.2f}", va="center", fontsize=11)
    ax.set_xlabel("LOEO MAE (µm) — lower is better")
    ax.set_xlim(0, 62)
    ax.annotate("temporal degeneracy:\na quadratic in $t$ alone beats\nevery sensor-based model",
                xy=(4.96, 0.18), xytext=(26, 1.15), fontsize=11,
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (C["tonly"], C["xt"], C["pinn"], C["sensor"], C["dummy"])]
    ax.legend(handles, ["t-only baseline", "(x,t) baseline",
                        "physics-informed", "sensor-only", "floor"],
              loc="lower right", fontsize=9.5)
    _save(fig, "timeaware_comparison_v2.png")


# ---------------------------------------------------------------- 2. PINN ablation
def fig_pinn_ablation():
    r = pd.read_csv(MET / "pinn_comparison_results.csv")
    g3 = r[r.group == "G3_physics"][["model", "MAE"]]
    order = ["PINN_mono", "PINN_no_physics", "PINN_mono_smooth", "PINN_smooth",
             "PINN_rate", "PINN_full_boundary_initial", "PINN_full"]
    labels = ["mono (best)", "no physics", "mono+smooth", "smooth",
              "rate", "full+bound.", "full"]
    vals = [float(g3.loc[g3.model == m, "MAE"].iloc[0]) for m in order]
    colors = [C["pinn"] if m == "PINN_mono" else
              ("#9CC7A5" if v < 30 else C["bad"]) for m, v in zip(order, vals)]
    fig, ax = plt.subplots(figsize=(7, 4.0))
    y = np.arange(len(order))
    ax.barh(y, vals, color=colors, alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 1.2, i, f"{v:.1f}", va="center", fontsize=11)
    l1 = ax.axvline(4.96, color=C["tonly"], ls="--", lw=1.8,
                    label="Poly2(t) = 4.96 (binding temporal baseline)")
    l2 = ax.axvline(19.07, color=C["sensor"], ls=":", lw=1.8,
                    label="ElasticNet(x) = 19.07 (sensor-only)")
    ax.annotate("rate-law misspecification\n(negative target rates)",
                xy=(76.7, 4), xytext=(55, 1.9), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=C["bad"], lw=1.2))
    ax.legend(handles=[l1, l2], loc="lower right", fontsize=10)
    ax.set_xlabel("LOEO MAE (µm) — lower is better")
    ax.set_title("Initial PINN ablations: evolution baseline, not final verdict",
                 fontsize=12.5)
    _save(fig, "pinn_comparison_mae_v2.png")


# ---------------------------------------------------------------- 3. physical metrics
def fig_physical_metrics():
    r = pd.read_csv(MET / "pinn_comparison_results.csv")
    models = ["Poly2(t)", "PINN_no_physics", "PINN_mono", "PINN_smooth",
              "PINN_rate", "PINN_full"]
    short = ["Poly2(t)", "no-phys", "mono", "smooth", "rate", "full"]
    sub = r.set_index("model").loc[models]
    cols = [("monotonicity_violations", "Monotonicity violations (OOF)"),
            ("negative_rate_fraction", "Negative-rate fraction (OOF)"),
            ("smoothness_penalty", "Smoothness penalty (µm², log)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (c, title) in zip(axes, cols):
        v = sub[c].values.astype(float)
        colors = [C["tonly"]] + [C["pinn"]] * (len(models) - 1)
        ax.bar(range(len(models)), v, color=colors, alpha=0.92)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=10)
        ax.set_title(title, fontsize=11.5)
        if c == "smoothness_penalty":
            ax.set_yscale("log")
        for i, val in enumerate(v):
            ax.text(i, val, f"{val:.2g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Physical coherence on the pooled out-of-fold trajectory: "
                 "the unconstrained quadratic is more coherent than the "
                 "soft-constrained PINNs", fontsize=12, y=1.04)
    _save(fig, "pinn_physical_metrics_v2.png")


# ---------------------------------------------------------------- 4. OOF wear curve
def fig_vb_curve():
    p = pd.read_csv(MET / "pinn_fold_predictions.csv")
    bp = p[p.model == "PINN_mono"].sort_values("experiment_order")
    pp = p[p.model == "Poly2(t)"].sort_values("experiment_order")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(bp.experiment_order, bp.VB_true, "ko-", lw=1.6, ms=8,
            label="observed VB")
    ax.plot(bp.experiment_order, bp.VB_pred, "s--", color=C["pinn"], ms=8,
            label="PINN$_{mono}$ OOF (MAE 18.6)")
    ax.plot(pp.experiment_order, pp.VB_pred, "^:", color=C["tonly"], ms=7,
            alpha=0.85, label="Poly2(t) OOF (MAE 5.0)")
    ax.annotate("end-of-life flattening:\nPINN underestimates final wear",
                xy=(10, 209), xytext=(6.1, 245), fontsize=11,
                arrowprops=dict(arrowstyle="->", color=C["pinn"], lw=1.3))
    ax.set_xlabel("experiment order $t$ (temporal proxy)")
    ax.set_ylabel("VB (µm)")
    ax.legend(loc="upper left", fontsize=10.5)
    _save(fig, "pinn_vb_curve_v2.png")


# ---------------------------------------------------------------- 5. RUL curves
def fig_rul_curves():
    from phm.config import PROCESSED_DATASET, EXP_ORDER_COL, TARGET_COLUMN, RANDOM_SEED
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
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(t_obs, vb, "ko", ms=8, zorder=5, label="observed VB (max 280)")
    for name, (vals, color, ls) in curves.items():
        idx = np.argmax(vals >= vb_fail)
        tf = grid[idx] if (vals >= vb_fail).any() else np.nan
        ax.plot(grid, vals, ls, color=color, lw=2.2,
                label=f"{name}: $t_{{fail}}$ = {tf:.2f}")
        ax.plot(tf, vb_fail, "X", color=color, ms=14, zorder=6)
    ax.axhline(vb_fail, color=C["bad"], ls="--", lw=1.8)
    ax.text(5.6, vb_fail + 7, "VB$_{fail}$ = 300 µm (provisional, configurable)",
            color=C["bad"], fontsize=10.5)
    ax.axvspan(10, 13.6, alpha=0.08, color="gray")
    ax.text(10.25, 100, "extrapolation\n(no data; conceptual RUL)",
            fontsize=10.5, color="gray")
    ax.set_xlim(0.7, 13.5)
    ax.set_ylim(60, 360)
    ax.set_xlabel("experiment order $t$ (temporal proxy)")
    ax.set_ylabel("VB (µm)")
    ax.legend(loc="upper left", fontsize=10.5)
    _save(fig, "rul_curves_v2.png")


# ---------------------------------------------------------------- 6. VB bands
def fig_vb_bands():
    b = pd.read_csv(MET / "uncertainty_vb_bands.csv")
    from phm.config import PROCESSED_DATASET, EXP_ORDER_COL, TARGET_COLUMN
    df = pd.read_csv(PROCESSED_DATASET).sort_values(EXP_ORDER_COL)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, m, color in ((axes[0], "Poly2(t)", C["tonly"]),
                         (axes[1], "PINN_mono", C["pinn"])):
        d = b[b.model == m]
        ax.fill_between(d.t, d.VB_q025, d.VB_q975, alpha=0.18, color=color,
                        label="95% band (descriptive)")
        ax.fill_between(d.t, d.VB_q16, d.VB_q84, alpha=0.32, color=color,
                        label="68% band")
        ax.plot(d.t, d.VB_mean, color=color, lw=2,
                label=("bootstrap mean (B=500)" if m == "Poly2(t)"
                       else "ensemble mean (K=10)"))
        ax.plot(df[EXP_ORDER_COL], df[TARGET_COLUMN], "ko", ms=6,
                label="observed VB")
        ax.axhline(300, color=C["bad"], ls="--", lw=1.5)
        ax.axvspan(10, 14, alpha=0.07, color="gray")
        ax.set_xlim(0.5, 14)
        ax.set_ylim(60, 380)
        ax.set_title("Poly2(t) — residual bootstrap" if m == "Poly2(t)"
                     else "PINN$_{mono}$ — deep ensemble", fontsize=12)
        ax.set_xlabel("experiment order $t$")
        ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel("VB (µm)")
    fig.suptitle("Descriptive VB(t) bands (n=10: not calibrated coverage)",
                 fontsize=12, y=1.02)
    _save(fig, "uncertainty_vb_bands_v2.png")


# ---------------------------------------------------------------- 7. structural
def fig_structural():
    s = pd.read_csv(MET / "structural_uncertainty_summary.csv")
    order = ["PINN_mono", "Linear(t)", "Poly2(t)"]
    s = s.set_index("model").loc[order].reset_index()
    cmap = {"Poly2(t)": C["tonly"], "Linear(t)": C["dummy"],
            "PINN_mono": C["pinn"]}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, r in s.iterrows():
        c = cmap[r.model]
        ax.plot([r.epistemic_t_failure_q025, r.epistemic_t_failure_q975],
                [i, i], color=c, lw=9, alpha=0.35, solid_capstyle="round")
        ax.plot(r.t_failure_point, i, "X", color=c, ms=15, zorder=5)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(["PINN$_{mono}$", "Linear(t)", "Poly2(t)"])
    ax.axvline(10, color="k", ls=":", lw=1.3)
    ax.text(10.03, 2.30, "last observation\n(t = 10)", fontsize=10)
    ax.annotate("", xy=(11.24, 2.0), xytext=(11.0, 2.0),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.4))
    ax.text(10.74, 1.72, "bands do not overlap:\nstructural (model-form)\n"
            "uncertainty dominates", fontsize=10.5)
    ax.set_xlabel("$t_{fail}$ (threshold crossing, experiment-order steps)")
    ax.set_title("Point estimate (X) and 95% descriptive epistemic band per "
                 "curve family", fontsize=12)
    ax.set_xlim(9.8, 12.7)
    _save(fig, "uncertainty_structural_v2.png")


# ---------------------------------------------------------------- 8. roadmap
def fig_roadmap():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axis("off")
    boxes = [
        (0.50, 0.90, "T01 single-tool stage (this paper)\nhonest baselines · frozen PINN scorecard · failure modes", "#EAF0F6", C["sensor"]),
        (0.50, 0.68, "Multi-tool ingestion (engine ready — NOT optional)\nmanifest: tool_id · cumulative cutting time/length ·\nfailure events · process metadata", "#FDF3E7", C["bad"]),
        (0.50, 0.46, "Leave-One-Tool-Out validation\ntool-level splits · early-life prognosis (30/50/70%)\n≥ 1 tool reaching the failure threshold", "#EAF0F6", C["sensor"]),
        (0.50, 0.24, "Improved PINN configurations\npositive rate law g = softplus(a+bE) · monotone-by-\nconstruction VB = VB$_0$ + ∫ softplus(rate) · physical $t$", "#E8F4EE", C["pinn"]),
        (0.50, 0.045, "CLAIM GATE: the PINN is claimed superior only if it beats\ntemporal AND classical baselines under LOTO", "#FBEAEA", C["tonly"]),
    ]
    for x, y, text, fc, ec in boxes:
        ax.add_patch(FancyBboxPatch((x - 0.46, y - 0.075), 0.92, 0.15,
                     boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=1.8,
                     transform=ax.transAxes))
        ax.text(x, y, text, ha="center", va="center", fontsize=10.3,
                transform=ax.transAxes)
    for y0, y1 in [(0.825, 0.755), (0.605, 0.535), (0.385, 0.315),
                   (0.165, 0.12)]:
        ax.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1),
                     transform=ax.transAxes, arrowstyle="-|>",
                     mutation_scale=22, lw=1.8, color="#333333"))
    _save(fig, "multitool_roadmap_v2.png")


# ---------------------------------------------------------------- 9. benchmark
def fig_benchmark():
    m = pd.read_csv(MET / "layered_pipeline" / "09_all_metrics.csv")
    m = m[m.validation_type == "loeo"].dropna(subset=["MAE"])
    views = ["SOLO_A", "SOLO_R", "FUSION"]
    conds = [("defaults (ST, real)", lambda d: d[(d.tuning_method == "none") & (d.data_branch == "N")]),
             ("nested tuning (CT)", lambda d: d[d.tuning_method.isin(["random", "grid"]) & (d.data_branch == "N")]),
             ("augmented (best)", lambda d: d[(d.data_branch == "A")])]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
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
            ax.text(x, val + 0.6, f"{val:.1f}", ha="center", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(["SOLO_A (axial)", "SOLO_R (rotational)", "FUSION"])
    ax.set_ylabel("best LOEO MAE (µm)")
    ax.set_title("Sensor-only benchmark: the axial view dominates; honest "
                 "nested tuning degrades; augmentation does not help",
                 fontsize=11.5)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 38)
    _save(fig, "benchmark_branches_v2.png")


if __name__ == "__main__":
    fig_timeaware()
    fig_pinn_ablation()
    fig_physical_metrics()
    fig_vb_curve()
    fig_vb_bands()
    fig_structural()
    fig_roadmap()
    fig_benchmark()
    fig_rul_curves()   # last: trains the PINN (~40 s)
    print("[done] 9 figures v2")
