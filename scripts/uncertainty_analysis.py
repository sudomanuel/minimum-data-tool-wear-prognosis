"""
uncertainty_analysis.py — P5: incertidumbre descriptiva sobre VB(t)/t_failure/RUL.

  Estructural : desacuerdo Poly2(t) vs PINN_mono vs Linear(t) (lee P4).
  Epistemica  : PolyBootstrapEnsemble (B=500, residual bootstrap, deg 2 y 1)
                + PINNDeepEnsemble (K=10 PINN_mono, seeds distintos).

CAVEAT: n=10 -> bandas DESCRIPTIVAS, no calibradas.

Outputs:
  outputs/metrics/uncertainty_vb_bands.csv
  outputs/metrics/uncertainty_failure_distribution.csv
  outputs/metrics/uncertainty_rul_distribution.csv
  outputs/metrics/structural_uncertainty_summary.csv
  outputs/figures/uncertainty_vb_bands.png
  outputs/figures/uncertainty_t_failure_distribution.png
  outputs/figures/uncertainty_rul_bands.png
  outputs/figures/structural_uncertainty_model_disagreement.png
  reports/uncertainty_analysis.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phm.config import (PROCESSED_DATASET, EXPERIMENT_ID_COL, EXP_ORDER_COL,
                        TARGET_COLUMN, METRICS_DIR, FIGURES_DIR)
from phm.pinn import resolve_driver_col, select_minimal_physical_features
from phm.uncertainty import (PolyBootstrapEnsemble, PINNDeepEnsemble,
                             failure_and_rul_distributions, first_crossing,
                             summarize_curves)

REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "physics.yaml"
CAVEAT = ("Con n=10 estas bandas son DESCRIPTIVAS, no calibradas; "
          "no constituyen garantia estadistica de cobertura.")


def main() -> int:
    t0 = time.time()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    vb_fail = float(cfg["vb_failure_um"])
    t_max_ext = float(cfg["extrapolation_t_max"])
    step = float(cfg["extrapolation_grid_step"])
    rul_units = cfg["rul_units"]

    df = pd.read_csv(PROCESSED_DATASET).sort_values(EXP_ORDER_COL)
    t_obs = df[EXP_ORDER_COL].values.astype(float)
    vb_obs = df[TARGET_COLUMN].values.astype(float)
    t_max_obs = float(t_obs.max())
    x_min = select_minimal_physical_features(df.columns)
    driver = resolve_driver_col(df.columns)
    grid = np.arange(float(t_obs.min()), t_max_ext + step, step)
    print(f"[P5] umbral={vb_fail}; malla t<={t_max_ext}; {CAVEAT}", flush=True)

    bands, fails, ruls = [], [], []

    # ---------- epistemica: Poly2 y Linear (residual bootstrap) ----------
    for name, deg in (("Poly2(t)", 2), ("Linear(t)", 1)):
        ens = PolyBootstrapEnsemble(degree=deg, n_boot=500).fit(t_obs, vb_obs)
        curves = ens.predict_members(grid)
        bands.append(summarize_curves(grid, curves, name, t_max_obs))
        f, r = failure_and_rul_distributions(grid, curves, vb_fail, t_obs,
                                             name, rul_units)
        fails.append(f); ruls.append(r)
        tfs = f.loc[f.crosses, "t_failure"]
        print(f"  {name:10s} bootstrap B=500: t_failure mediana={tfs.median():.2f} "
              f"IQR=[{tfs.quantile(.25):.2f},{tfs.quantile(.75):.2f}] "
              f"cruzan={f.crosses.sum()}/500", flush=True)

    # ---------- epistemica: PINN deep ensemble ----------
    variant = cfg.get("rul_pinn_variant", "PINN_mono")
    print(f"[P5] PINNDeepEnsemble K=10 ({variant}), seeds base+101k", flush=True)
    pe = PINNDeepEnsemble(variant=variant, n_members=10)
    pe.fit(df[x_min].values, t_obs, vb_obs,
           e_rot=df[driver].values if driver else None, verbose=True)
    curves_p = pe.predict_members(grid)
    bands.append(summarize_curves(grid, curves_p, variant, t_max_obs))
    f, r = failure_and_rul_distributions(grid, curves_p, vb_fail, t_obs,
                                         variant, rul_units)
    fails.append(f); ruls.append(r)
    tfs_p = f.loc[f.crosses, "t_failure"]
    print(f"  {variant} K=10: t_failure mediana={tfs_p.median():.2f} "
          f"min={tfs_p.min():.2f} max={tfs_p.max():.2f} "
          f"cruzan={f.crosses.sum()}/10", flush=True)

    bands_df = pd.concat(bands, ignore_index=True)
    fails_df = pd.concat(fails, ignore_index=True)
    ruls_df = pd.concat(ruls, ignore_index=True)
    for d, name in ((bands_df, "uncertainty_vb_bands.csv"),
                    (fails_df, "uncertainty_failure_distribution.csv"),
                    (ruls_df, "uncertainty_rul_distribution.csv")):
        d.to_csv(METRICS_DIR / name, index=False)
        print(f"[write] {METRICS_DIR / name}", flush=True)

    # ---------- estructural: punto por familia (P4) + spread epistemico ----------
    p4 = pd.read_csv(METRICS_DIR / "rul_threshold_summary.csv")
    rows = []
    for _, pr in p4.iterrows():
        m = pr["model"]
        sub = fails_df[(fails_df.model == m) & fails_df.crosses]
        tf_pt = float(pr["t_failure_pred"])
        rows.append({
            "model": m,
            "t_failure_point": tf_pt,
            "RUL_at_last_obs_point": tf_pt - t_max_obs,
            "epistemic_t_failure_q025": sub.t_failure.quantile(.025) if len(sub) else np.nan,
            "epistemic_t_failure_q975": sub.t_failure.quantile(.975) if len(sub) else np.nan,
            "epistemic_members_crossing": f"{len(sub)}/{(fails_df.model == m).sum()}",
            "structural_delta_vs_Poly2": tf_pt - float(
                p4.loc[p4.model == "Poly2(t)", "t_failure_pred"].iloc[0]),
            "caveat": CAVEAT,
        })
    struct_df = pd.DataFrame(rows)
    struct_df.to_csv(METRICS_DIR / "structural_uncertainty_summary.csv", index=False)
    print(f"[write] {METRICS_DIR / 'structural_uncertainty_summary.csv'}", flush=True)

    # ================= FIGURAS =================
    colors = {"Poly2(t)": "#D7263D", "Linear(t)": "#7A7A7A", variant: "#1B7F5A"}

    # 1) bandas VB(t)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, m in zip(axes, ["Poly2(t)", variant]):
        b = bands_df[bands_df.model == m]
        ax.fill_between(b.t, b.VB_q025, b.VB_q975, alpha=0.18, color=colors[m],
                        label="banda 95% (descriptiva)")
        ax.fill_between(b.t, b.VB_q16, b.VB_q84, alpha=0.30, color=colors[m],
                        label="banda 68%")
        ax.plot(b.t, b.VB_mean, color=colors[m], lw=2, label=f"{m} media ensemble")
        ax.plot(t_obs, vb_obs, "ko", ms=6, label="VB observado")
        ax.axhline(vb_fail, color="#A0521E", ls="--", lw=1.4)
        ax.axvspan(t_max_obs, t_max_ext, alpha=0.06, color="gray")
        ax.set_xlim(0.5, 14)
        ax.set_ylim(60, 380)
        ax.set_title(m)
        ax.set_xlabel("experiment_order")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("VB (µm)")
    fig.suptitle(f"P5 — bandas de VB(t) | {CAVEAT}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "uncertainty_vb_bands.png", dpi=150)
    plt.close(fig)

    # 2) distribucion de t_failure
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in ["Poly2(t)", "Linear(t)", variant]:
        sub = fails_df[(fails_df.model == m) & fails_df.crosses]["t_failure"]
        if len(sub):
            ax.hist(sub, bins=30, alpha=0.55, color=colors[m], density=True,
                    label=f"{m} (n={len(sub)})")
    for _, pr in p4.iterrows():
        ax.axvline(pr["t_failure_pred"], color=colors[pr["model"]], ls="--", lw=1.5)
    ax.axvline(t_max_obs, color="k", ls=":", lw=1.2, label="ultima observacion (t=10)")
    ax.set_xlabel("t_failure (experiment_order)")
    ax.set_ylabel("densidad")
    ax.set_title(f"P5 — distribucion de t_failure por familia | {CAVEAT}", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "uncertainty_t_failure_distribution.png", dpi=150)
    plt.close(fig)

    # 3) bandas de RUL
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in ["Poly2(t)", variant]:
        r = ruls_df[ruls_df.model == m]
        ax.fill_between(r.experiment_order, r.RUL_q025, r.RUL_q975,
                        alpha=0.18, color=colors[m])
        ax.plot(r.experiment_order, r.RUL_mean, "o-", color=colors[m],
                label=f"RUL {m} (media ± banda 95%)")
    ax.set_xlabel("experiment_order")
    ax.set_ylabel(f"RUL ({rul_units})")
    ax.set_title(f"P5 — RUL con bandas epistemicas | {CAVEAT}", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "uncertainty_rul_bands.png", dpi=150)
    plt.close(fig)

    # 4) estructural vs epistemico
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ypos = np.arange(len(struct_df))
    for i, (_, r) in enumerate(struct_df.iterrows()):
        c = colors[r["model"]]
        lo = r["epistemic_t_failure_q025"]
        hi = r["epistemic_t_failure_q975"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color=c, lw=6, alpha=0.35,
                    solid_capstyle="round")
        ax.plot(r["t_failure_point"], i, "X", color=c, ms=14)
    ax.set_yticks(ypos)
    ax.set_yticklabels(struct_df["model"])
    ax.axvline(float(p4.loc[p4.model == "Poly2(t)", "t_failure_pred"].iloc[0]),
               color="#D7263D", ls="--", lw=1)
    ax.axvline(t_max_obs, color="k", ls=":", lw=1.2)
    ax.text(t_max_obs + 0.03, len(struct_df) - 0.6, "t=10 (ultima obs.)",
            fontsize=8.5)
    ax.set_xlabel("t_failure")
    ax.set_title("P5 — desacuerdo ESTRUCTURAL (X = punto por familia) vs\n"
                 "incertidumbre EPISTEMICA (barra = banda 95% del ensemble)",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "structural_uncertainty_model_disagreement.png", dpi=150)
    plt.close(fig)
    print("[write] 4 figuras", flush=True)

    # ================= REPORTE =================
    tf_p2 = fails_df[(fails_df.model == "Poly2(t)") & fails_df.crosses]["t_failure"]
    tf_pn = fails_df[(fails_df.model == variant) & fails_df.crosses]["t_failure"]
    struct_delta = float(struct_df.loc[struct_df.model == variant,
                                       "structural_delta_vs_Poly2"].iloc[0])
    overlap = (tf_pn.quantile(.025) <= tf_p2.quantile(.975)) and \
              (tf_p2.quantile(.025) <= tf_pn.quantile(.975))

    md = f"""# P5 — Uncertainty analysis (descriptivo, T01)

Generado por `scripts/uncertainty_analysis.py` ({time.strftime('%Y-%m-%d')}).

**{CAVEAT}**

## Metodos
- **Estructural** (entre familias): desacuerdo de estimaciones puntuales
  Poly2(t) / {variant} / Linear(t) (curvas de P4).
- **Epistemica** (dentro de familia): residual bootstrap B=500 para los
  polinomios; deep ensemble K=10 ({variant}, seeds base+101k, misma
  arquitectura y lambdas de P3) para la PINN. x fuera de rango: hold-last.

## Resultados clave

| familia | t_failure punto | banda 95% epistemica | cruzan |
|---|---:|---|---|
{chr(10).join(f"| {r['model']} | {r['t_failure_point']:.2f} | [{r['epistemic_t_failure_q025']:.2f}, {r['epistemic_t_failure_q975']:.2f}] | {r['epistemic_members_crossing']} |" for _, r in struct_df.iterrows())}

- Desacuerdo estructural {variant} − Poly2(t): **{struct_delta:+.2f} pasos**.
- Bandas epistemicas {'SE SOLAPAN' if overlap else 'NO se solapan'}: la
  incertidumbre estructural {'queda parcialmente cubierta por' if overlap else 'EXCEDE'}
  el spread dentro de familia — {'aun asi' if overlap else 'por eso'} ninguna familia
  individual "contiene" a la otra y el desacuerdo entre familias debe
  reportarse como fuente separada.
- RUL en t=10: Poly2 media {float(ruls_df[(ruls_df.model=='Poly2(t)') & (ruls_df.experiment_order==t_max_obs)]['RUL_mean'].iloc[0]):.2f} ± {float(ruls_df[(ruls_df.model=='Poly2(t)') & (ruls_df.experiment_order==t_max_obs)]['RUL_std'].iloc[0]):.2f};
  {variant} media {float(ruls_df[(ruls_df.model==variant) & (ruls_df.experiment_order==t_max_obs)]['RUL_mean'].iloc[0]):.2f} ± {float(ruls_df[(ruls_df.model==variant) & (ruls_df.experiment_order==t_max_obs)]['RUL_std'].iloc[0]):.2f} ({rul_units}).

## Lectura
1. La PINN no solo cruza tarde (estructural): su ensemble muestra el spread
   reportado arriba — la no-convexidad del entrenamiento introduce varianza
   de inicializacion que el bootstrap polinomial no tiene.
2. El desacuerdo estructural (forma de la curva al extrapolar) es la fuente
   dominante de incertidumbre de RUL en T01.
3. Nada de esto valida RUL: T01 no cruza el umbral real (max 280 < 300 µm).

## Claims
- Permitido: "structural disagreement between curve families dominates the
  descriptive epistemic spread within each family" (si aplica segun tabla).
- Prohibido: cobertura calibrada, "validated uncertainty", RUL error.
"""
    (REPORTS / "uncertainty_analysis.md").write_text(md, encoding="utf-8")
    print(f"[write] {REPORTS / 'uncertainty_analysis.md'}", flush=True)
    print(f"[P5] TOTAL {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
