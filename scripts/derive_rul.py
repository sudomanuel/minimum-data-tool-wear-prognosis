"""
derive_rul.py — P4: RUL conceptual derivada desde curvas VB(t).

Formula (sin etiquetas RUL inventadas, sin modelo directo de RUL):

    features / t  ->  VB_hat(t)  ->  t_failure = min{ t : VB_hat(t) >= VB_failure }
                  ->  RUL(t_i) = t_failure - t_i

Curvas usadas (P3 cerro la eleccion):
    Poly2(t)   — campeon de T01 en exactitud y coherencia OOF (baseline RUL);
    PINN_mono  — mejor PINN por trade-off (curva fisico-informada candidata),
                 con advertencia: se aplana al final -> puede cruzar tarde;
    Linear(t)  — baseline simple opcional.
    (PINN_full/rate EXCLUIDAS: P3 mostro degradacion fuerte.)

Las curvas se ajustan sobre TODOS los datos observados (esto es derivacion
conceptual de RUL, no validacion: la exactitud OOF honesta de cada familia
quedo establecida en P2/P3). T01 nunca alcanza el umbral (max 280 < 300 um),
por lo tanto NO existe RUL_true y NO se reportan metricas de error de RUL.

Outputs:
    outputs/metrics/rul_derivation_results.csv
    outputs/metrics/rul_threshold_summary.csv
    outputs/figures/rul_derivation_curves.png
    outputs/figures/rul_model_disagreement.png
    reports/rul_derivation.md
"""
from __future__ import annotations

import sys
import time
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

from phm.config import (PROCESSED_DATASET, EXPERIMENT_ID_COL, EXP_ORDER_COL,
                        TARGET_COLUMN, METRICS_DIR, FIGURES_DIR, RANDOM_SEED)
from phm.pinn import (PINNRegressor, PINN_VARIANTS, resolve_driver_col,
                      select_minimal_physical_features)

REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "physics.yaml"
PINN_HIDDEN = (32, 32)
PINN_EPOCHS = 3000


def main() -> int:
    t0 = time.time()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    vb_fail = float(cfg["vb_failure_um"])
    t_max_ext = float(cfg["extrapolation_t_max"])
    step = float(cfg["extrapolation_grid_step"])
    rul_units = cfg["rul_units"]
    print(f"[P4] VB_failure={vb_fail} um ({cfg['vb_failure_source']}); "
          f"horizonte t<={t_max_ext}; unidades={rul_units}", flush=True)

    df = pd.read_csv(PROCESSED_DATASET).sort_values(EXP_ORDER_COL)
    t_obs = df[EXP_ORDER_COL].values.astype(float)
    vb_obs = df[TARGET_COLUMN].values.astype(float)
    eids = df[EXPERIMENT_ID_COL].astype(int).values
    t_min, t_max_obs = float(t_obs.min()), float(t_obs.max())
    max_vb_obs = float(vb_obs.max())
    assert max_vb_obs < vb_fail, "T01 cruza el umbral?! revisar config"

    x_min = select_minimal_physical_features(df.columns)
    driver = resolve_driver_col(df.columns)
    X_obs = df[x_min].values.astype(float)

    # ---------------- curvas continuas ----------------
    grid = np.arange(t_min, t_max_ext + step, step)

    # Poly2 / Linear: polinomios sobre (t, VB) observados
    p2 = np.polyfit(t_obs, vb_obs, 2)
    p1 = np.polyfit(t_obs, vb_obs, 1)
    curves = {
        "Poly2(t)": np.polyval(p2, grid),
        "Linear(t)": np.polyval(p1, grid),
    }
    curve_at_obs = {
        "Poly2(t)": np.polyval(p2, t_obs),
        "Linear(t)": np.polyval(p1, t_obs),
    }

    # PINN_mono: full-data fit; x(t) interpolada en rango, hold-last fuera
    variant = cfg.get("rul_pinn_variant", "PINN_mono")
    print(f"[P4] entrenando {variant} full-data (x_min={x_min})", flush=True)
    pinn = PINNRegressor(hidden=PINN_HIDDEN, epochs=PINN_EPOCHS,
                         random_state=RANDOM_SEED, **PINN_VARIANTS[variant])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pinn.fit(X_obs, t_obs, vb_obs,
                 e_rot=df[driver].values if driver else None)
    X_grid = np.empty((len(grid), X_obs.shape[1]))
    for j in range(X_obs.shape[1]):
        X_grid[:, j] = np.interp(grid, t_obs, X_obs[:, j])  # hold-last fuera de rango
    curves[variant] = pinn.predict(X_grid, grid)
    curve_at_obs[variant] = pinn.predict(X_obs, t_obs)

    # ---------------- cruce con el umbral ----------------
    summary_rows, result_rows = [], []
    t_failure = {}
    for model, vals in curves.items():
        idx = np.argmax(vals >= vb_fail) if np.any(vals >= vb_fail) else -1
        crosses = idx >= 0
        tf = float(grid[idx]) if crosses else np.nan
        t_failure[model] = tf
        status = ("extrapolated_threshold_crossing" if crosses
                  else "no_threshold_crossing")
        warning = ("PINN se aplana al final (P3): cruce tardio probable -> "
                   "riesgo de SOBREESTIMAR RUL" if model == variant else
                   "parabola sin restricciones fuera de rango" if model == "Poly2(t)"
                   else "recta: ignora la aceleracion final del desgaste")
        interp = (f"cruza {vb_fail:.0f} um en t={tf:.2f} "
                  f"({tf - t_max_obs:+.2f} pasos tras la ultima observacion)"
                  if crosses else
                  f"NO cruza {vb_fail:.0f} um dentro de t<={t_max_ext:.0f}")
        summary_rows.append({
            "model": model, "VB_failure_um": vb_fail,
            "t_failure_pred": tf, "max_observed_t": t_max_obs,
            "max_observed_VB": max_vb_obs,
            "extrapolation_horizon": t_max_ext,
            "crosses_threshold": crosses, "warning": warning,
            "interpretation": interp,
        })
        for i in range(len(t_obs)):
            result_rows.append({
                "model": model, "experiment_id": int(eids[i]),
                "experiment_order": float(t_obs[i]),
                "VB_true": float(vb_obs[i]),
                "VB_pred": float(curve_at_obs[model][i]),
                "VB_failure_um": vb_fail,
                "t_failure_pred": tf,
                "RUL_pred": (tf - float(t_obs[i])) if crosses else np.nan,
                "RUL_units": rul_units,
                "crossing_status": status,
                "is_extrapolated": bool(crosses and tf > t_max_obs),
                "notes": ("conceptual: T01 no alcanza el umbral; sin RUL_true; "
                          "curva ajustada a todos los datos observados"),
            })
        print(f"  {model:12s} -> {interp}", flush=True)

    res = pd.DataFrame(result_rows)
    summ = pd.DataFrame(summary_rows)
    out1 = METRICS_DIR / "rul_derivation_results.csv"
    out2 = METRICS_DIR / "rul_threshold_summary.csv"
    res.to_csv(out1, index=False)
    summ.to_csv(out2, index=False)
    print(f"[write] {out1}\n[write] {out2}", flush=True)

    # ---------------- figura 1: curvas + cruce + RUL ----------------
    colors = {"Poly2(t)": "#D7263D", "Linear(t)": "#7A7A7A", variant: "#1B7F5A"}
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(t_obs, vb_obs, "ko", ms=8, zorder=5, label="VB observado")
    for model, vals in curves.items():
        ls = ":" if model == "Linear(t)" else "-"
        ax.plot(grid, vals, ls, color=colors[model], lw=2, alpha=0.9,
                label=f"{model}" + (f"  (t_fail={t_failure[model]:.2f})"
                                    if np.isfinite(t_failure[model])
                                    else "  (no cruza)"))
        if np.isfinite(t_failure[model]):
            ax.plot(t_failure[model], vb_fail, "X", color=colors[model],
                    ms=13, zorder=6)
    ax.axhline(vb_fail, color="#A0521E", ls="--", lw=1.6,
               label=f"VB_failure = {vb_fail:.0f} µm (provisional)")
    ax.axvspan(t_max_obs, t_max_ext, alpha=0.08, color="gray")
    ax.text(t_max_obs + 0.15, 95, "extrapolación\n(sin datos)", fontsize=8.5,
            color="gray")
    # flecha RUL desde la ultima observacion para cada modelo que cruza
    for k, (model, tf) in enumerate(t_failure.items()):
        if np.isfinite(tf):
            y = 120 + 18 * k
            ax.annotate("", xy=(tf, y), xytext=(t_max_obs, y),
                        arrowprops=dict(arrowstyle="->", color=colors[model], lw=1.8))
            ax.text(t_max_obs + 0.1, y + 3,
                    f"RUL({model}) = {tf - t_max_obs:.2f} pasos",
                    fontsize=8.5, color=colors[model])
    ax.set_xlim(t_min - 0.3, min(t_max_ext, max(
        [tf for tf in t_failure.values() if np.isfinite(tf)] + [12]) + 1.5))
    ax.set_ylim(60, 360)
    ax.set_xlabel("experiment_order (proxy temporal)")
    ax.set_ylabel("VB (µm)")
    ax.set_title("P4 — RUL conceptual: VB(t) extrapolada hasta el umbral "
                 "(T01 no alcanza la falla)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    f1 = FIGURES_DIR / "rul_derivation_curves.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # ---------------- figura 2: desacuerdo entre modelos ----------------
    rp = res[res.model == "Poly2(t)"].set_index("experiment_order")["RUL_pred"]
    rn = res[res.model == variant].set_index("experiment_order")["RUL_pred"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(rp.index, rp.values, "o-", color="#D7263D", label="RUL Poly2(t)")
    ax1.plot(rn.index, rn.values, "s--", color="#1B7F5A", label=f"RUL {variant}")
    ax1.set_xlabel("experiment_order"); ax1.set_ylabel(f"RUL_pred ({rul_units})")
    ax1.set_title("RUL derivada por modelo")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.25)
    if rp.notna().any() and rn.notna().any():
        diff = rn - rp
        ax2.bar(diff.index, diff.values, color="#A0521E", alpha=0.85)
        ax2.set_title(f"Desacuerdo {variant} − Poly2(t)\n"
                      f"(incertidumbre ESTRUCTURAL, no estadistica)")
    else:
        nc = variant if rn.isna().all() else "Poly2(t)"
        ax2.text(0.5, 0.5, f"{nc}: no cruza el umbral\nen t<={t_max_ext:.0f} -> "
                 "desacuerdo estructural MAXIMO\n(RUL indefinida para ese modelo)",
                 ha="center", va="center", fontsize=11, transform=ax2.transAxes)
        ax2.set_xticks([])
    ax2.set_xlabel("experiment_order")
    ax2.set_ylabel(f"Δ RUL ({rul_units})")
    ax2.grid(alpha=0.25)
    fig.suptitle("P4 — desacuerdo entre curvas como indicador de incertidumbre")
    fig.tight_layout()
    f2 = FIGURES_DIR / "rul_model_disagreement.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    print(f"[write] {f1}\n[write] {f2}", flush=True)

    # ---------------- reporte ----------------
    tf_p, tf_n = t_failure["Poly2(t)"], t_failure[variant]
    both = np.isfinite(tf_p) and np.isfinite(tf_n)
    rul_last_p = tf_p - t_max_obs if np.isfinite(tf_p) else np.nan
    rul_last_n = tf_n - t_max_obs if np.isfinite(tf_n) else np.nan

    md = f"""# P4 — RUL derivada desde VB(t) (conceptual)

Generado por `scripts/derive_rul.py` ({time.strftime('%Y-%m-%d')}).

## Objetivo
Derivar RUL desde la trayectoria estimada de desgaste, sin inventar etiquetas
RUL, sin modelo directo de RUL y sin reportar error de RUL (T01 no alcanza el
umbral: max VB observado = {max_vb_obs:.0f} < {vb_fail:.0f} µm).

## Formula
    t_failure = min {{ t : VB_hat(t) >= VB_failure }}
    RUL(t_i)  = t_failure - t_i

## Configuracion (config/physics.yaml)
- VB_failure = {vb_fail:.0f} µm — **provisional y configurable** (sin valor
  oficial del experimento; ISO 3685 como referencia tipica).
- t = `experiment_order` — **proxy temporal**, no tiempo fisico; si aparece
  cutting_time/length debe reemplazarlo.
- Horizonte de extrapolacion: t <= {t_max_ext:.0f}; politica de x fuera de
  rango: hold-last-observed.
- Curvas ajustadas a TODOS los datos observados (derivacion conceptual; la
  exactitud OOF honesta quedo establecida en P2/P3: Poly2 4.96, {variant} 18.57).

## Resultados

| Modelo | cruza umbral | t_failure | RUL en t={t_max_obs:.0f} (ultimo obs.) |
|---|---|---:|---:|
{chr(10).join(f"| {m} | {'SI' if np.isfinite(t_failure[m]) else 'NO (t<=' + format(t_max_ext, '.0f') + ')'} | {t_failure[m]:.2f} |".replace('nan', '—') + (f" {t_failure[m]-t_max_obs:.2f} |" if np.isfinite(t_failure[m]) else ' — |') for m in curves)}

## Desacuerdo entre modelos
{f'''Poly2(t) cruza en t={tf_p:.2f}; {variant} cruza en t={tf_n:.2f}.
Diferencia = {abs(tf_n-tf_p):.2f} pasos ({abs(tf_n-tf_p)/max(rul_last_p,1e-9)*100:.0f}% de la RUL Poly2 en el ultimo punto).
{variant} cruza {'DESPUES' if tf_n>tf_p else 'antes'} que Poly2(t) — consistente con el aplanamiento
end-of-life detectado en P3 ({'sobreestima' if tf_n>tf_p else 'subestima'} RUL).''' if both else
f'''{variant if not np.isfinite(tf_n) else 'Poly2(t)'} NO cruza el umbral dentro del horizonte
t<={t_max_ext:.0f}: el aplanamiento end-of-life detectado en P3 se confirma en extrapolacion —
la curva PINN satura por debajo del umbral y deja la RUL INDEFINIDA. El desacuerdo estructural
con Poly2(t) (que cruza en t={tf_p:.2f}) es maximo.'''}

Este desacuerdo es **incertidumbre estructural** (de forma de modelo), no
estadistica: dos familias de curva, igualmente plausibles a ojo de los datos
observados, divergen al extrapolar. Es el argumento empirico para P5
(cuantificacion de incertidumbre) y para exigir datos multi-herramienta.

## Limitaciones
1. Sin RUL_true: el desgaste observado nunca llega a {vb_fail:.0f} µm -> no hay
   cruce real -> **no se puede calcular MAE_RUL ni ninguna metrica de error**.
2. El umbral es provisional; la RUL escala con el (sensibilidad no explorada).
3. t es un proxy ordinal: la RUL esta en pasos de experimento, no en minutos.
4. Extrapolacion fuera del soporte de los datos (zona gris de la figura).
5. La curva PINN asume x constante mas alla del ultimo experimento (declarado).

## Que puede afirmarse
- "RUL is derived from the estimated VB(t) trajectory as the time to reach a
  configurable critical wear threshold."
- "For T01, the RUL result is a conceptual extrapolation because the observed
  trajectory does not reach the threshold."
- "Model disagreement between Poly2(t) and {variant} provides a structural
  uncertainty indicator."

## Que NO puede afirmarse
- "We accurately predict RUL." / "RUL error is X." / "The model is validated
  for RUL." / "The PINN improves RUL prediction."
- "The threshold 300 µm is definitive" (sin fuente oficial).
"""
    REPORTS.mkdir(exist_ok=True)
    out_md = REPORTS / "rul_derivation.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"[write] {out_md}", flush=True)
    print(f"[P4] TOTAL {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
