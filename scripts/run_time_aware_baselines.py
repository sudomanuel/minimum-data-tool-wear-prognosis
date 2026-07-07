"""
run_time_aware_baselines.py — P2: baselines time-aware bajo el mismo LOEO.

Corre el Grupo 2 (Linear(t), Poly2(t), ElasticNet(x,t), MLP(x,t) por vista),
recalcula las referencias sensor-only (Dummy, ElasticNet SOLO_A — debe
reproducir el 19.07 del benchmark como sanity check) y genera:

    outputs/metrics/time_aware_results.csv
    outputs/metrics/time_aware_vs_sensor_only.csv
    outputs/figures/time_aware_actual_vs_predicted.png
    outputs/figures/time_aware_comparison.png
    reports/time_aware_baselines.md

Sin tuning (ST): P1 demostro que el tuning honesto degrada a n=10.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phm.config import (PROCESSED_DATASET, EXPERIMENT_ID_COL, EXP_ORDER_COL,
                        METRICS_DIR, FIGURES_DIR)
from phm.dataset_builder import get_feature_columns
from phm.layered_pipeline import get_features_for_subset
from phm.modeling import build_dummy, build_elasticnet
from phm.time_aware import (T_POLICY, loeo_evaluate_time_aware,
                            time_aware_model_grid)

REPORTS = ROOT / "reports"
SENSOR_BEST_EXPECTED = 19.07  # ElasticNet @ SOLO_A_N_ST (P1)


def main() -> int:
    t0 = time.time()
    df = pd.read_csv(PROCESSED_DATASET)
    assert EXP_ORDER_COL in df.columns, f"falta {EXP_ORDER_COL} en el dataset"
    feat_cols = get_feature_columns(df)
    views = {v: get_features_for_subset(feat_cols, v)
             for v in ("SOLO_A", "SOLO_R", "FUSION")}
    print(f"[time-aware] df={df.shape} exps={df[EXPERIMENT_ID_COL].nunique()} "
          f"feats: " + ", ".join(f"{k}={len(c)}" for k, c in views.items()),
          flush=True)

    # ----------------- referencias sensor-only (Grupo 1) -----------------
    results, preds, res_by_key = [], {}, {}

    def _run(name, builder, cols, use_t, view, group, notes=""):
        r = loeo_evaluate_time_aware(df, builder, cols, use_t, model_name=name)
        m = r['metrics']
        flag = f"; {r['n_failed_folds']} folds fallidos" if r['n_failed_folds'] else ""
        row = {
            'baseline_group': group, 'model': name, 'feature_view': view,
            'uses_x': bool(cols), 'uses_t': use_t,
            'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2': m['R2'],
            'MAPE': m['MAPE_%'], 't_policy': T_POLICY if use_t else '',
            'notes': notes + flag,
        }
        results.append(row)
        key = name + ('' if view in ('T_ONLY', '-') else f"|{view}")
        preds[key] = r['predictions']
        res_by_key[key] = row
        print(f"  {name:24s} {view:8s} MAE={m['MAE']:7.2f} RMSE={m['RMSE']:7.2f} "
              f"R2={m['R2']:6.3f}{flag}", flush=True)

    print("[time-aware] Grupo 1 — referencias sensor-only (recalculadas):", flush=True)
    _run('Dummy', build_dummy, views['SOLO_A'], False, 'SOLO_A',
         'G1_sensor_only', 'media del train; ignora x')
    _run('ElasticNet(x)', build_elasticnet, views['SOLO_A'], False, 'SOLO_A',
         'G1_sensor_only', 'mejor modelo del benchmark clasico (ST)')

    en_x = next(r for r in results if r['model'] == 'ElasticNet(x)')
    drift = abs(en_x['MAE'] - SENSOR_BEST_EXPECTED)
    print(f"[sanity] ElasticNet(x) SOLO_A MAE={en_x['MAE']:.2f} "
          f"(esperado ~{SENSOR_BEST_EXPECTED}; drift={drift:.3f})", flush=True)
    if drift > 0.5:
        print("[sanity] WARN: no reproduce el benchmark — revisar", flush=True)

    # ----------------- Grupo 2 — time-aware -----------------
    print("[time-aware] Grupo 2 — time-aware:", flush=True)
    for spec in time_aware_model_grid(views):
        _run(spec['model'], spec['builder'], spec['feat_cols'], spec['use_t'],
             spec['feature_view'], 'G2_time_aware', spec['notes'])

    res_df = pd.DataFrame(results)
    out1 = METRICS_DIR / "time_aware_results.csv"
    res_df[['model', 'feature_view', 'uses_x', 'uses_t', 'MAE', 'RMSE', 'R2',
            'MAPE', 't_policy', 'notes']].to_csv(out1, index=False)
    print(f"[write] {out1}", flush=True)

    # ----------------- comparacion vs sensor-only -----------------
    base_mae = float(en_x['MAE'])

    def _interp(row):
        d = row['MAE'] - base_mae
        if row['model'] == 'Dummy':
            return 'piso de referencia (media del train)'
        if row['model'] == 'ElasticNet(x)':
            return 'mejor sensor-only (referencia)'
        if abs(d) < 1.0:
            return f'empate practico con sensor-only ({d:+.2f} um)'
        if d < -5.0:
            return f'mejora fuerte vs sensor-only ({d:+.2f} um): t aporta'
        if d < 0:
            return f'mejora marginal ({d:+.2f} um, <5 um: no robusta a n=10)'
        if d > 5.0:
            return f'peor que sensor-only ({d:+.2f} um)'
        return f'leve degradacion ({d:+.2f} um, <5 um)'

    cmp_rows = res_df[
        (res_df['model'].isin(['Dummy', 'ElasticNet(x)', 'Linear(t)', 'Poly2(t)'])) |
        (res_df['feature_view'].isin(['SOLO_A', 'FUSION']))
    ].copy()
    cmp_rows['delta_vs_sensor_only'] = cmp_rows['MAE'] - base_mae
    cmp_rows['interpretation'] = cmp_rows.apply(_interp, axis=1)
    out2 = METRICS_DIR / "time_aware_vs_sensor_only.csv"
    cmp_rows[['baseline_group', 'model', 'feature_view', 'MAE', 'RMSE', 'R2',
              'MAPE', 'delta_vs_sensor_only', 'interpretation']]\
        .sort_values('MAE').to_csv(out2, index=False)
    print(f"[write] {out2}", flush=True)

    # ----------------- figura 1: actual vs predicted -----------------
    sel = [('ElasticNet(x)|SOLO_A', 'ElasticNet(x) SOLO_A', '#1F4E79', 'o'),
           ('Linear(t)', 'Linear(t)', '#D7263D', 's'),
           ('ElasticNet(x,t)|SOLO_A', 'ElasticNet(x,t) SOLO_A', '#1B7F5A', '^'),
           ('MLP(x,t)|SOLO_A', 'MLP(x,t) SOLO_A', '#A0521E', 'D')]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    lims = [df['VB_um'].min() - 20, df['VB_um'].max() + 20] \
        if 'VB_um' in df.columns else [50, 320]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.6, label='ideal (y=x)')
    for key, label, color, marker in sel:
        p = preds.get(key)
        row = res_by_key.get(key)
        if p is None or row is None:
            continue
        ax.scatter(p['VB_real'], p['VB_pred'], s=70, alpha=0.85, color=color,
                   marker=marker, edgecolor='white', linewidth=0.8,
                   label=f"{label}  (MAE {row['MAE']:.1f})")
    ax.set_xlabel('VB real (µm)'); ax.set_ylabel('VB predicho (µm)')
    ax.set_title('Time-aware baselines — LOEO actual vs predicted (n=10)')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    f1 = FIGURES_DIR / "time_aware_actual_vs_predicted.png"
    fig.savefig(f1, dpi=150); plt.close(fig)
    print(f"[write] {f1}", flush=True)

    # ----------------- figura 2: comparacion MAE -----------------
    plot_df = cmp_rows.sort_values('MAE', ascending=True)
    labels = [f"{r['model']}" + (f" [{r['feature_view']}]"
              if r['feature_view'] not in ('T_ONLY', '-') else '')
              for _, r in plot_df.iterrows()]
    colors = ['#7A7A7A' if r['model'] == 'Dummy'
              else '#1F4E79' if r['baseline_group'] == 'G1_sensor_only'
              else '#D7263D' if r['feature_view'] == 'T_ONLY'
              else '#1B7F5A' for _, r in plot_df.iterrows()]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(plot_df) + 2))
    bars = ax.barh(range(len(plot_df)), plot_df['MAE'], color=colors, alpha=0.9)
    ax.set_yticks(range(len(plot_df))); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(base_mae, color='#1F4E79', ls='--', lw=1.2,
               label=f'mejor sensor-only ({base_mae:.1f})')
    for i, v in enumerate(plot_df['MAE']):
        ax.text(v + 0.6, i, f"{v:.1f}", va='center', fontsize=8.5)
    ax.set_xlabel('MAE LOEO (µm) — menor es mejor')
    ax.set_title('Sensor-only vs t-only vs (x,t) — mismo LOEO, sin tuning')
    ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    f2 = FIGURES_DIR / "time_aware_comparison.png"
    fig.savefig(f2, dpi=150); plt.close(fig)
    print(f"[write] {f2}", flush=True)

    # ----------------- reporte markdown -----------------
    lin_t = next(r for r in results if r['model'] == 'Linear(t)')
    poly_t = next(r for r in results if r['model'] == 'Poly2(t)')
    en_xt_a = next(r for r in results if r['model'] == 'ElasticNet(x,t)'
                   and r['feature_view'] == 'SOLO_A')
    mlp_xt_a = next(r for r in results if r['model'] == 'MLP(x,t)'
                    and r['feature_view'] == 'SOLO_A')
    g2 = [r for r in results if r['baseline_group'] == 'G2_time_aware'
          and np.isfinite(r['MAE'])]
    best_g2 = min(g2, key=lambda r: r['MAE'])
    mlp_unstable = mlp_xt_a['MAE'] > 3 * en_xt_a['MAE']

    def _f(r):
        return (f"| {r['model']} | {r['feature_view']} | {r['MAE']:.2f} | "
                f"{r['RMSE']:.2f} | {r['R2']:.3f} | {r['MAPE']:.1f} |")

    t_dominant = lin_t['MAE'] <= en_x['MAE'] + 5.0
    xt_helps = en_xt_a['MAE'] < en_x['MAE'] - 1.0
    md = f"""# Time-aware baselines (P2)

Generado por `scripts/run_time_aware_baselines.py` ({time.strftime('%Y-%m-%d')}).

## Protocolo

- LOEO externo identico al benchmark clasico (10 folds, test = 1 experimento).
- Por fold: normalizacion de t SOLO con train; imputer/scaler del pipeline
  fitteados SOLO con train; el held-out jamas influye en nada.
- **Politica de t:** {T_POLICY}.
- `experiment_order` (no `experiment_id`) como proxy temporal ordenado.
- Sin tuning (ST): P1 demostro que el tuning honesto degrada a n=10.
- `experiment_order` sigue EXCLUIDO del benchmark sensor-only
  (NON_FEATURE_COLS); solo entra en estos modelos, explicitamente.

## Resultados (LOEO pooled, n=10)

| Modelo | Vista | MAE | RMSE | R2 | MAPE% |
|---|---|---:|---:|---:|---:|
{chr(10).join(_f(r) for r in sorted(results, key=lambda r: r['MAE']))}

## Lectura

- **Linear(t) = {lin_t['MAE']:.2f} µm** vs mejor sensor-only
  ElasticNet(x) = {en_x['MAE']:.2f} µm.
- **ElasticNet(x,t) SOLO_A = {en_xt_a['MAE']:.2f} µm**
  (delta vs sensor-only: {en_xt_a['MAE']-en_x['MAE']:+.2f} µm).
- **MLP(x,t) SOLO_A = {mlp_xt_a['MAE']:.2f} µm**
  (delta vs ElasticNet(x,t): {mlp_xt_a['MAE']-en_xt_a['MAE']:+.2f} µm).

## Estabilidad del MLP

{'**MLP(x,t) INESTABLE**: MAE ' + format(mlp_xt_a['MAE'], '.0f') + ' µm con seed fijo — descartado como baseline principal de P3; se reporta por transparencia.' if mlp_unstable else 'MLP(x,t) estable en esta corrida.'}

## Advertencia de degeneracion temporal

{'**CONFIRMADA**' if t_dominant else 'No confirmada en esta corrida'}: en T01
(una sola herramienta, una sola trayectoria monotona de desgaste) el orden
temporal {'explica la mayor parte de' if t_dominant else 'no basta para explicar'}
la variacion de VB. Cualquier modelo que reciba t — incluida la PINN de P3 —
{'puede parecer bueno SOLO por usar t' if t_dominant else 'debe ademas explotar x'}.

## Baseline vinculante para P3

El mejor baseline time-aware es **{best_g2['model']} [{best_g2['feature_view']}]
= {best_g2['MAE']:.2f} µm**. La PINN de P3 (VB = f(x,t)) debe compararse contra:

1. **{best_g2['model']} = {best_g2['MAE']:.2f} µm** — el listón real (misma
   informacion temporal);
2. ElasticNet(x,t) SOLO_A = {en_xt_a['MAE']:.2f} µm — representante (x,t);
3. ElasticNet(x) SOLO_A = {en_x['MAE']:.2f} µm — solo como referencia
   sensor-only (NO como evidencia de aporte fisico).

Si la PINN no supera a {best_g2['model']}, su valor en T01 NO es predictivo:
solo puede defenderse por coherencia fisica (monotonia, tasa) y por su rol
en RUL/extrapolacion — y debe declararse asi.

Regla de lectura: diferencias <5 µm no son mejoras robustas (n=10).
"""
    REPORTS.mkdir(exist_ok=True)
    out_md = REPORTS / "time_aware_baselines.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"[write] {out_md}", flush=True)
    print(f"[time-aware] TOTAL {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
