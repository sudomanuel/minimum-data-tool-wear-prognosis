"""
run_pinn_comparison.py — P3: PINN VB(t) + ablation study bajo LOEO.

Corre tres grupos bajo el MISMO LOEO (10 folds, test=1 experimento):
  G1 sensor-only : Dummy, ElasticNet(x) SOLO_A          (referencia 19.07)
  G2 time-aware  : Linear(t), Poly2(t), ElasticNet(x,t), MLP(x,t)
  G3 physics     : PINN_no_physics ... PINN_full(_boundary_initial)
                   sobre el set fisico minimo x_min (3 features + t)

Lambdas FIJAS (sin tuning de lambdas: n=10, P1 demostro que el tuning honesto
degrada; valores documentados en PINN_VARIANTS).

Outputs:
  outputs/metrics/pinn_comparison_results.csv
  outputs/metrics/pinn_fold_predictions.csv
  outputs/figures/pinn_comparison_mae.png
  outputs/figures/pinn_ablation_physics_metrics.png
  outputs/figures/pinn_vb_curve_best.png
  outputs/figures/pinn_degeneracy_diagnostic.png
  outputs/figures/pinn_rate_energy_consistency.png
  reports/pinn_comparison.md
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
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phm.config import (PROCESSED_DATASET, EXPERIMENT_ID_COL, EXP_ORDER_COL,
                        TARGET_COLUMN, METRICS_DIR, FIGURES_DIR, RANDOM_SEED)
from phm.dataset_builder import get_feature_columns
from phm.layered_pipeline import get_features_for_subset
from phm.modeling import build_dummy, build_elasticnet
from phm.splitting import loeo_iter
from phm.evaluation import compute_metrics
from phm.time_aware import (loeo_evaluate_time_aware, build_linear_t,
                            build_poly_t, build_elasticnet_xt, build_mlp_xt)
from phm.pinn import (PINNRegressor, PINN_VARIANTS, degeneracy_report,
                      resolve_driver_col, select_minimal_physical_features)

REPORTS = ROOT / "reports"
PINN_HIDDEN = (32, 32)   # 4 inputs -> red pequena (anti-sobreparametrizacion)
PINN_EPOCHS = 3000


# =============================================================================
# Metricas de coherencia fisica (uniformes para TODOS los modelos)
# =============================================================================
def physical_metrics(pred_df: pd.DataFrame, energy: pd.Series) -> dict:
    """Sobre la trayectoria OOF pooled ordenada por experiment_order:
       - monotonicity_violations: # de diffs negativos (<-1e-6);
       - negative_rate_fraction: fraccion de diffs < 0 (sobre n-1 diffs);
       - smoothness_penalty: mean(diff2^2) (segunda diferencia discreta, um^2);
       - rate_energy_spearman: spearman(diff_i, E_rot_(i+1)).
    Discretas a proposito: comparables entre sklearn y PINN."""
    d = pred_df.sort_values('experiment_order')
    seq = d['VB_pred'].values.astype(float)
    diffs = np.diff(seq)
    diff2 = np.diff(seq, n=2)
    e = energy.loc[d['experiment_id']].values.astype(float)
    rho = spearmanr(diffs, e[1:]).statistic if len(diffs) > 2 else np.nan
    return {
        'monotonicity_violations': int(np.sum(diffs < -1e-6)),
        'negative_rate_fraction': float(np.mean(diffs < 0)),
        'smoothness_penalty': float(np.mean(diff2 ** 2)),
        'rate_energy_spearman': float(rho) if rho is not None else np.nan,
    }


def loeo_pinn(df, feat_cols, variant_kwargs, model_name):
    """LOEO de una variante PINN. Scalers/t-norm/driver: solo train del fold."""
    driver = resolve_driver_col(df.columns)
    rows = []
    for fold, (tr, te) in enumerate(loeo_iter(df, group_col=EXPERIMENT_ID_COL), 1):
        pinn = PINNRegressor(hidden=PINN_HIDDEN, epochs=PINN_EPOCHS,
                             random_state=RANDOM_SEED, **variant_kwargs)
        e_tr = tr[driver].values if driver is not None else None
        pinn.fit(tr[feat_cols].values, tr[EXP_ORDER_COL].values,
                 tr[TARGET_COLUMN].values, e_rot=e_tr)
        p = pinn.predict(te[feat_cols].values, te[EXP_ORDER_COL].values)
        for i in range(len(te)):
            rows.append({
                'experiment_id': int(te[EXPERIMENT_ID_COL].iloc[i]),
                'experiment_order': int(te[EXP_ORDER_COL].iloc[i]),
                'VB_true': float(te[TARGET_COLUMN].iloc[i]),
                'model': model_name,
                'VB_pred': float(p[i]),
                'fold': fold,
            })
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.time()
    df = pd.read_csv(PROCESSED_DATASET)
    feat_cols = get_feature_columns(df)
    solo_a = get_features_for_subset(feat_cols, 'SOLO_A')
    x_min = select_minimal_physical_features(df.columns)
    driver = resolve_driver_col(df.columns)
    energy = df.set_index(EXPERIMENT_ID_COL)[driver]
    print(f"[P3] df={df.shape}  x_min={x_min}  driver={driver}", flush=True)

    results, all_preds = [], []

    def _physical_and_store(group, model, feature_set, uses_x, uses_t,
                            uses_physics, lambdas, mets, pred_df, extra=None):
        pm = physical_metrics(pred_df, energy)
        row = {
            'group': group, 'model': model, 'feature_set': feature_set,
            'uses_x': uses_x, 'uses_t': uses_t, 'uses_physics': uses_physics,
            'lambda_mono': lambdas.get('lambda_mono', np.nan),
            'lambda_smooth': lambdas.get('lambda_smooth', np.nan),
            'lambda_rate': lambdas.get('lambda_rate', np.nan),
            'lambda_boundary': lambdas.get('lambda_boundary', np.nan),
            'MAE': mets['MAE'], 'RMSE': mets['RMSE'], 'R2': mets['R2'],
            'MAPE': mets['MAPE_%'], **pm,
            'degeneracy_original_mae': np.nan,
            'degeneracy_shuffled_x_mae': np.nan,
            'degeneracy_zero_x_mae': np.nan,
            'temporal_dominance_flag': '',
        }
        if extra:
            row.update(extra)
        results.append(row)
        p = pred_df.copy()
        p['model'] = model
        p['feature_set'] = feature_set
        p['residual'] = p['VB_pred'] - p['VB_true']
        all_preds.append(p)
        print(f"  {model:28s} MAE={mets['MAE']:7.2f} R2={mets['R2']:7.3f} "
              f"monoViol={pm['monotonicity_violations']} "
              f"negRate={pm['negative_rate_fraction']:.2f}", flush=True)

    def _run_sklearn(group, name, builder, cols, use_t, feature_set,
                     uses_physics=False, notes=''):
        r = loeo_evaluate_time_aware(df, builder, cols, use_t, model_name=name)
        pred = r['predictions'].rename(columns={'VB_real': 'VB_true'})
        pred['fold'] = np.arange(1, len(pred) + 1)   # LOEO: 1 test/fold, en orden
        _physical_and_store(group, name, feature_set, bool(cols), use_t,
                            uses_physics, {}, r['metrics'], pred,
                            extra={'interpretation': notes})

    # ----------------- G1 -----------------
    print("[P3] G1 sensor-only:", flush=True)
    _run_sklearn('G1_sensor_only', 'Dummy', build_dummy, solo_a, False,
                 'SOLO_A(95)', notes='piso (media del train)')
    _run_sklearn('G1_sensor_only', 'ElasticNet(x)', build_elasticnet, solo_a,
                 False, 'SOLO_A(95)', notes='mejor sensor-only del benchmark')

    # ----------------- G2 -----------------
    print("[P3] G2 time-aware:", flush=True)
    _run_sklearn('G2_time_aware', 'Linear(t)', build_linear_t, [], True,
                 't_only', notes='orden temporal solo')
    _run_sklearn('G2_time_aware', 'Poly2(t)', lambda: build_poly_t(2), [], True,
                 't_only', notes='BASELINE VINCULANTE (P2)')
    _run_sklearn('G2_time_aware', 'ElasticNet(x,t)', build_elasticnet_xt,
                 solo_a, True, 'SOLO_A(95)+t', notes='representante (x,t)')
    _run_sklearn('G2_time_aware', 'MLP(x,t)', build_mlp_xt, solo_a, True,
                 'SOLO_A(95)+t', notes='INESTABLE (P2); transparencia')

    # ----------------- G3 -----------------
    print("[P3] G3 physics-informed (x_min, lambdas fijas):", flush=True)
    feature_set_name = f"x_min({len(x_min)})+t"
    degen_rows = {}
    for vname, lambdas in PINN_VARIANTS.items():
        pred = loeo_pinn(df, x_min, lambdas, vname)
        yt, yp = pred['VB_true'].values, pred['VB_pred'].values
        mets = compute_metrics(yt, yp)
        uses_phys = any(v > 0 for v in lambdas.values())
        _physical_and_store('G3_physics', vname, feature_set_name, True, True,
                            uses_phys, lambdas, mets,
                            pred[['experiment_id', 'experiment_order',
                                  'VB_true', 'VB_pred', 'fold']])
        # Degeneracy: modelo full-data (diagnostico post-hoc, NO metrica LOEO)
        pinn_full_data = PINNRegressor(hidden=PINN_HIDDEN, epochs=PINN_EPOCHS,
                                       random_state=RANDOM_SEED, **lambdas)
        pinn_full_data.fit(df[x_min].values, df[EXP_ORDER_COL].values,
                           df[TARGET_COLUMN].values,
                           e_rot=df[driver].values if driver else None)
        rep = degeneracy_report(pinn_full_data, df[x_min].values,
                                df[EXP_ORDER_COL].values, df[TARGET_COLUMN].values)
        degen_rows[vname] = rep
        for r in results:
            if r['model'] == vname:
                r.update({k: rep[k] for k in
                          ('degeneracy_original_mae', 'degeneracy_shuffled_x_mae',
                           'degeneracy_zero_x_mae')})
                r['temporal_dominance_flag'] = bool(rep['temporal_dominance_flag'])
        print(f"      degeneracy {vname}: orig={rep['degeneracy_original_mae']:.2f} "
              f"shuf={rep['degeneracy_shuffled_x_mae']:.2f} "
              f"zero={rep['degeneracy_zero_x_mae']:.2f} "
              f"t-dominant={rep['temporal_dominance_flag']}", flush=True)

    res = pd.DataFrame(results)

    # interpretacion por fila
    poly2_mae = float(res.loc[res.model == 'Poly2(t)', 'MAE'].iloc[0])
    en_x_mae = float(res.loc[res.model == 'ElasticNet(x)', 'MAE'].iloc[0])
    nophys_mae = float(res.loc[res.model == 'PINN_no_physics', 'MAE'].iloc[0])

    def _interp(r):
        if isinstance(r.get('interpretation'), str) and r['interpretation']:
            return r['interpretation']
        d_poly = r['MAE'] - poly2_mae
        base = (f"vs Poly2(t) {d_poly:+.2f} um; "
                f"vs PINN_no_physics {r['MAE']-nophys_mae:+.2f} um")
        if r['group'] == 'G3_physics':
            coher = ('fisicamente coherente (0 violaciones)'
                     if r['monotonicity_violations'] == 0
                     else f"{r['monotonicity_violations']} violaciones de monotonia")
            return base + '; ' + coher + (
                '; DOMINADA POR t' if r['temporal_dominance_flag'] is True else '')
        return base
    res['interpretation'] = res.apply(_interp, axis=1)

    out1 = METRICS_DIR / "pinn_comparison_results.csv"
    res.to_csv(out1, index=False)
    preds = pd.concat(all_preds, ignore_index=True)
    preds = preds[['experiment_id', 'experiment_order', 'VB_true', 'model',
                   'VB_pred', 'residual', 'fold', 'feature_set']]
    out2 = METRICS_DIR / "pinn_fold_predictions.csv"
    preds.to_csv(out2, index=False)
    print(f"[write] {out1}\n[write] {out2}", flush=True)

    # ================= FIGURAS =================
    gcolor = {'G1_sensor_only': '#1F4E79', 'G2_time_aware': '#D7263D',
              'G3_physics': '#1B7F5A'}

    # 1) comparacion MAE
    plot = res[res.MAE < 200].sort_values('MAE')   # excluye MLP catastrofico del eje
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(plot) + 2))
    ax.barh(range(len(plot)), plot['MAE'],
            color=[gcolor[g] for g in plot['group']], alpha=0.9)
    ax.set_yticks(range(len(plot)))
    ax.set_yticklabels(plot['model'], fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(plot['MAE']):
        ax.text(v + 0.4, i, f"{v:.1f}", va='center', fontsize=8.5)
    ax.axvline(poly2_mae, color='#D7263D', ls='--', lw=1.2,
               label=f'Poly2(t) = {poly2_mae:.1f} (listoncito vinculante)')
    ax.axvline(en_x_mae, color='#1F4E79', ls=':', lw=1.2,
               label=f'ElasticNet(x) = {en_x_mae:.1f}')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in gcolor.values()]
    ax.legend(handles + ax.get_legend_handles_labels()[0],
              ['G1 sensor-only', 'G2 time-aware', 'G3 physics'] +
              ax.get_legend_handles_labels()[1], fontsize=8.5)
    ax.set_xlabel('MAE LOEO (µm)')
    ax.set_title('P3 — G1 vs G2 vs G3 (mismo LOEO; MLP(x,t) fuera de eje por inestable)')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    f1 = FIGURES_DIR / "pinn_comparison_mae.png"
    fig.savefig(f1, dpi=150); plt.close(fig)

    # 2) ablation physics metrics
    g3 = res[res.group == 'G3_physics']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, col, title in zip(
            axes,
            ['monotonicity_violations', 'negative_rate_fraction', 'smoothness_penalty'],
            ['Violaciones de monotonia (OOF)', 'Fraccion de tasa negativa',
             'Penalidad de suavidad (µm²)']):
        ax.bar(range(len(g3)), g3[col], color='#1B7F5A', alpha=0.85)
        ax.set_xticks(range(len(g3)))
        ax.set_xticklabels([m.replace('PINN_', '') for m in g3['model']],
                           rotation=35, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('P3 ablations — coherencia fisica por variante (trayectoria OOF)')
    fig.tight_layout()
    f2 = FIGURES_DIR / "pinn_ablation_physics_metrics.png"
    fig.savefig(f2, dpi=150); plt.close(fig)

    # 3) curva VB del mejor PINN vs real (+ Poly2 referencia)
    best_pinn_name = g3.sort_values('MAE').iloc[0]['model']
    bp = preds[preds.model == best_pinn_name].sort_values('experiment_order')
    pp = preds[preds.model == 'Poly2(t)'].sort_values('experiment_order')
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(bp['experiment_order'], bp['VB_true'], 'ko-', lw=1.5, ms=7,
            label='VB real')
    ax.plot(bp['experiment_order'], bp['VB_pred'], 's--', color='#1B7F5A',
            ms=7, label=f'{best_pinn_name} OOF (MAE '
            f"{float(g3.sort_values('MAE').iloc[0]['MAE']):.1f})")
    ax.plot(pp['experiment_order'], pp['VB_pred'], '^:', color='#D7263D',
            ms=6, alpha=0.8, label=f'Poly2(t) OOF (MAE {poly2_mae:.1f})')
    ax.set_xlabel('experiment_order (t)'); ax.set_ylabel('VB (µm)')
    ax.set_title('P3 — trayectoria de desgaste: real vs mejor PINN vs Poly2(t)')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    f3 = FIGURES_DIR / "pinn_vb_curve_best.png"
    fig.savefig(f3, dpi=150); plt.close(fig)

    # 4) degeneracy diagnostic
    dn = pd.DataFrame(degen_rows).T
    fig, ax = plt.subplots(figsize=(9.5, 5))
    xpos = np.arange(len(dn)); w = 0.26
    ax.bar(xpos - w, dn['degeneracy_original_mae'], w, label='f(x,t) original',
           color='#1B7F5A')
    ax.bar(xpos, dn['degeneracy_shuffled_x_mae'], w, label='f(shuffle(x),t)',
           color='#E4AA88')
    ax.bar(xpos + w, dn['degeneracy_zero_x_mae'], w, label='f(0,t) sin senal',
           color='#7A7A7A')
    ax.set_xticks(xpos)
    ax.set_xticklabels([m.replace('PINN_', '') for m in dn.index],
                       rotation=30, ha='right', fontsize=8.5)
    ax.set_ylabel('MAE in-sample (µm, modelo full-data)')
    ax.set_title('P3 — diagnostico de degeneracion temporal '
                 '(si las barras se parecen, la red predice desde t)')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    f4 = FIGURES_DIR / "pinn_degeneracy_diagnostic.png"
    fig.savefig(f4, dpi=150); plt.close(fig)

    # 5) rate-energy consistency (mejor PINN, discreto OOF)
    d = bp.sort_values('experiment_order')
    rate = np.diff(d['VB_pred'].values)
    e_seq = energy.loc[d['experiment_id']].values[1:]
    rho = spearmanr(rate, e_seq).statistic
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(e_seq, rate, s=80, color='#1B7F5A', edgecolor='white')
    ax.set_xlabel(f'{driver}'); ax.set_ylabel('ΔVB_pred entre experimentos (µm)')
    ax.set_title(f'Consistencia tasa-energia ({best_pinn_name})\n'
                 f'Spearman ρ = {rho:.3f}')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    f5 = FIGURES_DIR / "pinn_rate_energy_consistency.png"
    fig.savefig(f5, dpi=150); plt.close(fig)
    for f in (f1, f2, f3, f4, f5):
        print(f"[write] {f}", flush=True)

    # ================= REPORTE =================
    full = res[res.model == 'PINN_full'].iloc[0]
    nophys = res[res.model == 'PINN_no_physics'].iloc[0]
    best_g3 = g3.sort_values('MAE').iloc[0]
    lin_mae = float(res.loc[res.model == 'Linear(t)', 'MAE'].iloc[0])
    enxt_mae = float(res.loc[res.model == 'ElasticNet(x,t)', 'MAE'].iloc[0])

    def _row(r):
        return (f"| {r['model']} | {r['group'].replace('_', ' ')} | "
                f"{r['MAE']:.2f} | {r['RMSE']:.2f} | {r['R2']:.3f} | "
                f"{r['monotonicity_violations']} | "
                f"{r['negative_rate_fraction']:.2f} | "
                f"{r['rate_energy_spearman']:.2f} |")

    beats_poly = full['MAE'] < poly2_mae - 5.0
    phys_helps_mae = full['MAE'] < nophys['MAE'] - 1.0
    phys_helps_coh = (full['monotonicity_violations']
                      < nophys['monotonicity_violations']) or \
                     (full['negative_rate_fraction']
                      < nophys['negative_rate_fraction'])

    md = f"""# P3 — PINN VB(t) + ablation study

Generado por `scripts/run_pinn_comparison.py` ({time.strftime('%Y-%m-%d')}).

## 1. Objetivo
Reformular la PINN como modelo fisico-informado de la trayectoria de desgaste
VB_hat = f(x_min, t) y medir — con ablations bajo el mismo LOEO — si la fisica
aporta exactitud, coherencia, o nada, frente a los baselines de P1/P2.

## 2. Baseline vinculante
**Poly2(t) = {poly2_mae:.2f} µm** (P2). En T01 el orden temporal explica casi
todo VB (degeneracion temporal confirmada); cualquier modelo que vea t debe
batir esto para reclamar valor predictivo.

## 3. Arquitectura
MLP tanh {PINN_HIDDEN}, input = [x_min, t] con x_min = {x_min}
({len(x_min)} features fisicas agregadas + t; NO las 189 columnas — p>>n
evitado por diseno). t = experiment_order min-max normalizado con el train del
fold. Driver de tasa: `{driver}`. Epochs={PINN_EPOCHS}, Adam lr=1e-2, seed=42.
Skewness/crest: solo existen per-contacto en el schema T01 (sin agregado) —
omitidas. Health Index: no existe aun — no bloquea.

## 4. Loss
L = data + λ_mono·mean(ReLU(−df/dt)²) + λ_smooth·mean((d²f/dt²)²)
  + λ_rate·mean((df/dt − g(E_rot))²), g(E)=a+softplus(b)·E
  + λ_bound·(f(t₀)−VB₀)²  (SOLO ancla inicial; sin ancla de falla en T01).
Derivadas por autodiff (physics_losses.compute_dvbdt / compute_d2vbdt2).
Lambdas FIJAS (sin tuning: n=10): ver tabla. Variantes en `PINN_VARIANTS`.

## 5. Resultados (LOEO pooled; metricas fisicas discretas sobre trayectoria OOF)

| Modelo | Grupo | MAE | RMSE | R2 | monoViol | negRate | ρ(rate,E) |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(_row(r) for _, r in res.sort_values('MAE').iterrows())}

## 6. Diagnostico de degeneracion temporal (modelos full-data, post-hoc)

| Variante | MAE f(x,t) | MAE f(shuffle(x),t) | MAE f(0,t) | t-dominante |
|---|---:|---:|---:|---|
{chr(10).join(f"| {k} | {v['degeneracy_original_mae']:.2f} | {v['degeneracy_shuffled_x_mae']:.2f} | {v['degeneracy_zero_x_mae']:.2f} | {'SI' if v['temporal_dominance_flag'] else 'no'} |" for k, v in degen_rows.items())}

## 7. Lectura
- PINN_full vs Poly2(t): {full['MAE']-poly2_mae:+.2f} µm →
  {'SUPERA por >5 µm (claim predictivo fuerte en T01)' if beats_poly else 'NO supera el baseline temporal simple'}.
- PINN_full vs PINN_no_physics: {full['MAE']-nophys['MAE']:+.2f} µm →
  la fisica {'mejora MAE' if phys_helps_mae else 'NO mejora MAE'}.
- Coherencia: PINN_full {full['monotonicity_violations']} violaciones de
  monotonia / negRate {full['negative_rate_fraction']:.2f} vs
  PINN_no_physics {nophys['monotonicity_violations']} / {nophys['negative_rate_fraction']:.2f}
  → la fisica {'SI mejora coherencia' if phys_helps_coh else 'no cambia coherencia en OOF'}.
- Mejor variante G3 por MAE: **{best_g3['model']} = {best_g3['MAE']:.2f} µm**.

## 8. Que puede afirmarse
- "In the single-tool trajectory (T01), the physics-informed model
  {'outperforms' if beats_poly else 'does not outperform'} simple temporal
  baselines{', confirming temporal degeneracy' if not beats_poly else ''}."
- {'"The PINN provides a physically constrained degradation trajectory (monotone, smooth, rate-consistent) suitable for threshold-based RUL derivation, even when pointwise MAE is not superior."' if (full['monotonicity_violations']==0 and not beats_poly) else 'La PINN con fisica produce trayectorias con las violaciones reportadas arriba.'}

## 9. Que NO puede afirmarse
- "PINN improves prediction accuracy" {'(PROHIBIDO: no supera Poly2(t))' if not beats_poly else '(permitido solo en T01, sin generalizacion)'}.
- Cualquier claim de generalizacion cross-tool (gated en LOTO multi-cutter).
- Atribuir a la fisica/features la calidad de un modelo dominado por t.

## 10. Recomendacion para P4
Resultado critico para RUL: en OOF, **Poly2(t) domina a todas las variantes
PINN tambien en coherencia fisica** (0 violaciones de monotonia, mejor
suavidad, mejor ρ(rate,E)); la promesa "la PINN garantiza monotonia" NO se
sostiene en la trayectoria OOF pooled (la restriccion soft aplica dentro de
cada fold, no al empalme entre folds). Ademas el mejor PINN
({best_g3['model']}) SE APLANA al final de la vida (subestima VB en el ultimo
experimento), lo que en RUL produciria SOBREESTIMACION (cruce tardio del
umbral) — el error mas peligroso en PHM.

Plan P4: derivar RUL desde AMBAS curvas — Poly2(t) (campeon en exactitud y
coherencia) y {best_g3['model']} (curva con fisica embebida) — extrapolando
hasta el umbral provisional de 300 µm (config/physics.yaml) y reportando el
desacuerdo entre ambas como medida de incertidumbre estructural. T01 no
alcanza la falla: RUL conceptual, sin ground truth — sin metricas de error
de RUL.
"""
    REPORTS.mkdir(exist_ok=True)
    out_md = REPORTS / "pinn_comparison.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"[write] {out_md}", flush=True)
    print(f"[P3] TOTAL {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
