"""
time_aware.py — baselines time-aware (Grupo 2 del roadmap, Punto 2).

Objetivo: separar cuanto aporta la senal `x`, cuanto aporta solo el tiempo
`t`, y cuanto aporta combinarlos — para que en P3 la PINN se compare contra
el baseline correcto (MLP(x,t) / ElasticNet(x,t)) y no contra modelos
sensor-only que no ven `t`.

Modelos:
    Linear(t)          — regresion lineal sobre t normalizado (solo orden).
    Poly2(t)           — polinomio grado 2 sobre t (curva temporal simple).
    ElasticNet(x, t)   — mismos defaults ST del benchmark + columna t.
    MLP(x, t)          — mismos defaults ST del benchmark + columna t.

Politica de t (registrada en cada fila de resultados):
    t = experiment_order (NO experiment_id), normalizado min-max con
    parametros calculados SOLO en el train de cada fold LOEO:
        t_norm = (t - t_min_train) / (t_max_train - t_min_train)
    El test se transforma con los parametros del train (puede salir de
    [0, 1] si el experimento held-out es el primero o el ultimo: correcto,
    es extrapolacion honesta).

Reglas anti-leakage (identicas al benchmark clasico):
    - LOEO externo identico (loeo_iter, 10 folds, n_test=1);
    - imputer/scaler del pipeline se fittean solo con train (Pipeline);
    - normalizacion de t solo con train;
    - sin tuning (ST): P1 demostro que el tuning honesto degrada a n=10;
    - `experiment_order` NUNCA entra al benchmark sensor-only (esta en
      NON_FEATURE_COLS); solo entra aqui, explicitamente.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from .config import EXPERIMENT_ID_COL, EXP_ORDER_COL, TARGET_COLUMN
from .evaluation import compute_metrics
from .modeling import build_elasticnet, build_mlp
from .splitting import loeo_iter

T_POLICY = ("t=experiment_order; min-max normalizado con train-fold "
            "(t_norm=(t-min_tr)/(max_tr-min_tr)); test transformado con "
            "params del train")


# =============================================================================
# Builders
# =============================================================================
def build_linear_t():
    """Regresion lineal sobre [t_norm]. Mide cuanto predice SOLO el orden."""
    return LinearRegression()


def build_poly_t(degree: int = 2):
    """Polinomio de grado `degree` sobre [t_norm]. Curva temporal simple y
    transparente; baseline de referencia, no modelo principal."""
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('model', LinearRegression()),
    ])


def build_elasticnet_xt():
    """ElasticNet con defaults ST del benchmark (alpha=1.0, l1_ratio=0.5),
    mismo pipeline imputer+scaler. La columna t entra como una feature mas."""
    return build_elasticnet()


def build_mlp_xt():
    """MLP sklearn con defaults ST del benchmark (hidden=(32,), seed fijo).
    Con n=9 de train es potencialmente inestable: se reporta tal cual."""
    return build_mlp()


# =============================================================================
# t handling
# =============================================================================
def normalize_t(train_orders: np.ndarray, test_orders: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Min-max de t con parametros del TRAIN. Devuelve (t_tr, t_te, params)."""
    t_min = float(np.min(train_orders))
    t_max = float(np.max(train_orders))
    span = (t_max - t_min) if (t_max - t_min) > 0 else 1.0
    t_tr = (train_orders.astype(float) - t_min) / span
    t_te = (test_orders.astype(float) - t_min) / span
    return t_tr, t_te, {'t_min_train': t_min, 't_max_train': t_max}


def _design_matrix(df_fold: pd.DataFrame, feat_cols: List[str],
                   t_norm: Optional[np.ndarray]) -> np.ndarray:
    """[x_view | t_norm] — t como ULTIMA columna. Si feat_cols=[] -> solo t."""
    parts = []
    if feat_cols:
        parts.append(df_fold[feat_cols].values.astype(float))
    if t_norm is not None:
        parts.append(t_norm.reshape(-1, 1))
    return np.hstack(parts)


# =============================================================================
# LOEO evaluation
# =============================================================================
def loeo_evaluate_time_aware(df: pd.DataFrame,
                             builder,
                             feat_cols: Optional[List[str]],
                             use_t: bool,
                             model_name: str = '',
                             ) -> Dict:
    """
    LOEO externo identico al benchmark. Por fold:
      - normaliza t con el train (si use_t);
      - selecciona columnas x de la vista (si feat_cols);
      - fit (pipeline fittea imputer/scaler solo con train);
      - predice el experimento held-out.
    Metricas pooled sobre las 10 predicciones. Sin tuning.
    """
    feat_cols = feat_cols or []
    y_true_all, y_pred_all, eids_all, orders_all = [], [], [], []

    for tr_df, te_df in loeo_iter(df, group_col=EXPERIMENT_ID_COL):
        t_tr = t_te = None
        if use_t:
            t_tr, t_te, _ = normalize_t(
                tr_df[EXP_ORDER_COL].values, te_df[EXP_ORDER_COL].values)
        X_tr = _design_matrix(tr_df, feat_cols, t_tr)
        X_te = _design_matrix(te_df, feat_cols, t_te)
        y_tr = tr_df[TARGET_COLUMN].values.astype(float)
        y_te = te_df[TARGET_COLUMN].values.astype(float)

        from sklearn.base import clone
        est = clone(builder) if not callable(builder) else builder()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                est.fit(X_tr, y_tr)
                y_p = est.predict(X_te)
        except Exception as exc:
            warnings.warn(f"[time_aware/{model_name}] fold fallo: {exc}")
            y_p = np.full_like(y_te, np.nan, dtype=float)

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(np.asarray(y_p, dtype=float).tolist())
        eids_all.extend(te_df[EXPERIMENT_ID_COL].astype(int).tolist())
        orders_all.extend(te_df[EXP_ORDER_COL].astype(int).tolist())

    yt = np.array(y_true_all, dtype=float)
    yp = np.array(y_pred_all, dtype=float)
    ok = np.isfinite(yp)
    mets = (compute_metrics(yt[ok], yp[ok]) if ok.any()
            else {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE_%': np.nan})
    return {
        'metrics': mets,
        'predictions': pd.DataFrame({
            'model': model_name,
            'experiment_id': eids_all,
            'experiment_order': orders_all,
            'VB_real': yt,
            'VB_pred': yp,
        }),
        'n_failed_folds': int((~ok).sum()),
    }


# =============================================================================
# Grid de modelos del Grupo 2
# =============================================================================
def time_aware_model_grid(feat_cols_by_view: Dict[str, List[str]]) -> List[dict]:
    """
    Devuelve la lista de configuraciones a evaluar:
        Linear(t), Poly2(t)                          (solo t)
        ElasticNet(x,t), MLP(x,t)  x  {SOLO_A, SOLO_R, FUSION}
    """
    grid = [
        {'model': 'Linear(t)', 'feature_view': 'T_ONLY',
         'builder': build_linear_t, 'feat_cols': [], 'use_t': True,
         'notes': 'solo orden temporal normalizado'},
        {'model': 'Poly2(t)', 'feature_view': 'T_ONLY',
         'builder': lambda: build_poly_t(2), 'feat_cols': [], 'use_t': True,
         'notes': 'curva temporal cuadratica (referencia transparente)'},
    ]
    for view, cols in feat_cols_by_view.items():
        grid.append({'model': 'ElasticNet(x,t)', 'feature_view': view,
                     'builder': build_elasticnet_xt, 'feat_cols': cols,
                     'use_t': True,
                     'notes': 'defaults ST del benchmark + t (sin tuning)'})
    for view, cols in feat_cols_by_view.items():
        grid.append({'model': 'MLP(x,t)', 'feature_view': view,
                     'builder': build_mlp_xt, 'feat_cols': cols,
                     'use_t': True,
                     'notes': 'hidden=(32,), seed fijo, sin tuning; '
                              'reportar inestabilidad'})
    return grid
