"""
uncertainty.py — incertidumbre descriptiva sobre VB(t), t_failure y RUL (P5).

Dos fuentes, separadas a proposito:

  ESTRUCTURAL (entre familias de modelo): Poly2(t) vs PINN_mono vs Linear(t)
    divergen al extrapolar (P4: t_failure 10.77 / 11.49 / 11.84). Se reporta
    como desacuerdo entre estimaciones puntuales de familias distintas.

  EPISTEMICA (dentro de familia):
    - PolyBootstrapEnsemble: residual bootstrap sobre el polinomio
      (refit sobre y* = y_hat + residuos remuestreados, B replicas);
    - PINNDeepEnsemble: K PINN_mono identicas salvo el seed (misma
      arquitectura y lambdas de P3; PINN_full EXCLUIDA por P3).

CAVEAT OBLIGATORIO: con n=10 estas bandas son DESCRIPTIVAS, no calibradas.
No se reporta cobertura como garantia estadistica.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import RANDOM_SEED
from .pinn import PINNRegressor, PINN_VARIANTS

QUANTILES = (0.025, 0.16, 0.84, 0.975)


def first_crossing(grid: np.ndarray, vals: np.ndarray, threshold: float):
    """Primer t de la malla donde vals >= threshold; NaN si no cruza."""
    mask = vals >= threshold
    return float(grid[np.argmax(mask)]) if mask.any() else float("nan")


def summarize_curves(grid: np.ndarray, curves: np.ndarray,
                     model: str, t_max_obs: float) -> pd.DataFrame:
    """curves: (n_members, len(grid)) -> bandas por punto de la malla."""
    qs = np.quantile(curves, QUANTILES, axis=0)
    return pd.DataFrame({
        "model": model, "t": grid,
        "VB_mean": curves.mean(axis=0), "VB_std": curves.std(axis=0),
        "VB_q025": qs[0], "VB_q16": qs[1], "VB_q84": qs[2], "VB_q975": qs[3],
        "is_extrapolation": grid > t_max_obs,
    })


class PolyBootstrapEnsemble:
    """Residual bootstrap de un polinomio VB(t) de grado `degree`.

    fit() ajusta el polinomio central, guarda residuos y refittea B replicas
    sobre y* = y_hat + residuos remuestreados con reemplazo (seed fijo).
    """

    def __init__(self, degree: int = 2, n_boot: int = 500,
                 random_state: int = RANDOM_SEED):
        self.degree = degree
        self.n_boot = n_boot
        self.random_state = random_state

    def fit(self, t: np.ndarray, y: np.ndarray):
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        self.coef_ = np.polyfit(t, y, self.degree)
        y_hat = np.polyval(self.coef_, t)
        resid = y - y_hat
        rng = np.random.RandomState(self.random_state)
        self.members_ = [
            np.polyfit(t, y_hat + rng.choice(resid, size=len(resid),
                                             replace=True), self.degree)
            for _ in range(self.n_boot)
        ]
        return self

    def predict_members(self, grid: np.ndarray) -> np.ndarray:
        return np.vstack([np.polyval(c, grid) for c in self.members_])

    def predict_central(self, grid: np.ndarray) -> np.ndarray:
        return np.polyval(self.coef_, grid)


class PINNDeepEnsemble:
    """K PINN identicas (variante de P3, lambdas fijas) salvo el seed.

    El spread entre miembros aproxima la incertidumbre epistemica de la
    familia PINN (inicializacion + optimizacion no convexa). x fuera del
    rango observado: politica hold-last (igual que P4).
    """

    def __init__(self, variant: str = "PINN_mono", n_members: int = 10,
                 hidden=(32, 32), epochs: int = 3000,
                 base_seed: int = RANDOM_SEED):
        if variant not in PINN_VARIANTS:
            raise ValueError(f"variante desconocida: {variant}")
        self.variant = variant
        self.lambdas = PINN_VARIANTS[variant]
        self.n_members = n_members
        self.hidden = hidden
        self.epochs = epochs
        self.base_seed = base_seed

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray,
            e_rot: Optional[np.ndarray] = None, verbose: bool = False):
        self.t_obs_ = np.asarray(t, float)
        self.X_obs_ = np.asarray(X, float)
        self.members_: List[PINNRegressor] = []
        for k in range(self.n_members):
            seed = self.base_seed + 101 * k
            m = PINNRegressor(hidden=self.hidden, epochs=self.epochs,
                              random_state=seed, **self.lambdas)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X, t, y, e_rot=e_rot)
            self.members_.append(m)
            if verbose:
                print(f"    [ensemble] member {k+1}/{self.n_members} "
                      f"(seed={seed}) OK", flush=True)
        return self

    def _x_for_grid(self, grid: np.ndarray) -> np.ndarray:
        """x(t) interpolada en rango observado; hold-last fuera (como P4)."""
        Xg = np.empty((len(grid), self.X_obs_.shape[1]))
        for j in range(self.X_obs_.shape[1]):
            Xg[:, j] = np.interp(grid, self.t_obs_, self.X_obs_[:, j])
        return Xg

    def predict_members(self, grid: np.ndarray) -> np.ndarray:
        Xg = self._x_for_grid(grid)
        return np.vstack([m.predict(Xg, grid) for m in self.members_])


def failure_and_rul_distributions(grid: np.ndarray, member_curves: np.ndarray,
                                  threshold: float, t_obs: np.ndarray,
                                  model: str, rul_units: str
                                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Por miembro: t_failure (NaN si no cruza). RUL(t_i) = t_failure - t_i.

    Devuelve (failure_df por miembro, rul_df por t observado con bandas).
    """
    tfs = np.array([first_crossing(grid, c, threshold) for c in member_curves])
    fail_df = pd.DataFrame({
        "model": model, "member": np.arange(len(tfs)),
        "t_failure": tfs, "crosses": np.isfinite(tfs),
    })
    rows = []
    finite = tfs[np.isfinite(tfs)]
    for ti in np.asarray(t_obs, float):
        if len(finite):
            ruls = finite - ti
            qs = np.quantile(ruls, QUANTILES)
            rows.append({
                "model": model, "experiment_order": ti,
                "RUL_mean": float(ruls.mean()), "RUL_std": float(ruls.std()),
                "RUL_q025": qs[0], "RUL_q16": qs[1],
                "RUL_q84": qs[2], "RUL_q975": qs[3],
                "n_members_crossing": int(len(finite)),
                "n_members_total": len(tfs),
                "RUL_units": rul_units,
            })
        else:
            rows.append({
                "model": model, "experiment_order": ti,
                "RUL_mean": np.nan, "RUL_std": np.nan,
                "RUL_q025": np.nan, "RUL_q16": np.nan,
                "RUL_q84": np.nan, "RUL_q975": np.nan,
                "n_members_crossing": 0, "n_members_total": len(tfs),
                "RUL_units": rul_units,
            })
    return fail_df, pd.DataFrame(rows)
