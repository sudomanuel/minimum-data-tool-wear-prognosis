"""
pinn.py — Physics-Informed Neural Network (deterministic) for flank wear.

Wear-curve formulation (P3):

    VB = f_theta(x_min, t)

where `x_min` is a MINIMAL physical feature set (NOT the 189-column matrix;
p>>n is avoided by design) and `t` is the normalized temporal coordinate
(experiment_order). Physical constraints are imposed by autodiff w.r.t. `t`
using the separated, testable terms in `physics_losses.py`:

    L = data_loss
        + lambda_mono   * monotonicity_loss(df/dt)        wear never decreases
        + lambda_smooth * smoothness_loss(d2f/dt2)        smooth trajectory
        + lambda_rate   * wear_rate_loss(df/dt, E_rot)    rate tracks rot. energy
        + lambda_bound  * boundary_loss(f(t0), VB_0)      initial anchor ONLY

No final-failure boundary on T01 (the tool never reaches the threshold).
The estimator takes `t` explicitly in fit/predict, so it runs through its own
LOEO routine (`loeo_evaluate_pinn`), not the generic fit(X,y) harness.

Honest note (P2 closed this quantitatively): on a single tool, Poly2(t) alone
reaches MAE 4.96 um, so ANY model that sees `t` can look good through `t`
alone. `degeneracy_report` measures exactly that. Physics value claims on T01
are gated on beating t-only baselines or on physical-coherence metrics.
"""
from __future__ import annotations

import warnings

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as _exc:  # pragma: no cover
    TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _exc

from .config import (
    EXPERIMENT_ID_COL, EXP_ORDER_COL, TARGET_COLUMN, PHYSICS_DRIVER_COL,
    PHYSICS_DRIVER_CANDIDATES, RANDOM_SEED,
)
from . import physics_losses as PL


# =============================================================================
# Minimal physical feature set (P3: avoid p>>n inside the PINN)
# =============================================================================
# Preference order per signal role; the first existing column wins.
_MINIMAL_FEATURE_ROLES = [
    ("rot_energy", ["R_energy_total_6_contacts", "R_energy_total"]),
    ("axial_rms",  ["A_rms_mean_6_contacts", "A_rms_mean"]),
    ("rot_rms",    ["R_rms_mean_6_contacts", "R_rms_mean"]),
    # axial skewness / crest factor: only aggregate columns qualify; the
    # legacy T01 schema only has per-contact versions -> omitted (documented).
    ("axial_skew", ["A_skewness_mean_6_contacts", "A_skewness_mean"]),
    ("axial_crest", ["A_crest_factor_mean_6_contacts", "A_crest_factor_mean"]),
    # optional Health Index if a previous stage produced it; absent -> skip.
    ("health_index", ["health_index", "HI"]),
]


def select_minimal_physical_features(columns) -> list:
    """Devuelve el set fisico minimo disponible en `columns` (en orden de rol).

    No bloquea si faltan roles opcionales (skewness/crest/HI); exige al menos
    una columna. Disenado para el dataset T01 (agregados *_6_contacts) y para
    el esquema multitool futuro (R_energy_total, A_rms_mean...).
    """
    cols = set(columns)
    out = []
    for _role, candidates in _MINIMAL_FEATURE_ROLES:
        for c in candidates:
            if c in cols:
                out.append(c)
                break
    if not out:
        raise ValueError("select_minimal_physical_features: ninguna columna "
                         "fisica minima encontrada en el dataset")
    return out


# =============================================================================
# Ablation variants (fixed lambdas, NO lambda tuning at n=10 — documented)
# =============================================================================
PINN_VARIANTS = {
    "PINN_no_physics":  dict(lambda_mono=0.0, lambda_smooth=0.0,
                             lambda_rate=0.0, lambda_boundary=0.0),
    "PINN_mono":        dict(lambda_mono=1.0, lambda_smooth=0.0,
                             lambda_rate=0.0, lambda_boundary=0.0),
    "PINN_smooth":      dict(lambda_mono=0.0, lambda_smooth=0.1,
                             lambda_rate=0.0, lambda_boundary=0.0),
    "PINN_rate":        dict(lambda_mono=0.0, lambda_smooth=0.0,
                             lambda_rate=0.1, lambda_boundary=0.0),
    "PINN_mono_smooth": dict(lambda_mono=1.0, lambda_smooth=0.1,
                             lambda_rate=0.0, lambda_boundary=0.0),
    "PINN_full":        dict(lambda_mono=1.0, lambda_smooth=0.1,
                             lambda_rate=0.1, lambda_boundary=0.0),
    # boundary inicial: re-pondera la observacion mas temprana del train
    # (la curva parte del desgaste inicial observado, ~85 um en T01).
    "PINN_full_boundary_initial":
                        dict(lambda_mono=1.0, lambda_smooth=0.1,
                             lambda_rate=0.1, lambda_boundary=1.0),
}


if TORCH_AVAILABLE:

    class _MLP(nn.Module):
        """Plain tanh MLP mapping [x, t] -> VB (scaled)."""

        def __init__(self, in_dim: int, hidden=(64, 64)):
            super().__init__()
            layers = []
            d = in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.Tanh()]
                d = h
            layers += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, xt):
            return self.net(xt).squeeze(-1)


class PINNRegressor:
    """Deterministic wear-curve PINN. Takes `t` explicitly in fit/predict.

    Todas las lambdas en 0 => MLP puro sobre [x, t] (PINN_no_physics).
    Internamente: imputacion por mediana + estandarizacion de x, min-max de t
    y estandarizacion de y — todos los parametros se fitean SOLO con el train
    que recibe fit() (en LOEO: los 9 experimentos del fold).
    """

    def __init__(self, hidden=(64, 64), lambda_mono: float = 1.0,
                 lambda_rate: float = 0.1, lambda_smooth: float = 0.0,
                 lambda_boundary: float = 0.0,
                 lr: float = 1e-2, epochs: int = 3000,
                 random_state: int = RANDOM_SEED, verbose: bool = False):
        self.hidden = hidden
        self.lambda_mono = lambda_mono
        self.lambda_rate = lambda_rate
        self.lambda_smooth = lambda_smooth
        self.lambda_boundary = lambda_boundary
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

    # ---- scaling helpers ---------------------------------------------------
    def _fit_scalers(self, X, t, y, e):
        # Median imputation (the harness models get this from SimpleImputer;
        # the PINN bypasses the Pipeline so it imputes here). Raw features may
        # contain NaN, e.g. an experiment missing some contact segments.
        self.x_median_ = np.nanmedian(X, axis=0)
        self.x_median_[np.isnan(self.x_median_)] = 0.0
        Xi = np.where(np.isnan(X), self.x_median_, X)
        self.x_mean_ = Xi.mean(0)
        self.x_std_ = Xi.std(0)
        self.x_std_[self.x_std_ == 0] = 1.0
        self.t_min_ = float(np.nanmin(t))
        self.t_rng_ = float(np.nanmax(t) - np.nanmin(t)) or 1.0
        self.y_mean_ = float(np.nanmean(y))
        self.y_std_ = float(np.nanstd(y)) or 1.0
        if e is not None:
            self.e_median_ = float(np.nanmedian(e))
            ei = np.where(np.isnan(e), self.e_median_, e)
            self.e_mean_ = float(ei.mean())
            self.e_std_ = float(ei.std()) or 1.0

    def _scale_X(self, X):
        X = np.asarray(X, float)
        X = np.where(np.isnan(X), self.x_median_, X)
        return (X - self.x_mean_) / self.x_std_

    def _scale_t(self, t):
        return (np.asarray(t, float).reshape(-1) - self.t_min_) / self.t_rng_

    # ---- training ----------------------------------------------------------
    def fit(self, X, t, y, e_rot=None):
        if not TORCH_AVAILABLE:
            raise ImportError(
                f"PyTorch no disponible: {_TORCH_IMPORT_ERROR!r}. Instala torch>=2.5."
            )
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))

        X = np.asarray(X, float)
        t = np.asarray(t, float).reshape(-1)
        y = np.asarray(y, float).reshape(-1)
        e = np.asarray(e_rot, float).reshape(-1) if e_rot is not None else None
        self._fit_scalers(X, t, y, e)

        Xs = torch.tensor(self._scale_X(X), dtype=torch.float32)
        ts = torch.tensor(self._scale_t(t), dtype=torch.float32)
        ys = torch.tensor((y - self.y_mean_) / self.y_std_, dtype=torch.float32)
        if e is not None:
            e = np.where(np.isnan(e), self.e_median_, e)
        es = (torch.tensor((e - self.e_mean_) / self.e_std_, dtype=torch.float32)
              if e is not None else None)

        self.net_ = _MLP(Xs.shape[1] + 1, self.hidden)
        params = list(self.net_.parameters())
        self.g_a_ = nn.Parameter(torch.zeros(1))
        self.g_b_ = nn.Parameter(torch.zeros(1))
        use_rate = (self.lambda_rate > 0) and (es is not None)
        if use_rate:
            params += [self.g_a_, self.g_b_]
        opt = torch.optim.Adam(params, lr=self.lr)

        use_mono = self.lambda_mono > 0
        use_smooth = self.lambda_smooth > 0
        use_bound = self.lambda_boundary > 0
        need_d1 = use_mono or use_smooth or use_rate
        i0 = int(np.argmin(ts.numpy())) if use_bound else -1

        self.net_.train()
        for ep in range(int(self.epochs)):
            opt.zero_grad()
            if use_smooth:
                f, df_dt, d2f_dt2 = PL.compute_d2vbdt2(self.net_, Xs, ts)
            elif need_d1:
                f, df_dt, _ = PL.compute_dvbdt(self.net_, Xs, ts)
                d2f_dt2 = None
            else:
                inp = torch.cat([Xs, ts.unsqueeze(1)], dim=1)
                f = self.net_(inp)
                df_dt = d2f_dt2 = None

            loss = PL.data_loss(ys, f)
            if use_mono:
                loss = loss + self.lambda_mono * PL.monotonicity_loss(df_dt)
            if use_smooth:
                loss = loss + self.lambda_smooth * PL.smoothness_loss(d2f_dt2)
            if use_rate:
                loss = loss + self.lambda_rate * PL.wear_rate_loss(
                    df_dt, es, self.g_a_, self.g_b_)
            if use_bound:
                loss = loss + self.lambda_boundary * PL.boundary_loss(
                    f[i0], ys[i0])

            loss.backward()
            opt.step()
            if self.verbose and (ep % 500 == 0 or ep == self.epochs - 1):
                print(f"  [PINN] epoch {ep:4d}  loss={loss.item():.4f}")
        return self

    # ---- prediction --------------------------------------------------------
    def predict(self, X, t) -> np.ndarray:
        Xs = torch.tensor(self._scale_X(X), dtype=torch.float32)
        ts = torch.tensor(self._scale_t(t), dtype=torch.float32)
        self.net_.eval()
        with torch.no_grad():
            inp = torch.cat([Xs, ts.unsqueeze(1)], dim=1)
            f = self.net_(inp).cpu().numpy()
        return f * self.y_std_ + self.y_mean_

    def predict_zero_x(self, n: int, t) -> np.ndarray:
        """f(0, t): x = ceros en espacio escalado (= vector de medias del
        train). Mide cuanto predice la red ignorando la senal."""
        X0 = np.tile(self.x_mean_, (n, 1))
        return self.predict(X0, t)

    def wear_rate(self, X, t) -> np.ndarray:
        """df/dt in original units (µm per unit t). Diagnostic."""
        Xs = torch.tensor(self._scale_X(X), dtype=torch.float32)
        ts = torch.tensor(self._scale_t(t), dtype=torch.float32)
        self.net_.eval()
        _f, df_dt, _ = PL.compute_dvbdt(self.net_, Xs, ts, create_graph=False)
        return df_dt.detach().cpu().numpy() * self.y_std_ / self.t_rng_

    def wear_acceleration(self, X, t) -> np.ndarray:
        """d2f/dt2 in original units. Diagnostic (smoothness)."""
        Xs = torch.tensor(self._scale_X(X), dtype=torch.float32)
        ts = torch.tensor(self._scale_t(t), dtype=torch.float32)
        self.net_.eval()
        _f, _d1, d2 = PL.compute_d2vbdt2(self.net_, Xs, ts, create_graph=False)
        return d2.detach().cpu().numpy() * self.y_std_ / (self.t_rng_ ** 2)


# =============================================================================
# Degeneracy diagnostic (temporal dominance)
# =============================================================================
def degeneracy_report(model: PINNRegressor, X, t, y,
                      seed: int = RANDOM_SEED,
                      rel_tol: float = 0.15, abs_tol_um: float = 2.0) -> dict:
    """Mide si una PINN ENTRENADA esta dominada por t.

    Compara, sobre el mismo (X, t, y):
      - MAE original   : f(x, t)
      - MAE shuffled_x : f(shuffle(x), t)   (x permutado entre filas, seed fijo)
      - MAE zero_x     : f(0, t)            (x = media del train, sin senal)

    Regla: si shuffled_x o zero_x rinde ~igual que el original
    (delta relativo < rel_tol o delta absoluto < abs_tol_um), la red predice
    desde t y la senal x es decorativa -> temporal_dominance_flag = True.
    """
    X = np.asarray(X, float)
    t = np.asarray(t, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))

    mae = lambda p: float(np.mean(np.abs(np.asarray(p, float) - y)))
    mae_orig = mae(model.predict(X, t))
    mae_shuf = mae(model.predict(X[perm], t))
    mae_zero = mae(model.predict_zero_x(len(X), t))

    best_degraded = min(mae_shuf, mae_zero)
    delta = best_degraded - mae_orig
    flag = bool(delta < abs_tol_um or
                (mae_orig > 0 and delta / mae_orig < rel_tol))
    return {
        "degeneracy_original_mae": mae_orig,
        "degeneracy_shuffled_x_mae": mae_shuf,
        "degeneracy_zero_x_mae": mae_zero,
        "delta_original_vs_zero_x": mae_zero - mae_orig,
        "temporal_dominance_flag": flag,
    }


# =============================================================================
# LOEO evaluation environment for the PINN
# =============================================================================
def resolve_driver_col(columns, configured: str = PHYSICS_DRIVER_COL,
                       candidates=None):
    """Resuelve la columna driver de energia rotacional (C2).

    Devuelve el nombre o None. Orden: `configured` -> `candidates` ->
    patron (`r_*energy*total*`). None => el termino de tasa no puede correr.
    """
    cols = list(columns)
    cands = list(candidates) if candidates is not None else list(PHYSICS_DRIVER_CANDIDATES)
    if configured in cols:
        return configured
    for c in cands:
        if c in cols:
            return c
    for c in cols:
        cl = str(c).lower()
        if cl.startswith("r_") and "energy" in cl and ("total" in cl or "sum" in cl):
            return c
    return None


def loeo_evaluate_pinn(df, feat_cols, t_col: str = EXP_ORDER_COL,
                       driver_col: str = PHYSICS_DRIVER_COL,
                       target: str = TARGET_COLUMN, **pinn_kwargs):
    """Leave-One-Experiment-Out evaluation of the PINN.

    Trains a fresh PINN on nine experiments and predicts the held-out one,
    passing `t` and the rotational-energy driver from the TRAIN fold only.
    Returns (metrics, predictions_df) where metrics are pooled over the ten
    out-of-fold predictions, plus a monotonicity-violation count on the
    out-of-fold wear trajectory.
    """
    import pandas as pd
    from .splitting import loeo_iter
    from .evaluation import compute_metrics

    driver = resolve_driver_col(df.columns, driver_col)
    lam_rate = float(pinn_kwargs.get("lambda_rate", 0.0) or 0.0)
    if driver is None and lam_rate > 0:
        msg = (f"[PINN/C2] lambda_rate={lam_rate} pero NO se encontro columna driver de "
               f"energia rotacional (config: {driver_col!r}); el termino de TASA queda "
               f"DESACTIVADO. Revisa PHYSICS_DRIVER_CANDIDATES o el esquema del dataset.")
        warnings.warn(msg, stacklevel=2)
        print("!!! " + msg)
    eids, y_true, y_pred = [], [], []
    for tr, te in loeo_iter(df, group_col=EXPERIMENT_ID_COL):
        e_tr = tr[driver].values if driver is not None else None
        pinn = PINNRegressor(**pinn_kwargs)
        pinn.fit(tr[feat_cols].values, tr[t_col].values, tr[target].values, e_rot=e_tr)
        p = pinn.predict(te[feat_cols].values, te[t_col].values)
        eids += te[EXPERIMENT_ID_COL].astype(int).tolist()
        y_true += te[target].astype(float).tolist()
        y_pred += np.asarray(p, float).tolist()

    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mets = compute_metrics(y_true, y_pred)

    # monotonicity of the out-of-fold trajectory (ordered by t)
    order = df.set_index(EXPERIMENT_ID_COL).loc[eids, t_col].values
    seq = y_pred[np.argsort(order)]
    mets["mono_violations"] = int(np.sum(np.diff(seq) < -1e-6))

    pred_df = pd.DataFrame({
        "experiment_id": eids, "VB_real": y_true, "VB_pred": y_pred,
        "abs_error": np.abs(y_pred - y_true),
    })
    return mets, pred_df
