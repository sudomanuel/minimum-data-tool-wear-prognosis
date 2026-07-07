"""
wear_models.py — physics-integrated, monotone-by-construction wear-curve laws.

Each law is VB(t) built so that VB is monotone non-decreasing for t>=0 BY CONSTRUCTION
(positive coefficients via softplus; positive powers; saturating/linear/accelerated terms).
These are the closed-form rung of the equation ladder (S1/S2/S4); the neural global law
g_theta(x,t,c) and the per-tool few-shot adapter are added later (Step 3). No penalty is
used — the constraint cannot be violated.

Foundations: three-stage flank-wear curve (ISO 3685; Kalpakjian 2014); power/run-in wear.
"""
import numpy as np
from scipy.optimize import least_squares


def softplus(z):
    z = np.asarray(z, float)
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)   # numerically stable, >0


def _S1(p, t):                       # linear (steady-state), 2 params
    VB0, a = p
    return VB0 + softplus(a) * t


def _S2(p, t):                       # power / run-in (Archard-type), 3 params
    VB0, k, pw = p
    return VB0 + softplus(k) * np.power(np.maximum(t, 0.0), softplus(pw))


def _S4(p, t):                       # three-stage: run-in + steady + accelerated, 6 params
    VB0, a, tau, b, c, ts = p
    runin = softplus(a) * (1.0 - np.exp(-np.maximum(t, 0.0) / softplus(tau)))
    steady = softplus(b) * t
    accel = softplus(c) * np.power(softplus(t - ts), 2)
    return VB0 + runin + steady + accel


MODELS = {"S1": (_S1, 2), "S2": (_S2, 3), "S4": (_S4, 6)}


def _p0(name, y):
    if name == "S1":
        return [y[0], 0.0]
    if name == "S2":
        return [y[0], 0.0, 0.0]
    return [y[0], 0.0, 1.0, 0.0, -2.0, 1.0]    # S4


def fit(name, t, y):
    """Fit a law to (t, y). Returns (params, t0) where t is internally shifted to start at 0."""
    fn, _ = MODELS[name]
    t = np.asarray(t, float); y = np.asarray(y, float)
    t0 = float(t.min())
    ts = t - t0
    res = least_squares(lambda p: fn(p, ts) - y, _p0(name, y), max_nfev=10000)
    return res.x, t0


def predict(name, params, t0, t):
    fn, _ = MODELS[name]
    return fn(params, np.asarray(t, float) - t0)


def monotonic_violations(name, params, t0, t_grid):
    """Count downward steps on a dense grid (should be 0 by construction)."""
    g = np.linspace(np.min(t_grid), np.max(t_grid), 200)
    vb = predict(name, params, t0, g)
    return int(np.sum(np.diff(vb) < -1e-6))
