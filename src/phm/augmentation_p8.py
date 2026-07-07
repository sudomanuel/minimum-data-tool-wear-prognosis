"""
augmentation_p8.py — fold-safe data augmentation for scarce single-tool wear (P8.8).

Array-based, designed to run in the SCALED feature space (unit std) inside a single
LOEO fold. Augmented samples are built ONLY from the training rows of that fold; the
held-out experiment is never used and never augmented.

Two physically-conservative methods:
  - jitter:       copy a real train sample + small Gaussian noise (measurement noise
                  around the same wear level); the label VB is kept unchanged.
  - interpolate:  blend two ADJACENT-in-order real train samples; since VB is monotone
                  in experiment order, the blended VB stays between two real values
                  (physics-safe, no new extrapolation).

Regimes (the evaluated R0..R3 ladder):
  R0 = none | R1 = jitter | R2 = interpolate | R3 = jitter + interpolate

Separate from the legacy DataFrame-level `augmentation.py` (P1-P6); does not modify it.
"""
import numpy as np

REGIMES = ("R0", "R1", "R2", "R3")
REGIME_LABEL = {"R0": "ninguna", "R1": "jitter", "R2": "interpolacion", "R3": "jitter+interpolacion"}


def jitter(X, y, n_aug, sigma=0.05, rng=None):
    rng = np.random.default_rng(rng)
    if n_aug <= 0 or len(X) == 0:
        return np.empty((0, X.shape[1])), np.empty((0,))
    idx = rng.integers(0, len(X), size=n_aug)
    Xa = X[idx] + rng.normal(0.0, 1.0, size=(n_aug, X.shape[1])) * sigma
    ya = y[idx].copy()                      # label preserved (noise around same wear)
    return Xa, ya


def interpolate(X, y, order, n_aug, rng=None):
    rng = np.random.default_rng(rng)
    if n_aug <= 0 or len(X) < 2:
        return np.empty((0, X.shape[1])), np.empty((0,))
    o = np.argsort(np.asarray(order, float))
    Xs, ys = X[o], np.asarray(y, float)[o]
    p = rng.integers(0, len(Xs) - 1, size=n_aug)        # adjacent pair start index
    lam = rng.uniform(0.2, 0.8, size=n_aug)
    Xa = (1 - lam)[:, None] * Xs[p] + lam[:, None] * Xs[p + 1]
    ya = (1 - lam) * ys[p] + lam * ys[p + 1]            # VB between neighbors -> monotone-safe
    return Xa, ya


def augment(X, y, order, regime, n_aug, sigma=0.05, rng=None):
    """Return (X_aug, y_aug) = real train + synthetic, per the regime. Train-only inputs."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if regime == "R0":
        return X, y
    parts_X, parts_y = [X], [y]
    if regime in ("R1", "R3"):
        Xa, ya = jitter(X, y, n_aug, sigma=sigma, rng=rng)
        parts_X.append(Xa); parts_y.append(ya)
    if regime in ("R2", "R3"):
        Xa, ya = interpolate(X, y, order, n_aug, rng=rng)
        parts_X.append(Xa); parts_y.append(ya)
    return np.vstack(parts_X), np.concatenate(parts_y)


def physics_ok(y_real, y_aug):
    """Augmented labels must stay within the observed wear range (no new extrapolation)."""
    lo, hi = float(np.min(y_real)), float(np.max(y_real))
    return bool(np.all(y_aug >= lo - 1e-9) and np.all(y_aug <= hi + 1e-9))
