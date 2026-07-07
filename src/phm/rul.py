"""
rul.py — Health Index (HI), Degradation Index (DI) and RUL by threshold crossing (#2).

Turns a predicted wear curve VB_hat(t) into an auditable PHM output:
    VB_hat(t)  ->  HI(t)  ->  DI(t)  ->  t_failure  ->  RUL(t)
Honest separation of OBSERVED range vs EXTRAPOLATION. New module; does NOT modify the legacy
P4 `scripts/derive_rul.py` / `config/physics.yaml`. Thresholds in `config/rul_thresholds.yaml`.

Definitions (per project spec):
    HI(t) = clip( (VB_failure - VB_hat(t)) / (VB_failure - VB_0), 0, 1 )
    DI(t) = 1 - HI(t)
    t_failure = min{ t : VB_hat(t) >= VB_failure }
    RUL(t)    = t_failure - t
`rul_extrapolated = True` when the crossing is beyond the observed range OR VB_max_obs < threshold.
"""
import numpy as np


def health_index(vb_hat, vb_failure, vb_0):
    vb_hat = np.asarray(vb_hat, float)
    denom = (vb_failure - vb_0)
    if abs(denom) < 1e-9:
        return np.full_like(vb_hat, np.nan)
    return np.clip((vb_failure - vb_hat) / denom, 0.0, 1.0)


def degradation_index(vb_hat, vb_failure, vb_0):
    return 1.0 - health_index(vb_hat, vb_failure, vb_0)


def threshold_crossing(t_grid, vb_curve, vb_failure):
    """First t where vb_curve >= vb_failure; None if it never reaches it on the grid."""
    t_grid = np.asarray(t_grid, float)
    vb = np.asarray(vb_curve, float)
    hit = np.where(vb >= vb_failure)[0]
    if len(hit) == 0:
        return None
    return float(t_grid[hit[0]])


def derive_rul(t_grid, vb_curve, vb_failure, vb_0, t_last_observed, vb_max_observed):
    """Return a dict with t_failure, RUL at last observation, and honest extrapolation flags.

    `crossing_within_horizon` = the curve reaches the threshold somewhere on the evaluated
    extrapolation horizon (t_grid). `rul_extrapolated` = the crossing (if any) lies beyond the
    OBSERVED range or the threshold exceeds VB_max_observed (so it is not experimentally validated).
    """
    t_failure = threshold_crossing(t_grid, vb_curve, vb_failure)
    crossing_within_horizon = t_failure is not None
    extrapolated = (
        (vb_max_observed < vb_failure)            # threshold above anything observed
        or (not crossing_within_horizon)          # not reached within the evaluated horizon
        or (t_failure > t_last_observed)          # crossing beyond observed range
    )
    return {
        "vb_failure_um": vb_failure,
        "vb_0_um": vb_0,
        "vb_max_observed": round(float(vb_max_observed), 2),
        "t_failure": round(t_failure, 3) if crossing_within_horizon
        else "no_crossing_within_horizon",
        "rul_at_last_obs": round(t_failure - t_last_observed, 3) if crossing_within_horizon
        else "n/a",
        "crossing_within_horizon": bool(crossing_within_horizon),
        "rul_extrapolated": bool(extrapolated),
    }


# ---------------------------------------------------------------------------
# Interval-censored breakage endpoint (paper: "Failure Definition under
# Pre-Breakage VB Measurements"). The tool breaks AFTER the last measured VB,
# so the failure order is known only to lie in (T_L, T_R]:
#   T_L = experiment-order of the last measured VB before breakage
#   T_R = experiment-order at which breakage is recorded/first observed
# If T_R is unknown ("only after the last experiment"), the endpoint degrades
# gracefully to RIGHT-censored (one-sided safety check), not interval-censored.
# These are descriptive, single-tool case-study quantities, not calibrated
# generalization metrics.
# ---------------------------------------------------------------------------

def interval_censored_failure(predicted_t_fail, t_left, t_right=None):
    """Score a predicted failure order against the (interval-)censored breakage.

    Returns the report schema columns:
      failure_mode, T_L, T_R, predicted_T_fail, interval_hit,
      distance_to_interval, unsafe_overestimation, conservative_underestimation,
      interval_width.

    Semantics:
      * interval_hit               -> predicted_t_fail in (T_L, T_R]
      * distance_to_interval       -> 0 inside; else signed distance to nearest edge
      * unsafe_overestimation      -> predicted_t_fail > T_R  (maintenance-unsafe)
      * conservative_underestimation -> predicted_t_fail < T_L
    """
    out = {
        "failure_mode": "breakage_interval" if t_right is not None else "right_censored",
        "T_L": round(float(t_left), 3),
        "T_R": (round(float(t_right), 3) if t_right is not None else None),
        "predicted_T_fail": (round(float(predicted_t_fail), 3)
                             if predicted_t_fail is not None else None),
        "interval_hit": None,
        "distance_to_interval": None,
        "unsafe_overestimation": None,
        "conservative_underestimation": None,
        "interval_width": (round(float(t_right) - float(t_left), 3)
                           if t_right is not None else None),
    }
    if predicted_t_fail is None:
        out["failure_mode"] = "no_prediction"
        return out

    th, tl = float(predicted_t_fail), float(t_left)
    if t_right is None:
        # Right-censored: only "failure after T_L" is known; cannot judge overshoot.
        out["conservative_underestimation"] = bool(th < tl)
        out["distance_to_interval"] = round(max(0.0, tl - th), 3)  # one-sided
        return out

    tr = float(t_right)
    inside = (th > tl) and (th <= tr)
    out["interval_hit"] = bool(inside)
    out["distance_to_interval"] = round(max(0.0, tl - th) + max(0.0, th - tr), 3)
    out["unsafe_overestimation"] = bool(th > tr)
    out["conservative_underestimation"] = bool(th < tl)
    return out


def rul_interval_consistency(t_i, predicted_t_fail, t_left, t_right):
    """Per-time RUL interval consistency: RUL_hat(t_i) in (T_L - t_i, T_R - t_i]."""
    rul_hat = float(predicted_t_fail) - float(t_i)
    lo, hi = float(t_left) - float(t_i), float(t_right) - float(t_i)
    return {
        "t_i": round(float(t_i), 3),
        "rul_hat": round(rul_hat, 3),
        "rul_true_interval": (round(lo, 3), round(hi, 3)),
        "consistent": bool(rul_hat > lo and rul_hat <= hi),
    }
