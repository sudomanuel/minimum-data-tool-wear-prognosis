# Dataset schemas — three separate views

Every experiment is tagged in `manifest.csv` so the pipeline never mixes views.
A row can belong to several views at once (e.g. a clean experiment is in both the
sensor and trajectory views); a signal-less experiment is trajectory-only.

## 1. sensor-based dataset
Experiments with a **valid signal** and **enough valid contacts**.
- key: `(tool_id, experiment_id)`
- requires: `has_signal == true` AND `n_valid_contacts >= MIN_CONTACTS` (default 5)
- features: the ~326 multi-domain features (time/freq/wavelet) per channel A,R
- used by: classical sensor models, sensor branches, sensor-driven wear-rate term
- flag: `usable_sensor == true`

## 2. trajectory-based VB dataset
Every experiment with a **measured VB**, even if signal-less.
- key: `(tool_id, within_tool_order)`
- requires: `vb_um` present
- payload: `within_tool_order` (ordinal life coordinate), `vb_um`
- used by: VB(t) curve, PINN integral model, HI derivation, time-aware baselines
- flag: `usable_trajectory == true`

## 3. RUL / breakage / censoring dataset
One record **per tool** (not per experiment), from `failure_events`.
- key: `tool_id`
- payload: `last_measured_order` (T_L), `breakage_observed`, `breakage_order` (T_R or null)
- censoring:
  - `breakage_order` present  → **interval-censored** `T_f ∈ (T_L, T_R]`
  - `breakage_observed` true, `breakage_order` null → **right-censored** `T_f > T_L`
  - `breakage_observed` false → **no endpoint** (trajectory-only tool)
- used by: breakage-anchored RUL evaluation (interval-consistency, distance-to-interval,
  unsafe-overestimation, no-warning rate) — never exact RUL MAE without a real crossing.

## Identity / leakage columns (never features)
`tool_id`, `experiment_id`, `within_tool_order`, file paths, and any process metadata
are excluded from all feature matrices. Tool identity must never leak across a LOTO split.

## Two scarcity axes (must stay separate)
- **K** = number of training tools used to learn the global degradation law (Axis 1).
- **m** = number of early VB labels from the held-out test tool used for few-shot
  adaptation (Axis 2). The first `m` points adapt only tool-specific parameters
  (VB0_j, b_j, alpha_j); all later points are sealed for evaluation.
