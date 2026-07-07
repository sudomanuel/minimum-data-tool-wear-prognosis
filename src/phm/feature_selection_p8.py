"""
feature_selection_p8.py — P8.3 Kendall/Spearman/MMI feature selection (fold-safe).

Score:  S_j = w1*|tau_j| + w2*|rho_j| + w3*MMI_norm_j   (default w = 1/3 each).
- tau = Kendall, rho = Spearman with VB; MMI = mutual_info_regression, min-max normalized
  within the candidate set so it is comparable to the [0,1] correlations.
- Redundancy filter: greedy, drop a feature if |corr| > thr with any already-selected feature.

Anti-leakage: callers pass TRAIN-ONLY rows for fold-wise selection. The global call is
exploratory only and must be tagged exploratory_only=True.
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.feature_selection import mutual_info_regression

TIME_LIKE = re.compile(r"cumulative|experiment_order|recorded_signal_order|contact_count|life_fraction")
# reliability-aware exclusions (B3): energy aggregates (exp77 4-contact bias, P7.1) + time-like.
RELIABILITY_EXCLUDE = re.compile(r"cumulative|experiment_order|recorded_signal_order|"
                                 r"contact_count|life_fraction|energy_mean|energy_std|"
                                 r"energy_median|combined_energy")
META = {"experiment_id", "physical_experiment_order", "recorded_signal_order", "VB_um",
        "contact_count_valid", "missing_contact_count", "feature_reliability",
        "energy_total_reliability", "rms_mean_reliability", "segmentation_source"}

# B4 PINN-ready minimal robust set (per channel): rms/waveform/dominant-freq/wavelet-entropy/energy.
PINN_READY = ["A_rms_mean", "R_rms_mean", "A_waveform_length_mean", "R_waveform_length_mean",
              "A_dominant_freq_mean", "R_dominant_freq_mean", "A_wavelet_entropy_mean",
              "R_wavelet_entropy_mean", "A_energy_mean", "R_energy_mean"]


def all_feature_cols(df):
    return [c for c in df.columns if c not in META]


def sensor_cols(df):
    return [c for c in all_feature_cols(df) if not TIME_LIKE.search(c)]


def reliability_aware_cols(df):
    return [c for c in all_feature_cols(df) if not RELIABILITY_EXCLUDE.search(c)]


def score_features(X: pd.DataFrame, y: np.ndarray, cols,
                   w=(1 / 3, 1 / 3, 1 / 3), seed=0) -> pd.DataFrame:
    """Kendall/Spearman/MMI score for each column. Returns sorted DataFrame."""
    y = np.asarray(y, float)
    rows = []
    mmi_raw = {}
    valid = [c for c in cols if X[c].nunique() > 1]
    if valid:
        mi = mutual_info_regression(X[valid].to_numpy(float), y, random_state=seed)
        mmi_raw = dict(zip(valid, mi))
    mmi_max = max(mmi_raw.values()) if mmi_raw else 1.0
    for c in cols:
        x = X[c].to_numpy(float)
        if np.nanstd(x) == 0:
            tau = rho = mmi = 0.0
        else:
            tau = abs(kendalltau(x, y).statistic)
            rho = abs(spearmanr(x, y).statistic)
            tau = 0.0 if np.isnan(tau) else tau
            rho = 0.0 if np.isnan(rho) else rho
            mmi = (mmi_raw.get(c, 0.0) / mmi_max) if mmi_max > 0 else 0.0
        S = w[0] * tau + w[1] * rho + w[2] * mmi
        rows.append({"feature": c, "kendall_abs": round(tau, 4), "spearman_abs": round(rho, 4),
                     "mmi_norm": round(mmi, 4), "score": round(S, 4)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def redundancy_filter(X: pd.DataFrame, ranked_feats, thr=0.95):
    """Greedy: keep top-ranked features dropping any with |corr|>thr to an already-kept one."""
    kept = []
    for f in ranked_feats:
        if not kept:
            kept.append(f)
            continue
        xf = X[f].to_numpy(float)
        red = False
        for k in kept:
            xk = X[k].to_numpy(float)
            if np.nanstd(xf) == 0 or np.nanstd(xk) == 0:
                continue
            c = abs(np.corrcoef(xf, xk)[0, 1])
            if not np.isnan(c) and c > thr:
                red = True
                break
        if not red:
            kept.append(f)
    return kept


def select_topk(X: pd.DataFrame, y, cols, k=10, w=(1 / 3, 1 / 3, 1 / 3),
                redundancy_thr=0.95, seed=0):
    """TRAIN-ONLY selection: score -> redundancy filter -> top-k."""
    sc = score_features(X, y, cols, w=w, seed=seed)
    nonred = redundancy_filter(X, sc.feature.tolist(), thr=redundancy_thr)
    return nonred[:k], sc


def shap_scores(X: pd.DataFrame, y, cols, seed=0, n_estimators=200) -> dict:
    """TRAIN-ONLY SHAP importance per feature (P8.7).

    Fits a RandomForest on the TRAIN rows only and returns normalized mean|SHAP|
    in [0,1] per feature (TreeExplainer; falls back to impurity importance if SHAP
    is unavailable). Anti-leakage: callers must pass train-only rows. SHAP becomes
    the 4th consensus vote alongside Kendall/Spearman/MMI.
    """
    from sklearn.ensemble import RandomForestRegressor
    y = np.asarray(y, float)
    valid = [c for c in cols if X[c].nunique() > 1]
    out = {c: 0.0 for c in cols}
    if not valid:
        return out
    Xv = X[valid].to_numpy(float)
    med = np.nanmedian(Xv, axis=0)
    Xv = np.where(np.isnan(Xv), med, Xv)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed).fit(Xv, y)
    try:
        import shap
        sv = shap.TreeExplainer(rf).shap_values(Xv, check_additivity=False)
        imp = np.abs(np.asarray(sv)).mean(axis=0)
    except Exception:
        imp = np.asarray(rf.feature_importances_, float)
    mx = float(imp.max()) if imp.size and imp.max() > 0 else 1.0
    for c, v in zip(valid, imp):
        out[c] = float(v / mx)
    return out
