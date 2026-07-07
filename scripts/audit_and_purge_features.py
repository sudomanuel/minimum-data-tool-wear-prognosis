"""
audit_and_purge_features.py — multi-level feature audit + purge for the minimum-data pipeline.

Multi-profile review encoded as deterministic levels:
  L1 (engineer)      : drop constant / near-zero-variance columns.
  L2 (data scientist): drop exact-duplicate columns and one of each |corr|>=THR redundant pair
                       (A/R channel duplication is the main offender; keep the higher-variance rep).
  L3 (critic)        : flag features with negligible univariate association to VB (|Kendall| < TAU)
                       AND never useful -> documented as non-predictive (kept only if not redundant).
  L4 (auditor)       : every drop carries a reason; nothing is removed silently; raw file untouched.

Honesty note: ALL sensor features are out-of-sample non-predictive (proven 4 ways). The purge removes
REDUNDANCY and zero-variance, producing a lean, documented feature set; it does not claim the survivors
predict per-tool deviation. Raw data/input/derived/features_experiment.csv is the frozen source; this
writes a CURATED view + an auditable ledger.

Outputs: results/feature_audit.csv (per-feature verdict), data/input/derived/features_curated.csv.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META = ["experiment_id", "tool_id", "within_tool_order", "vb_um", "vc", "fz", "cooling"]
CORR_THR = 0.98     # redundancy threshold
TAU = 0.10          # univariate association floor (|Kendall tau| with VB)
NZV = 1e-9          # near-zero variance floor


def main():
    f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
    feat = [c for c in f.columns if c not in META]
    X = f[feat].astype(float)
    y = f["vb_um"].astype(float).to_numpy()
    n0 = len(feat)
    verdict = {c: {"feature": c, "drop": False, "reason": "", "cluster": "",
                   "kendall_abs": np.nan, "variance": float(X[c].var())} for c in feat}

    # L1 — constant / near-zero variance
    for c in feat:
        if X[c].var() <= NZV:
            verdict[c].update(drop=True, reason="L1_zero_variance")

    alive = [c for c in feat if not verdict[c]["drop"]]

    # L2 — exact duplicates + high-correlation redundancy (keep higher-variance representative)
    # exact duplicates
    seen = {}
    for c in alive:
        key = tuple(np.round(X[c].to_numpy(), 6))
        if key in seen:
            verdict[c].update(drop=True, reason="L2_duplicate_of:" + seen[key], cluster=seen[key])
        else:
            seen[key] = c
    alive = [c for c in alive if not verdict[c]["drop"]]
    # high-corr clusters via greedy pass on the correlation matrix
    corr = X[alive].corr().abs().fillna(0.0)
    kept = []
    order = sorted(alive, key=lambda c: -X[c].var())   # prefer high-variance reps
    for c in order:
        red = next((k for k in kept if corr.loc[c, k] >= CORR_THR), None)
        if red is not None:
            verdict[c].update(drop=True, reason=f"L2_corr>={CORR_THR}_with:" + red, cluster=red)
        else:
            kept.append(c)
    alive = kept

    # L3 — univariate association (flag, do not auto-drop survivors that are non-redundant)
    for c in feat:
        if not verdict[c]["drop"]:
            tau = kendalltau(X[c], y, nan_policy="omit").statistic
            verdict[c]["kendall_abs"] = round(abs(float(tau)) if tau == tau else 0.0, 3)
            if verdict[c]["kendall_abs"] < TAU:
                verdict[c]["reason"] = (verdict[c]["reason"] + ";" if verdict[c]["reason"] else "") \
                    + f"L3_low_assoc(|tau|<{TAU})"

    led = pd.DataFrame(verdict.values()).sort_values(["drop", "feature"])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    led.to_csv(os.path.join(ROOT, "results", "feature_audit.csv"), index=False)

    curated = [c for c in feat if not verdict[c]["drop"]]
    out = f[META + curated]
    out.to_csv(os.path.join(ROOT, "data", "input", "derived", "features_curated.csv"), index=False)

    dropped = n0 - len(curated)
    zv = sum(1 for c in feat if verdict[c]["reason"].startswith("L1"))
    dup = sum(1 for c in feat if "L2_duplicate" in verdict[c]["reason"])
    cor = sum(1 for c in feat if "L2_corr" in verdict[c]["reason"])
    low = sum(1 for c in feat if "L3_low_assoc" in verdict[c]["reason"] and not verdict[c]["drop"])
    print(f"Feature audit: {n0} -> {len(curated)} kept ({dropped} purged)")
    print(f"  L1 zero-variance : {zv}")
    print(f"  L2 duplicates    : {dup}")
    print(f"  L2 corr>={CORR_THR}    : {cor}")
    print(f"  L3 low-assoc flag (kept, documented non-predictive): {low}")
    print(f"\nwrote results/feature_audit.csv + data/input/derived/features_curated.csv")


if __name__ == "__main__":
    main()
