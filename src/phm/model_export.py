"""
model_export.py — portable, deployable wear model for NEW data (P8 packaging).

`WearModel` bundles everything needed to predict on a fresh dataset:
feature contract (the exact columns it expects) + train-median imputation + scaler +
estimator, plus the VB -> Health Index -> RUL derivation. Saved as a single joblib
artifact for portability.

DESIGN NOTE (honest, important):
  For a NEW, unseen tool the experiment-order signal does NOT transfer (the leakage
  audit P8.11 showed the PINN_mono headline is t-driven). A deployable model for other
  data must therefore rely on the SENSOR features. So the exported model defaults to the
  reliability-aware sensor RandomForest, whose honest LOEO error on T01 is ~30 µm (the
  axial-only SOLO_A branch reaches ~24 µm) — that is the realistic expectation on new data,
  NOT the 6.65 µm (which is the temporal-control-level number that needs the same tool's order).
  The exact LOEO number for the exported config is written into models/MODEL_CARD.md.

Usage:
    from phm.model_export import WearModel
    m = WearModel.fit_from_dataframe(df, target="VB_um")     # df: features + VB_um
    m.save("models/wear_model.joblib")
    m2 = WearModel.load("models/wear_model.joblib")
    vb = m2.predict_vb(new_features_df)                       # new data, same feature schema
    hi = m2.health_index(vb, vb_failure=300)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"


@dataclass
class WearModel:
    estimator: object
    scaler: object
    feature_cols: list
    train_median: np.ndarray
    vb_0: float
    vb_observed_max: float
    meta: dict = field(default_factory=dict)

    # ---- construction ----
    @classmethod
    def fit_from_dataframe(cls, df: pd.DataFrame, target: str = "VB_um",
                           feature_cols=None, topk: int = 10, estimator=None, seed: int = 0):
        """Fit the deployable model on ALL provided rows (final artifact, not an evaluation).

        feature_cols=None -> reliability-aware sensor pool + fold-free top-k consensus
        selection on the full data (deployable artifact). estimator=None -> RandomForest.
        """
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
        from phm.feature_selection_p8 import reliability_aware_cols, select_topk
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        y = df[target].to_numpy(float)
        if feature_cols is None:
            pool = [c for c in reliability_aware_cols(df) if c != "physical_experiment_order"]
            feature_cols, _ = select_topk(df, y, pool, k=topk, seed=seed)
        X = df[feature_cols].to_numpy(float)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        X = np.where(np.isnan(X), med, X)
        scaler = StandardScaler().fit(X)
        est = estimator if estimator is not None else RandomForestRegressor(
            n_estimators=300, random_state=seed)
        est.fit(scaler.transform(X), y)
        meta = {"schema_version": SCHEMA_VERSION, "n_train": int(len(df)),
                "target": target, "estimator": type(est).__name__,
                "selection": "reliability_aware + Kendall/Spearman/MMI top-k" if feature_cols else "given",
                "note": "sensor-only deployable model; honest LOEO error on T01 ~24 um"}
        return cls(estimator=est, scaler=scaler, feature_cols=list(feature_cols),
                   train_median=med, vb_0=float(np.min(y)), vb_observed_max=float(np.max(y)),
                   meta=meta)

    # ---- prediction ----
    def _matrix(self, data) -> np.ndarray:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"new data is missing {len(missing)} required feature columns: "
                             f"{missing[:8]}{'...' if len(missing) > 8 else ''}")
        X = df[self.feature_cols].to_numpy(float)
        X = np.where(np.isnan(X), self.train_median, X)
        return self.scaler.transform(X)

    def predict_vb(self, data) -> np.ndarray:
        """Predict flank wear VB (µm) on new data with the same feature schema."""
        return np.asarray(self.estimator.predict(self._matrix(data)), float)

    def health_index(self, vb_pred, vb_failure: float):
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
        from phm.rul import health_index, degradation_index
        hi = health_index(vb_pred, vb_failure, self.vb_0)
        return {"HI": hi, "DI": degradation_index(vb_pred, vb_failure, self.vb_0)}

    def derive_rul(self, t_grid, vb_curve, vb_failure: float, t_last_observed: float):
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
        from phm.rul import derive_rul
        return derive_rul(t_grid, vb_curve, vb_failure, self.vb_0,
                          t_last_observed, self.vb_observed_max)

    # ---- persistence ----
    def save(self, path):
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        # human-readable sidecar (feature contract + provenance)
        sidecar = {"schema_version": SCHEMA_VERSION, "feature_cols": self.feature_cols,
                   "vb_0": self.vb_0, "vb_observed_max": self.vb_observed_max, "meta": self.meta}
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        import joblib
        return joblib.load(Path(path))
