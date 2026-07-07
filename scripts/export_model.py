#!/usr/bin/env python3
"""
export_model.py — train the deployable wear model on all official data and save it.

Produces a single portable artifact (models/wear_model.joblib + .json feature contract)
and a MODEL_CARD.md with the honest expected error and usage. The exported model is the
reliability-aware SENSOR model (see model_export.py design note: order/t does not transfer
to new tools, so the deployable model is sensor-based).

Uso:  python run.py export-model
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "src"))
from phm.model_export import WearModel  # noqa: E402
from phm.feature_selection_p8 import reliability_aware_cols, select_topk  # noqa: E402
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

FEAT = WT / "data" / "features" / "p8_2_features_experiment_full_contact.csv"
MODELS = WT / "models"


def loeo_honest_mae(df, topk=10, seed=0):
    """Honest expected error of this exact config under LOEO (for the model card)."""
    y = df["VB_um"].to_numpy(float)
    pool = [c for c in reliability_aware_cols(df) if c != "physical_experiment_order"]
    yp = np.zeros(len(df))
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        sel, _ = select_topk(df.iloc[tr], y[tr], pool, k=topk, seed=seed)
        Xtr = df.iloc[tr][sel].to_numpy(float)
        Xte = df.iloc[[i]][sel].to_numpy(float)
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        sc = StandardScaler().fit(Xtr)
        m = RandomForestRegressor(n_estimators=300, random_state=seed).fit(sc.transform(Xtr), y[tr])
        yp[i] = m.predict(sc.transform(Xte))[0]
    mae = float(np.mean(np.abs(y - yp)))
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((y - yp) ** 2) / ss) if ss > 0 else float("nan")
    return mae, r2


def main():
    t0 = time.time()
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEAT).sort_values("physical_experiment_order").reset_index(drop=True)

    model = WearModel.fit_from_dataframe(df, target="VB_um", topk=10, seed=0)
    art = model.save(MODELS / "wear_model.joblib")
    mae, r2 = loeo_honest_mae(df)

    card = f"""# Model card — deployable wear model (`wear_model.joblib`)

**Predicts:** flank wear `VB` (µm) from vibration features (reliability-aware sensor set).
**Trained on:** {model.meta['n_train']} official T01 experiments (VB {model.vb_0:.0f}–{model.vb_observed_max:.0f} µm).
**Estimator:** {model.meta['estimator']} on {len(model.feature_cols)} selected features.
**Feature contract:** see `wear_model.json` (the new data must contain these columns).

## Honest expected performance (LOEO on T01)
- MAE ≈ **{mae:.1f} µm**, R² ≈ {r2:.2f}  (sensor-only; this is the realistic number on new data).
- NOTE: the 6.65 µm headline (PINN_mono) is **temporal-control-level** and needs the SAME tool's
  experiment order; it does NOT transfer to a new tool (leakage audit P8.11). For new tools, expect
  the sensor-only figure above.

## How to use on new data
```bash
python run.py export-model                 # (re)train + save this artifact
python scripts/predict_new.py --input my_features.csv --vb-failure 300 --output preds.csv
```
`my_features.csv` must contain the feature columns listed in `wear_model.json`.

## Caveats (read before trusting a number)
- Single tool (T01): generalization to new tools is **not yet validated** (LOTO pending).
- RUL is **extrapolated** (observed VB_max {model.vb_observed_max:.0f} < typical thresholds 220–600 µm).
- Provenance: schema {model.meta['schema_version']}, selection = {model.meta['selection']}.
"""
    (MODELS / "MODEL_CARD.md").write_text(card, encoding="utf-8")

    print(f"saved {art}  (+ .json contract, MODEL_CARD.md)")
    print(f"features ({len(model.feature_cols)}): {model.feature_cols}")
    print(f"honest LOEO: MAE={mae:.2f} R2={r2:.2f}  vb_0={model.vb_0:.0f} vb_max={model.vb_observed_max:.0f}")
    print(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
