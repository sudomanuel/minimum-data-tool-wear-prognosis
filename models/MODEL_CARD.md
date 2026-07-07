# Model card — deployable wear model (`wear_model.joblib`)

**Predicts:** flank wear `VB` (µm) from vibration features (reliability-aware sensor set).
**Trained on:** 10 official T01 experiments (VB 103–212 µm).
**Estimator:** RandomForestRegressor on 10 selected features.
**Feature contract:** see `wear_model.json` (the new data must contain these columns).

## Honest expected performance (LOEO on T01)
- MAE ≈ **30.1 µm**, R² ≈ 0.11  (sensor-only; this is the realistic number on new data).
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
- RUL is **extrapolated** (observed VB_max 212 < typical thresholds 220–600 µm).
- Provenance: schema 1.0, selection = reliability_aware + Kendall/Spearman/MMI top-k.
