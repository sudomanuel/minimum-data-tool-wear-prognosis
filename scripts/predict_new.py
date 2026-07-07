#!/usr/bin/env python3
"""
predict_new.py — run the exported wear model on a NEW feature CSV.

Loads models/wear_model.joblib, predicts VB for each row of the input CSV, and (optionally)
derives the Health Index at a failure threshold. Writes an output CSV and prints a summary.

The input CSV must contain the feature columns listed in models/wear_model.json. Extra
columns are ignored; an `experiment_id`/`physical_experiment_order` column, if present, is
carried through to the output.

Uso:
  python scripts/predict_new.py --input my_features.csv --vb-failure 300 --output preds.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "src"))
from phm.model_export import WearModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=WT / "models" / "wear_model.joblib")
    ap.add_argument("--input", type=Path, required=True, help="CSV with the model's feature columns")
    ap.add_argument("--output", type=Path, default=WT / "results" / "predictions_new.csv")
    ap.add_argument("--vb-failure", type=float, default=300.0, help="failure threshold for HI (µm)")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}  (run: python run.py export-model)")
    model = WearModel.load(args.model)
    df = pd.read_csv(args.input)

    vb = model.predict_vb(df)                         # raises clearly if columns are missing
    h = model.health_index(vb, vb_failure=args.vb_failure)
    out = pd.DataFrame({"VB_pred_um": np.round(vb, 2),
                        "HI": np.round(h["HI"], 3), "DI": np.round(h["DI"], 3)})
    for c in ("experiment_id", "physical_experiment_order", "tool_id"):
        if c in df.columns:
            out.insert(0, c, df[c].values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"model: {args.model.name}  | features expected: {len(model.feature_cols)}")
    print(f"rows predicted: {len(out)}  | VB range: {vb.min():.1f}–{vb.max():.1f} µm  "
          f"| threshold {args.vb_failure:.0f} µm")
    print(f"wrote {args.output}")
    print(out.head(12).to_string(index=False))
    if model.vb_observed_max < args.vb_failure:
        print(f"\nNOTE: VB_failure {args.vb_failure:.0f} > observed max {model.vb_observed_max:.0f} "
              f"-> any RUL would be EXTRAPOLATED (not validated).")


if __name__ == "__main__":
    main()
