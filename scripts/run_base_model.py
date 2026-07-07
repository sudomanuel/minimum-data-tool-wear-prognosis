"""
run_base_model.py — the SIMPLEST possible flow (no augmentation, no feature selection, no physics,
no few-shot): take the raw features, apply standard regressors + a neural network, predict VB directly,
and measure. This is the naive ML/NN baseline, the honest contrast to the full physics+few-shot pipeline.

Protocol: leave-one-tool-out; standardize on train; predict the held-out tool's VB from its features.
Models: Ridge, RandomForest, MLP (the "network"). Output: results/base_model.csv + console.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


def main():
    f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
    meta = ["experiment_id", "tool_id", "within_tool_order", "vb_um", "vc", "fz", "cooling"]
    feats = [c for c in f.columns if c not in meta]
    f = f[f.vb_um <= 300].copy()                      # wear regime, same as the rest of the study
    X = f[feats].to_numpy(float); y = f.vb_um.to_numpy(float); g = f.tool_id.to_numpy()
    logo = LeaveOneGroupOut()

    models = {
        "Ridge (linear)": lambda: RidgeCV(alphas=[0.1, 1, 10, 100, 1000]),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
        "MLP neural net": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                               random_state=0, early_stopping=True),
    }
    print(f"BASE MODEL — simplest flow (features -> model, no augmentation/selection/physics/few-shot)")
    print(f"{len(f)} experiments, {len(feats)} features, LOTO over {f.tool_id.nunique()} tools.\n")
    print(f"{'model':18} {'MAE µm':>8} {'RMSE µm':>8} {'R²':>7}")
    recs = []
    for name, mk in models.items():
        pred = np.zeros(len(y))
        for tr, te in logo.split(X, y, g):
            sc = StandardScaler().fit(X[tr])
            m = mk().fit(sc.transform(X[tr]), y[tr])
            pred[te] = m.predict(sc.transform(X[te]))
        mae = np.mean(np.abs(pred - y)); rmse = np.sqrt(np.mean((pred - y) ** 2))
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"{name:18} {mae:8.1f} {rmse:8.1f} {r2:7.2f}")
        recs.append(dict(model=name, MAE_um=round(mae, 1), RMSE_um=round(rmse, 1), R2=round(r2, 2)))
    pd.DataFrame(recs).to_csv(os.path.join(ROOT, "results", "base_model.csv"), index=False)
    best = max(recs, key=lambda r: r["R2"])
    print(f"\nVERDICT: best base model R²={best['R2']} ({best['model']}). "
          + ("All R²<=0 -> naive feature->VB does NOT generalize across tools; "
             "motivates the physics + few-shot pipeline (R²=0.52-0.67)." if best["R2"] <= 0.05
             else "a base model generalizes; revisit."))
    print("wrote results/base_model.csv")


if __name__ == "__main__":
    main()
