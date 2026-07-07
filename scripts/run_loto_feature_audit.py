"""
run_loto_feature_audit.py — leakage-safe LOTO with fold-safe feature selection + SHAP.

Documents, auditably, whether vibration features carry generalizable per-tool signal beyond
the population model. Target = per-tool wear rate (the deviation a model would need to predict
to beat the average-wear-curve). Inside EVERY leave-one-tool-out fold, consensus selection
(Kendall/Spearman/MMI) and a SHAP 4th-vote audit are computed on the TRAINING tools only; the
held-out tool is never seen during selection or fitting. Expected (and reported honestly):
R^2 <= 0 -> no generalizable sensor signal (the population model is near-optimal).

Output: results/loto_feature_audit.csv (+ console summary).
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from phm.feature_selection_p8 import select_topk, shap_scores
from phm.augmentation_p8 import jitter        # fold-safe augmentation (train-tools only)

K = 8
N_AUG = 12          # synthetic train rows per fold (fold-safe, never touches held-out tool)


def build_table():
    f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
    meta = ["experiment_id", "tool_id", "within_tool_order", "vb_um", "vc", "fz", "cooling"]
    feat = [c for c in f.columns if c not in meta]
    rows = []
    for t, g in f.groupby("tool_id"):
        g = g.sort_values("within_tool_order")
        rate = np.polyfit(g.within_tool_order.values.astype(float), g.vb_um.values.astype(float), 1)[0]
        rec = {"tool_id": t, "rate": rate,
               "vc": (g.vc.iloc[0] - 67) / 10.0, "fz": (g.fz.iloc[0] - 0.19) / 0.1,
               "cool": 1.0 if "cool" in str(g.cooling.iloc[0]).lower() else 0.0}
        rec.update(g[feat].mean().to_dict())     # tool-mean sensor features
        rows.append(rec)
    return pd.DataFrame(rows), feat


def loto(df, cols, label):
    y = df["rate"].values
    X = df[cols].reset_index(drop=True)
    pred = np.zeros(len(y)); sel_count = {}
    for tr, te in LeaveOneOut().split(X):
        Xtr, ytr = X.iloc[tr], y[tr]
        sel, _ = select_topk(Xtr, ytr, cols, k=K)        # fold-safe consensus selection
        sh = shap_scores(Xtr, ytr, sel)                  # fold-safe SHAP 4th-vote audit
        sel = [c for c, _ in sorted(sh.items(), key=lambda kv: -kv[1])][:K] or sel
        for c in sel:
            sel_count[c] = sel_count.get(c, 0) + 1
        mu, sd = Xtr[sel].mean(), Xtr[sel].std() + 1e-9
        Xs = ((Xtr[sel] - mu) / sd).to_numpy(float)
        # fold-safe augmentation: jitter on the SCALED train rows only (label preserved)
        Xa, ya = jitter(Xs, ytr, N_AUG, sigma=0.05, rng=np.random.default_rng(te[0]))
        Xfit = np.vstack([Xs, Xa]); yfit = np.concatenate([ytr, ya])
        m = RidgeCV(alphas=[1, 10, 100, 1000]).fit(Xfit, yfit)
        pred[te] = m.predict((X.iloc[te][sel] - mu) / sd)
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    mae = np.mean(np.abs(y - pred))
    top = sorted(sel_count.items(), key=lambda kv: -kv[1])[:6]
    print(f"  {label:28} LOTO R2={r2:+.2f}  MAE={mae:.2f}  | most-selected: "
          f"{', '.join(f'{c}({n})' for c, n in top)}")
    return dict(view=label, r2=r2, mae=mae, top_features=";".join(f"{c}:{n}" for c, n in top))


def main():
    df, feat = build_table()
    sensors = [c for c in feat]
    cond = ["vc", "fz", "cool"]
    print(f"Leakage-safe LOTO feature audit (18 tools, target=per-tool wear rate, fold-safe "
          f"consensus+SHAP, k={K}):")
    res = [loto(df, cond, "condition only"),
           loto(df, sensors, "sensors (fold-safe sel+SHAP)"),
           loto(df, cond + sensors, "condition + sensors")]
    print(f"  {'dummy (mean rate)':28} LOTO R2=+0.00  (reference)")
    out = os.path.join(ROOT, "results", "loto_feature_audit.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(res).to_csv(out, index=False)
    print(f"\nwrote {out}")
    if all(r["r2"] <= 0.05 for r in res):
        print("VERDICT: no generalizable sensor/condition signal under fold-safe selection "
              "(R2<=0) -> population model near-optimal; sensor null documented for review.")


if __name__ == "__main__":
    main()
