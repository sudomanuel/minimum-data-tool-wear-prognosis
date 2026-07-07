#!/usr/bin/env python3
"""
run_p8_6_sensor_branches.py — P8.6 sensor branch consolidation.

Explicit, reproducible LOEO comparison on the OFFICIAL VB of four channel branches:
  SOLO_A                      axial-only sensor features
  SOLO_R                      rotational-only sensor features
  FUSION_AR                   axial + rotational + A/R fusion (ratios/diffs/combined)
  FUSION_AR_reliability_aware FUSION_AR minus contaminated features (energy aggregates biased by
                              exp77's missing contacts, cumulative time-like, contact counts)

Fold-safe Kendall/Spearman/MMI top-k selection (train-only) + classical battery. Answers:
 1) does axial-only beat rotational-only? 2) does fusion help or add noise? 3) does the
 reliability-aware branch avoid contaminated vars (energy_total-style)? 4) which branch should
 feed the PINN and the SHAP audit?

No synthetic data, gate closed, paper untouched, legacy untouched.
Outputs: results/sensor_branch_comparison.csv, reports/sensor_branch_consolidation_report.md,
outputs/figures/sensor_branch_performance.png, results/sensor_branch_top_features.csv.

Uso:  python run.py p8-6-branches   [--topk 10]
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "src"))
from phm.feature_selection_p8 import select_topk, TIME_LIKE, RELIABILITY_EXCLUDE, META  # noqa: E402

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

RESULTS = WT / "results"
FIGS = WT / "outputs" / "figures"
FEAT = WT / "data" / "features" / "p8_2_features_experiment_full_contact.csv"
REF = {"Linear(t)_official": 3.10, "Poly2(t)_official": 3.83, "PINN_mono_P8.1": 6.65,
       "P8.3_sensor_only_best": 26.24}


def models():
    return {
        "Dummy(mean)": lambda: DummyRegressor(strategy="mean"),
        "Ridge": lambda: Ridge(alpha=10.0),
        "Lasso": lambda: Lasso(alpha=1.0, max_iter=50000),
        "ElasticNet": lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=50000),
        "SVR": lambda: SVR(kernel="rbf", C=10.0, epsilon=1.0),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=0),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=0),
    }


def metrics(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1 - np.sum((yt - yp) ** 2) / ss) if ss > 0 else np.nan
    return mae, rmse, r2


def define_branches(df):
    sensor = [c for c in df.columns if c not in META and not TIME_LIKE.search(c)]
    A = [c for c in sensor if c.startswith("A_")]
    R = [c for c in sensor if c.startswith("R_")]
    fusion_extra = [c for c in sensor if c.startswith("AR_") or c.startswith("combined_")]
    fusion = A + R + fusion_extra
    rel = [c for c in fusion if not RELIABILITY_EXCLUDE.search(c)]
    contaminated = sorted(set(fusion) - set(rel))
    return {"SOLO_A": A, "SOLO_R": R, "FUSION_AR": fusion,
            "FUSION_AR_reliability_aware": rel}, contaminated


def loeo(df, pool, builder, topk, seed=0):
    ids = df["experiment_id"].to_numpy()
    y = df["VB_um"].to_numpy(float)
    yp = np.zeros(len(df))
    selected = Counter()
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        Xtr_full = df.iloc[tr]
        if len(pool) > topk:
            sel, _ = select_topk(Xtr_full, y[tr], pool, k=topk, seed=seed)
        else:
            sel = list(pool)
        selected.update(sel)
        Xtr = Xtr_full[sel].to_numpy(float)
        Xte = df.iloc[[i]][sel].to_numpy(float)
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xte = np.where(np.isnan(Xte), med, Xte)
        sc = StandardScaler().fit(Xtr)
        m = builder().fit(sc.transform(Xtr), y[tr])
        yp[i] = m.predict(sc.transform(Xte))[0]
    return y, yp, selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEAT).sort_values("physical_experiment_order").reset_index(drop=True)
    branches, contaminated = define_branches(df)
    print("branch sizes:", {k: len(v) for k, v in branches.items()})
    print(f"contaminated features excluded by reliability-aware branch: {len(contaminated)} "
          f"(e.g. {contaminated[:4]})\n")

    rows, topfeat_rows = [], []
    e77 = int(np.where(df.experiment_id == 77)[0][0])
    for bname, pool in branches.items():
        best = None
        for mname, mb in models().items():
            y, yp, sel = loeo(df, pool, mb, args.topk)
            mae, rmse, r2 = metrics(y, yp)
            rows.append(dict(branch=bname, n_features_pool=len(pool), model=mname,
                             selected_k=min(args.topk, len(pool)), MAE=round(mae, 3),
                             RMSE=round(rmse, 3), R2=round(r2, 3),
                             exp77_residual=round(float(yp[e77] - y[e77]), 2)))
            if best is None or mae < best[0]:
                best = (mae, mname, sel)
        # top features (fold-wise frequency) for the branch's best model
        for feat, cnt in best[2].most_common(8):
            topfeat_rows.append(dict(branch=bname, best_model=best[1], feature=feat, fold_count=cnt))
        print(f"  {bname:30s} best {best[1]:14s} MAE={best[0]:.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "sensor_branch_comparison.csv", index=False)
    pd.DataFrame(topfeat_rows).to_csv(RESULTS / "sensor_branch_top_features.csv", index=False)

    _fig(res)
    # answers
    best_per = res.loc[res.groupby("branch").MAE.idxmin()].set_index("branch")
    print("\n=== best MAE per branch ===")
    print(best_per[["model", "MAE", "R2", "exp77_residual"]].to_string())
    return res, best_per, contaminated


def _fig(res):
    best = res.loc[res.groupby("branch").MAE.idxmin()]
    order = ["SOLO_A", "SOLO_R", "FUSION_AR", "FUSION_AR_reliability_aware"]
    best = best.set_index("branch").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    colors = ["#1F4E79", "#B3541E", "#4A6628", "#2E6F62"]
    ax.bar(range(len(best)), best.MAE, color=colors, alpha=0.9)
    for i, (_, r) in enumerate(best.iterrows()):
        ax.text(i, r.MAE + 0.3, f"{r.model}\n{r.MAE:.1f}", ha="center", fontsize=8)
    for lbl, v in [("Linear(t) 3.10", 3.10), ("PINN_mono P8.1 6.65", 6.65)]:
        ax.axhline(v, ls="--", lw=1, color="#888")
        ax.text(len(best) - 0.4, v + 0.2, lbl, fontsize=7, ha="right", color="#666")
    ax.set_xticks(range(len(best))); ax.set_xticklabels(order, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("best LOEO MAE (µm)")
    ax.set_title("P8.6 — sensor branch consolidation (official VB; best model per branch)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(FIGS / "sensor_branch_performance.png", bbox_inches="tight",
                                    facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    main()
