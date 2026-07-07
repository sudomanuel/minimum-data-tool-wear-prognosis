#!/usr/bin/env python3
"""
run_p8_3_benchmark.py — P8.3 fold-safe feature selection + official-VB benchmark.

Crosses segmentation_source {full_contact_original, active_window_refined}
     x feature_branch {B0 time_only, B1 sensor_only, B2 sensor_plus_time,
                       B3 reliability_aware, B4 PINN_ready_minimal}
     x classical models, under LOEO with TRAIN-ONLY selection/scaling.

NO synthetic data, NO gate, NO PINN training (PINN re-baseline is a separate step).
Target = official VB (microscope_vb.csv) via the branch tables (VB_um column).

Uso:  python run.py p8-3-benchmark  [--topk 10]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "src"))
from phm.feature_selection_p8 import (sensor_cols, reliability_aware_cols, PINN_READY,
                                      select_topk, score_features, all_feature_cols)  # noqa: E402

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

RESULTS = WT / "results"
FEATURES = WT / "data" / "features"
SOURCES = {"full_contact_original": FEATURES / "p8_2_features_experiment_full_contact.csv",
           "active_window_refined": FEATURES / "p8_2_features_experiment_active_window.csv"}
# official P8.0/P8.1 references
REF = {"Linear(t)_official": 3.10, "Poly2(t)_official": 3.83,
       "PINN_mono_P8.1": 6.65, "best_classical_P8.1": 23.66}


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
    r2 = float(1 - np.sum((yt - yp) ** 2) / ss) if ss > 0 else float("nan")
    mape = float(np.mean(np.abs((yt - yp) / yt)) * 100)
    return mae, rmse, r2, mape


def time_only(df, degree):
    t = df["physical_experiment_order"].to_numpy(float)
    y = df["VB_um"].to_numpy(float)
    yp = np.zeros(len(df))
    for i in range(len(df)):
        m = np.arange(len(df)) != i
        c = np.polyfit(t[m], y[m], degree)
        yp[i] = np.polyval(c, t[i])
    return yp, y


def branch_pool(df, branch):
    if branch == "B1_sensor_only":
        return sensor_cols(df)
    if branch == "B2_sensor_plus_time":
        return sensor_cols(df) + ["physical_experiment_order"]
    if branch == "B3_reliability_aware":
        return [c for c in reliability_aware_cols(df) if c != "physical_experiment_order"]
    if branch == "B4_PINN_ready_minimal":
        return [c for c in PINN_READY if c in df.columns]
    return []


def loeo_select_fit(df, pool, model_builder, topk, do_select, seed=0):
    ids = df["experiment_id"].to_numpy()
    y = df["VB_um"].to_numpy(float)
    yp = np.zeros(len(df))
    fold_sel = {}
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        Xtr_full, Xte_full = df.iloc[tr], df.iloc[[i]]
        if do_select and len(pool) > topk:
            sel, _ = select_topk(Xtr_full, y[tr], pool, k=topk, seed=seed)
        else:
            sel = list(pool)
        fold_sel[int(ids[i])] = sel  # selection for the fold leaving out experiment ids[i]
        Xtr = Xtr_full[sel].to_numpy(float)
        Xte = Xte_full[sel].to_numpy(float)
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xte = np.where(np.isnan(Xte), med, Xte)
        sc = StandardScaler().fit(Xtr)
        m = model_builder().fit(sc.transform(Xtr), y[tr])
        yp[i] = m.predict(sc.transform(Xte))[0]
    return yp, y, fold_sel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    branches = ["B1_sensor_only", "B2_sensor_plus_time", "B3_reliability_aware",
                "B4_PINN_ready_minimal"]
    rows, preds, fold_scores, sel_json, global_rows = [], [], [], {}, []

    for source, path in SOURCES.items():
        df = pd.read_csv(path).sort_values("physical_experiment_order").reset_index(drop=True)
        y = df["VB_um"].to_numpy(float)
        e77 = int(np.where(df.experiment_id == 77)[0][0])

        # ---- B0 time-only ----
        for name, deg in [("Linear(t)", 1), ("Poly2(t)", 2)]:
            yp, yt = time_only(df, deg)
            mae, rmse, r2, mape = metrics(yt, yp)
            rows.append(dict(segmentation_source=source, feature_branch="B0_time_only",
                             model=name, selected_feature_count=1, MAE=round(mae, 3),
                             RMSE=round(rmse, 3), R2=round(r2, 3), MAPE=round(mape, 2),
                             residual_exp77=round(float(yp[e77] - yt[e77]), 2),
                             notes="time-only control"))
            for j in range(len(df)):
                preds.append(dict(segmentation_source=source, feature_branch="B0_time_only",
                                  model=name, experiment_id=int(df.experiment_id[j]),
                                  physical_experiment_order=int(df.physical_experiment_order[j]),
                                  VB_true=float(yt[j]), VB_pred=float(yp[j])))

        # ---- global exploratory scores (tagged exploratory_only) ----
        gsc = score_features(df, y, sensor_cols(df), seed=0).head(40)
        gsc["segmentation_source"] = source
        gsc["exploratory_only"] = True
        global_rows.append(gsc)

        # ---- B1..B4 ----
        for branch in branches:
            pool = branch_pool(df, branch)
            do_select = branch in ("B1_sensor_only", "B2_sensor_plus_time", "B3_reliability_aware")
            for mname, mb in models().items():
                yp, yt, fold_sel = loeo_select_fit(df, pool, mb, args.topk, do_select)
                mae, rmse, r2, mape = metrics(yt, yp)
                k = int(np.median([len(v) for v in fold_sel.values()]))
                rows.append(dict(segmentation_source=source, feature_branch=branch, model=mname,
                                 selected_feature_count=k, MAE=round(mae, 3), RMSE=round(rmse, 3),
                                 R2=round(r2, 3), MAPE=round(mape, 2),
                                 residual_exp77=round(float(yp[e77] - yt[e77]), 2),
                                 notes=""))
                for j in range(len(df)):
                    preds.append(dict(segmentation_source=source, feature_branch=branch,
                                      model=mname, experiment_id=int(df.experiment_id[j]),
                                      physical_experiment_order=int(df.physical_experiment_order[j]),
                                      VB_true=float(yt[j]), VB_pred=float(yp[j])))
                if do_select and mname == "ElasticNet":
                    sel_json[f"{source}|{branch}"] = {str(k_): v for k_, v in fold_sel.items()}
                    # per-fold scores (record ElasticNet pool once)
                    for fid, sel in fold_sel.items():
                        for rank, feat in enumerate(sel, 1):
                            fold_scores.append(dict(segmentation_source=source, feature_branch=branch,
                                                    held_out_experiment=fid, rank=rank, feature=feat))
            print(f"  [{source[:12]}] {branch:24s} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "p8_3_official_vb_benchmark_results.csv", index=False)
    pd.DataFrame(preds).to_csv(RESULTS / "p8_3_fold_predictions.csv", index=False)
    pd.concat(global_rows, ignore_index=True).to_csv(
        FEATURES / "p8_3_feature_scores_global_exploratory.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(FEATURES / "p8_3_feature_scores_by_fold.csv", index=False)
    json.dump(sel_json, open(FEATURES / "p8_3_selected_features_by_fold.json", "w"), indent=1)

    # segmentation-source comparison (best MAE per source x branch)
    comp = (res[res.feature_branch != "B0_time_only"]
            .groupby(["feature_branch", "segmentation_source"])["MAE"].min().unstack())
    comp["winner"] = comp.idxmin(axis=1)
    comp["delta"] = (comp.get("active_window_refined") - comp.get("full_contact_original"))
    comp.reset_index().to_csv(RESULTS / "p8_3_segmentation_source_comparison.csv", index=False)

    print(f"\nDONE in {time.time()-t0:.0f}s. {len(res)} runs.")
    print("\n=== best MAE per branch x source ===")
    print(comp.round(2).to_string())
    print("\n=== overall best (non-time) sensor models ===")
    top = res[res.feature_branch.isin(branches)].nsmallest(6, "MAE")
    print(top[["segmentation_source", "feature_branch", "model", "selected_feature_count",
               "MAE", "R2", "residual_exp77"]].to_string(index=False))
    print(f"\nreferences: Linear(t)={REF['Linear(t)_official']} Poly2(t)={REF['Poly2(t)_official']} "
          f"PINN_mono_P8.1={REF['PINN_mono_P8.1']} bestClassical_P8.1={REF['best_classical_P8.1']}")


if __name__ == "__main__":
    main()
