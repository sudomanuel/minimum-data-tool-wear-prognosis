"""
build_before_after_nested.py — Point 1 closure artifact.

Compares the global-tuning benchmark (before) vs the nested-tuning benchmark
(after) and writes:
  - outputs/metrics/benchmark_before_after_nested_tuning.csv
  - outputs/metrics/nested_tuning_audit.csv
Then prints the P1 reporting block (best before/after, winner change, whether
tuning/augmentation still don't help, whether SOLO_A still dominates).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MET = ROOT / "outputs" / "metrics"
BEFORE = MET / "_before_nested_global_tuning" / "09_all_metrics.csv"
AFTER = MET / "layered_pipeline" / "09_all_metrics.csv"
AFTER_TUNING = MET / "layered_pipeline" / "09_tuning_results_all.csv"
OUT_CMP = MET / "benchmark_before_after_nested_tuning.csv"
OUT_AUDIT = MET / "nested_tuning_audit.csv"

KEY = ["branch_id", "model"]
METRICS = ["MAE", "RMSE", "R2", "MAPE_%"]


def _load(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df = df[df["validation_type"] == "loeo"].copy()
    return df


def _best_row(df: pd.DataFrame):
    d = df.dropna(subset=["MAE"]).sort_values("MAE")
    return d.iloc[0] if not d.empty else None


def main() -> int:
    if not AFTER.exists():
        print(f"ERROR: after metrics not found: {AFTER}")
        return 1
    before = _load(BEFORE)
    after = _load(AFTER)

    b = before[KEY + METRICS + ["feature_subset", "data_branch", "tuning_method",
                                "augmentation_strategy"]].copy()
    a = after[KEY + METRICS].copy()
    b = b.rename(columns={m: f"{m}_before" for m in METRICS})
    a = a.rename(columns={m: f"{m}_after" for m in METRICS})

    cmp = b.merge(a, on=KEY, how="outer")
    cmp = cmp.rename(columns={
        "feature_subset": "feature_view",
        "tuning_method": "tuning_mode",
    })
    # data_condition = N | A:<strategy>
    def _dc(r):
        if r.get("data_branch") == "A":
            return f"A:{r.get('augmentation_strategy')}"
        return "N(real)"
    cmp["data_condition"] = cmp.apply(_dc, axis=1)

    for m in METRICS:
        suffix = "MAPE" if m == "MAPE_%" else m
        cmp[f"delta_{suffix}"] = cmp[f"{m}_after"] - cmp[f"{m}_before"]
    cmp = cmp.rename(columns={"delta_MAPE_%": "delta_MAPE"}) if "delta_MAPE_%" in cmp else cmp

    # global winner before/after
    wb = _best_row(before); wa = _best_row(after)
    winner_before = (wb["branch_id"], wb["model"]) if wb is not None else None
    winner_after = (wa["branch_id"], wa["model"]) if wa is not None else None
    winner_changed = bool(winner_before != winner_after)
    cmp["winner_changed"] = winner_changed
    cmp["is_best_before"] = cmp.apply(
        lambda r: winner_before == (r["branch_id"], r["model"]), axis=1)
    cmp["is_best_after"] = cmp.apply(
        lambda r: winner_after == (r["branch_id"], r["model"]), axis=1)

    def _concl(r):
        if pd.isna(r.get("MAE_before")) or pd.isna(r.get("MAE_after")):
            return "missing in one run"
        d = r["delta_MAE"]
        if r["tuning_mode"] == "none":
            return ("ST branch: identical before/after by construction"
                    if abs(d) < 1e-6 else
                    f"ST branch but delta_MAE={d:+.3f} (UNEXPECTED — investigate)")
        if abs(d) < 1.0:
            return f"nested tuning: negligible change ({d:+.2f} µm)"
        return (f"nested tuning improved MAE by {-d:.2f} µm" if d < 0
                else f"nested tuning worsened MAE by {d:.2f} µm (honest: global tuning was optimistic)")
    cmp["conclusion"] = cmp.apply(_concl, axis=1)

    col_order = ["branch_id", "model", "tuning_mode", "data_condition", "feature_view",
                 "MAE_before", "MAE_after", "delta_MAE",
                 "RMSE_before", "RMSE_after", "delta_RMSE",
                 "R2_before", "R2_after", "delta_R2",
                 "MAPE_%_before", "MAPE_%_after", "delta_MAPE",
                 "winner_changed", "is_best_before", "is_best_after", "conclusion"]
    cmp = cmp[[c for c in col_order if c in cmp.columns]]
    cmp = cmp.sort_values("MAE_after", na_position="last")
    cmp.to_csv(OUT_CMP, index=False)
    print(f"[write] {OUT_CMP}  ({len(cmp)} rows)")

    # ---- nested tuning audit ----
    if AFTER_TUNING.exists():
        t = pd.read_csv(AFTER_TUNING)
        keep = [c for c in ["branch_id", "model", "feature_subset", "data_branch",
                            "tuning_method", "augmentation_strategy", "n_folds_tuned",
                            "params_stable_across_folds", "best_cv_score_mae",
                            "best_params", "best_params_full_data", "cv_strategy"]
                if c in t.columns]
        t[keep].to_csv(OUT_AUDIT, index=False)
        print(f"[write] {OUT_AUDIT}  ({len(t)} rows)")
        tuned = t[t["tuning_method"].isin(["random", "grid"])] if "tuning_method" in t else t
        if "params_stable_across_folds" in t.columns and not tuned.empty:
            n_unstable = int((tuned["params_stable_across_folds"] == False).sum())
            print(f"[audit] tuned (model,branch) configs: {len(tuned)}; "
                  f"with params UNSTABLE across folds: {n_unstable}")

    # ================= REPORT BLOCK =================
    print("\n" + "=" * 64)
    print("POINT 1 — BEFORE/AFTER REPORT")
    print("=" * 64)
    if wb is not None:
        print(f"BEST BEFORE (global tuning): {wb['model']} @ {wb['branch_id']}")
        print(f"   MAE={wb['MAE']:.2f}  RMSE={wb['RMSE']:.2f}  R2={wb['R2']:.3f}  MAPE={wb['MAPE_%']:.2f}%")
    if wa is not None:
        print(f"BEST AFTER  (nested tuning): {wa['model']} @ {wa['branch_id']}")
        print(f"   MAE={wa['MAE']:.2f}  RMSE={wa['RMSE']:.2f}  R2={wa['R2']:.3f}  MAPE={wa['MAPE_%']:.2f}%")
    print(f"WINNER CHANGED: {winner_changed}")

    # tuning still no help? (after data)
    nst = after[(after.data_branch == "N") & (after.tuning_method == "none")]["MAE"].dropna()
    nct = after[(after.data_branch == "N") & (after.tuning_method.isin(["random", "grid"]))]["MAE"].dropna()
    if len(nst) and len(nct):
        d = float(nct.min() - nst.min())
        print(f"TUNING (N, after): best ST={nst.min():.2f}  best CT={nct.min():.2f}  "
              f"Δ={d:+.2f} µm  -> {'tuning still does NOT help' if d >= -1 else 'tuning helps'}")
    # augmentation still no help?
    nbest = after[after.data_branch == "N"]["MAE"].dropna()
    abest = after[after.data_branch == "A"]["MAE"].dropna()
    if len(nbest) and len(abest):
        d = float(abest.min() - nbest.min())
        print(f"AUGMENTATION (after): best N={nbest.min():.2f}  best A={abest.min():.2f}  "
              f"Δ={d:+.2f} µm  -> {'augmentation still does NOT help' if d >= -1 else 'augmentation helps'}")
    # SOLO_A dominance?
    rank_after = after.dropna(subset=["MAE"]).sort_values("MAE")
    if not rank_after.empty:
        top_view = rank_after.iloc[0]["feature_subset"]
        top5 = rank_after.head(5)["feature_subset"].tolist()
        print(f"FEATURE VIEW: winner={top_view}; top-5 views={top5}  "
              f"-> {'SOLO_A still dominates' if top_view == 'SOLO_A' else 'dominance changed'}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
