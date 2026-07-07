"""
run_nested_benchmark_metrics.py — resumable, memory-safe benchmark runner for
Point 1 (nested tuning inside LOEO). Produces ONLY the metrics/ranking/summary
CSVs needed for the before/after comparison; no SHAP, no figures, no model
retention (those are regenerable via `python run.py benchmark`).

Why a separate runner: the full pipeline retains every trained model
(incl. torch BNN) across 36 branches and runs SHAP+figures, which made a long
single-process run fragile. This driver writes each branch's metrics to disk
immediately and records completion, so an interruption resumes instead of
restarting. Methodology is identical to layered_pipeline.run_branch (nested
tuning on real train-fold only; augmentation applied to fit, never inner CV).

Usage:
    python scripts/run_nested_benchmark_metrics.py            # run/resume
    python scripts/run_nested_benchmark_metrics.py --fresh     # ignore resume state
"""
from __future__ import annotations
import sys, time, warnings, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from phm.config import PROCESSED_DATASET, EXPERIMENT_ID_COL
from phm.dataset_builder import get_feature_columns
from phm import layered_pipeline as lp
from phm import leakage_audit

OUT = ROOT / "outputs" / "metrics" / "layered_pipeline"
OUT.mkdir(parents=True, exist_ok=True)
DONE_FILE = OUT / "_resume_completed_branches.txt"
ALL_METRICS = OUT / "09_all_metrics.csv"
ALL_TUNING = OUT / "09_tuning_results_all.csv"


def _append_csv(df: pd.DataFrame, path: Path):
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignore resume state")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated branch_ids to run (others skipped); "
                         "resume state still respected")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="skip final aggregation/audit (useful for partial runs)")
    args = ap.parse_args()
    only = {b.strip() for b in args.only.split(",") if b.strip()}

    if args.fresh:
        for p in (DONE_FILE, ALL_METRICS, ALL_TUNING):
            if p.exists():
                p.unlink()

    print(f"[nested-benchmark] loading {PROCESSED_DATASET}", flush=True)
    df = pd.read_csv(PROCESSED_DATASET)
    feat_cols = get_feature_columns(df)
    print(f"[nested-benchmark] df={df.shape} exps={df[EXPERIMENT_ID_COL].nunique()} "
          f"feats={len(feat_cols)}", flush=True)

    done = set()
    if DONE_FILE.exists():
        done = {l.strip() for l in DONE_FILE.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"[nested-benchmark] resume: {len(done)} branches already done", flush=True)

    branches = lp.enumerate_branches()
    t_start = time.time()
    for i, spec in enumerate(branches, 1):
        bid = spec["branch_id"]
        if only and bid not in only:
            continue
        if bid in done:
            print(f"[{i}/{len(branches)}] SKIP {bid} (resume)", flush=True)
            continue
        sub_feats = lp.get_features_for_subset(feat_cols, spec["feature_subset"])
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = lp.run_branch(
                    branch_id=bid, feature_subset=spec["feature_subset"],
                    data_branch=spec["data_branch"], tuning_method=spec["tuning_method"],
                    aug_strategy=spec["aug_strategy"], full_df=df, feat_cols=sub_feats,
                )
            _append_csv(pd.DataFrame(res["metrics_rows"]), ALL_METRICS)
            if res["tuning_rows"]:
                _append_csv(pd.DataFrame(res["tuning_rows"]), ALL_TUNING)
            with DONE_FILE.open("a", encoding="utf-8") as f:
                f.write(bid + "\n")
            dt = time.time() - t0
            print(f"[{i}/{len(branches)}] DONE {bid} in {dt:.1f}s "
                  f"(elapsed {(time.time()-t_start)/60:.1f}min)", flush=True)
        except Exception as exc:
            print(f"[{i}/{len(branches)}] FAIL {bid}: {exc}", flush=True)

    # ---- aggregate summaries from the accumulated metrics ----
    if args.no_aggregate:
        print("[nested-benchmark] --no-aggregate: stopping after branch runs", flush=True)
        return 0
    if not ALL_METRICS.exists():
        print("[nested-benchmark] no metrics produced", flush=True)
        return 1
    dfm = pd.read_csv(ALL_METRICS)
    print(f"\n[nested-benchmark] aggregating {len(dfm)} metric rows "
          f"from {dfm['branch_id'].nunique()} branches", flush=True)

    lp_rank = lp.build_final_ranking(dfm)
    lp_rank.to_csv(OUT / "09_final_layered_ranking.csv", index=False)
    lp.build_branch_best_summary(dfm).to_csv(OUT / "09_branch_best_summary.csv", index=False)
    lp.build_delta_vs_baseline(dfm).to_csv(OUT / "09_delta_vs_baseline.csv", index=False)
    lp.build_tuning_effect_summary(dfm).to_csv(OUT / "09_tuning_effect_summary.csv", index=False)
    lp.build_augmentation_effect_summary(dfm).to_csv(OUT / "09_augmentation_effect_summary.csv", index=False)
    lp.build_random_vs_grid_summary(dfm).to_csv(OUT / "09_random_vs_grid_summary.csv", index=False)
    lp.build_model_evolution_summary(dfm).to_csv(OUT / "09_model_evolution_summary.csv", index=False)

    # ---- refresh leakage audit (includes tuning_train_fold_only) ----
    print("[nested-benchmark] running leakage audit", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audit = leakage_audit.run_all_checks(df)
    audit.to_csv(OUT / "00_leakage_checks.csv", index=False)
    print(audit.to_string(index=False), flush=True)

    if not lp_rank.empty and lp_rank["MAE"].notna().any():
        top = lp_rank.dropna(subset=["MAE"]).iloc[0]
        print(f"\n[nested-benchmark] BEST: {top['model']} @ {top['branch_id']} "
              f"MAE={top['MAE']:.2f} R2={top['R2']:.3f}", flush=True)
    print(f"[nested-benchmark] TOTAL TIME {(time.time()-t_start)/60:.1f} min", flush=True)
    print("[nested-benchmark] COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
