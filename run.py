#!/usr/bin/env python3
"""Single entry point for the project's flows.

    python run.py              list the available tasks
    python run.py <task>       run a task (one or more scripts, in order)

Cross-platform (works on Windows PowerShell, macOS, Linux, CI) and needs no
extra tooling. Each task is just a documented sequence of the scripts in
scripts/. This file is the canonical map of how the project runs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# task -> (one-line description, [scripts to run in order])
TASKS: dict[str, tuple[str, list[str]]] = {
    "dataset":   ("Build experiment_features.csv (legacy T01, per-contact)",
                  ["scripts/build_dataset.py"]),
    "ingest":    ("Build experiment_features.csv from raw multi-cutter signals (aggregate-over-parts, cached)",
                  ["scripts/ingest.py"]),
    "benchmark": ("Run the 36-branch LOEO benchmark (rankings, figures, SHAP)",
                  ["scripts/run_layered_pipeline.py"]),
    "physics":   ("Validate the wear physics on the data (monotonicity, rate, energy)",
                  ["scripts/validate_physics_assumptions.py"]),
    "time-aware": ("Run time-aware baselines: Linear(t), ElasticNet(x,t), MLP(x,t) under LOEO (P2)",
                  ["scripts/run_time_aware_baselines.py"]),
    "pinn":      ("Train and evaluate the wear-curve PINN under LOEO (config search)",
                  ["scripts/run_pinn.py"]),
    "pinn-compare": ("Run P3 PINN ablation study: G1 vs G2 vs G3 under LOEO",
                  ["scripts/run_pinn_comparison.py"]),
    "rul":       ("Derive conceptual RUL from VB(t) curves (P4; threshold from config/physics.yaml)",
                  ["scripts/derive_rul.py"]),
    "uncertainty": ("Descriptive uncertainty on VB(t)/t_failure/RUL: bootstrap + PINN deep ensemble (P5)",
                  ["scripts/uncertainty_analysis.py"]),
    "paper-prepare": ("Extract the Overleaf ZIP from paper/overleaf_inbox into a working copy + manifest (P6)",
                  ["scripts/prepare_overleaf_workspace.py"]),
    "report":    ("Rebuild the presentation decks (Spanish + English)",
                  ["scripts/build_pipeline_presentation.py",
                   "scripts/build_pipeline_presentation_en.py"]),
    "audit":     ("Run data-quality and output audits",
                  ["scripts/audit_data.py", "scripts/audit_outputs.py"]),
    "signal-qa": ("Raw-signal QA: per-file metrics + sampling-rate consistency (P7 gate)",
                  ["scripts/run_signal_qa.py"]),
    "exp77-sensitivity": ("P7.1 diagnostic: 4-vs-6 contact sensitivity of experiment-level features",
                  ["scripts/run_exp77_contact_sensitivity.py"]),
    "rebaseline-vb": ("P8.0 re-baseline: time-aware baselines + RUL on the OFFICIAL VB source (microscope_vb.csv)",
                  ["scripts/rebaseline_official_vb.py"]),
    "rebaseline-p8-1": ("P8.1 bridge re-baseline: classical ML + PINN on official VB (current features)",
                  ["scripts/rebaseline_p8_1.py"]),
    "p8-2-features": ("P8.2 full-contact segmentation + multi-domain feature refactor",
                  ["scripts/run_p8_2_segmentation_features.py"]),
    "p8-2c-compare": ("P8.2C full-contact-original vs active-window-refined feature QA comparison",
                  ["scripts/run_p8_2c_full_vs_active.py"]),
    "p8-3-benchmark": ("P8.3 fold-safe Kendall/Spearman/MMI selection + official-VB benchmark (source x branch x model)",
                  ["scripts/run_p8_3_benchmark.py"]),
    "pinn-softplus": ("P8.4 PINN softplus-rate ablation (positive wear-rate) on official VB, reliability-aware features",
                  ["scripts/run_pinn_softplus_rate.py"]),
    "hi-rul-sweep": ("HI/DI + RUL threshold sweep (220/250/300/600) from VB curves; honest extrapolation flags",
                  ["scripts/run_hi_rul_threshold_sweep.py"]),
    "p8-6-branches": ("P8.6 sensor branch consolidation: SOLO_A / SOLO_R / FUSION_AR / reliability-aware (official VB)",
                  ["scripts/run_p8_6_sensor_branches.py"]),
    "p8-7-shap-audit": ("P8.7 SHAP-guided selection audit (fold-safe): SHAP as 4th consensus vote, coherence + MAE effect",
                  ["scripts/run_p8_7_shap_audit.py"]),
    "p8-8-augmentation": ("P8.8 fold-safe augmentation eval: R0/R1/R2/R3 (jitter/interpolation) under LOEO; adopt only if real held-out improves",
                  ["scripts/run_p8_8_augmentation_eval.py"]),
    "p8-9-pinn-rate-sweep": ("P8.9 PINN physics refinement: lambda_rate sweep (softplus positive rate); coherence-vs-accuracy trade-off",
                  ["scripts/run_p8_9_pinn_rate_sweep.py"]),
    "p8-10-uncertainty": ("P8.10 uncertainty & robustness: epistemic bands (bootstrap/ensemble) + structural disagreement + exp77 sensitivity",
                  ["scripts/run_p8_10_uncertainty.py"]),
    "p8-11-pinn-leakage": ("P8.11 leakage audit of headline PINN_mono (6.65): permutation + drop-t + isolation; confirms no leakage / t-driven",
                  ["scripts/run_p8_11_pinn_leakage_audit.py"]),
    "export-model": ("Train + save the deployable sensor wear model (models/wear_model.joblib + .json + MODEL_CARD.md) for use on new data",
                  ["scripts/export_model.py"]),
    "diag-pinn-vs-line": ("DIAGNOSTIC (standalone, not in `pipeline`): decompose why PINN loses to the time control "
                  "into NN-tax + noisy-sensor-tax",
                  ["scripts/diag_pinn_vs_line.py"]),
    "all":       ("dataset -> benchmark -> report",
                  ["scripts/build_dataset.py",
                   "scripts/run_layered_pipeline.py",
                   "scripts/build_pipeline_presentation.py",
                   "scripts/build_pipeline_presentation_en.py"]),
    "pipeline":  ("OFFICIAL end-to-end pipeline (run from the main repo where data/ lives): "
                  "QA -> segmentation+features -> fold-safe selection+benchmark -> SHAP audit -> "
                  "augmentation eval -> time baselines+RUL -> classical+PINN -> PINN ablation -> "
                  "PINN rate sweep -> sensor branches -> HI/RUL sweep -> uncertainty -> leakage audit",
                  ["scripts/run_signal_qa.py",
                   "scripts/run_p8_2_segmentation_features.py",
                   "scripts/run_p8_3_benchmark.py",
                   "scripts/run_p8_7_shap_audit.py",
                   "scripts/run_p8_8_augmentation_eval.py",
                   "scripts/rebaseline_official_vb.py",
                   "scripts/rebaseline_p8_1.py",
                   "scripts/run_pinn_softplus_rate.py",
                   "scripts/run_p8_9_pinn_rate_sweep.py",
                   "scripts/run_p8_6_sensor_branches.py",
                   "scripts/run_hi_rul_threshold_sweep.py",
                   "scripts/run_p8_10_uncertainty.py",
                   "scripts/run_p8_11_pinn_leakage_audit.py"]),
}


def _list() -> int:
    print("Usage: python run.py <task> [args]\n\nTasks:")
    width = max(len(k) for k in TASKS)
    for name, (desc, _) in TASKS.items():
        print(f"  {name:<{width}}  {desc}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        return _list()
    task = sys.argv[1]
    if task not in TASKS:
        print(f"Unknown task: {task!r}\n")
        return _list() or 2
    _, scripts = TASKS[task]
    for script in scripts:
        print(f"\n>>> {script}")
        result = subprocess.run([PY, str(ROOT / script), *sys.argv[2:]], cwd=ROOT)
        if result.returncode != 0:
            print(f"!!! {script} failed (exit {result.returncode}); stopping.")
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
