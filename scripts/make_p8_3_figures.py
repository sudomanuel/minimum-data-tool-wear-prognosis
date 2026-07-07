"""
make_p8_3_figures.py — P8.3 result figures (run after run_p8_3_benchmark.py). (2026-06-13)

Generates:
  outputs/figures/p8_3_model_comparison_mae.png
  outputs/figures/p8_3_segmentation_source_comparison.png
  outputs/figures/p8_3_selected_feature_frequency.png
  outputs/figures/p8_3_predicted_vs_actual_best_model.png
  outputs/figures/p8_3_exp77_residuals.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[1]
RES = WT / "results"
FEAT = WT / "data" / "features"
OUT = WT / "outputs" / "figures"
REF = {"Linear(t) 3.10": 3.10, "Poly2(t) 3.83": 3.83, "PINN_mono P8.1 6.65": 6.65}
C_FULL = "#9C6B1E"
C_ACTIVE = "#2E6F62"


def main():
    res = pd.read_csv(RES / "p8_3_official_vb_benchmark_results.csv")
    preds = pd.read_csv(RES / "p8_3_fold_predictions.csv")
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. model comparison MAE (best model per branch, both sources)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    branches = sorted(res.feature_branch.unique())
    x = np.arange(len(branches))
    for k, (src, c) in enumerate([("full_contact_original", C_FULL),
                                  ("active_window_refined", C_ACTIVE)]):
        best = [res[(res.feature_branch == b) & (res.segmentation_source == src)].MAE.min()
                for b in branches]
        ax.bar(x + (k - 0.5) * 0.4, best, 0.38, label=src, color=c, alpha=0.9)
    for lbl, v in REF.items():
        ax.axhline(v, ls="--", lw=1, alpha=0.7)
        ax.text(len(branches) - 0.4, v, lbl, fontsize=7, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(branches, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("best LOEO MAE (µm)")
    ax.set_title("P8.3 — best MAE per feature branch × segmentation source (official VB)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "p8_3_model_comparison_mae.png",
                                    bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 2. segmentation source comparison (scatter full vs active per branch×model)
    piv = res[res.feature_branch != "B0_time_only"].pivot_table(
        index=["feature_branch", "model"], columns="segmentation_source", values="MAE")
    piv = piv.dropna()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    ax.scatter(piv["full_contact_original"], piv["active_window_refined"],
               c="#1F4E79", alpha=0.7)
    lim = [0, max(piv.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="equal")
    ax.set_xlabel("full_contact_original MAE"); ax.set_ylabel("active_window_refined MAE")
    ax.set_title("P8.3 — segmentation source MAE (below line = active better)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "p8_3_segmentation_source_comparison.png",
                                    bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 3. selected feature frequency (fold-wise)
    fs = pd.read_csv(FEAT / "p8_3_feature_scores_by_fold.csv")
    freq = fs.feature.value_counts().head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    ax.barh(freq.index, freq.values, color="#4A6628")
    ax.set_xlabel("times selected across folds/branches")
    ax.set_title("P8.3 — most frequently selected features (fold-wise, train-only)")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "p8_3_selected_feature_frequency.png",
                                    bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 4. predicted vs actual (best non-time model) + 5. exp77 residuals
    nontime = res[res.feature_branch != "B0_time_only"]
    best = nontime.loc[nontime.MAE.idxmin()]
    bp = preds[(preds.segmentation_source == best.segmentation_source)
               & (preds.feature_branch == best.feature_branch)
               & (preds.model == best.model)].sort_values("physical_experiment_order")
    lt = preds[(preds.feature_branch == "B0_time_only") & (preds.model == "Linear(t)")
               & (preds.segmentation_source == best.segmentation_source)].sort_values("physical_experiment_order")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax.plot(bp.physical_experiment_order, bp.VB_true, "ko-", label="measured VB (official)")
    ax.plot(bp.physical_experiment_order, bp.VB_pred, "s--", color="#1F4E79",
            label=f"best sensor: {best.feature_branch}/{best.model} (MAE {best.MAE})")
    ax.plot(lt.physical_experiment_order, lt.VB_pred, "^:", color="#8C2D2D",
            label="Linear(t) control (MAE 3.10)")
    ax.set_xlabel("physical_experiment_order"); ax.set_ylabel("VB (µm)")
    ax.set_title("P8.3 — measured vs predicted VB (best sensor model vs time control)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "p8_3_predicted_vs_actual_best_model.png",
                                    bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 5. exp77 residuals across branches (best model per branch, full source)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    rows = []
    for b in sorted(res.feature_branch.unique()):
        sub = res[res.feature_branch == b]
        bm = sub.loc[sub.MAE.idxmin()]
        rows.append((f"{b}\n{bm.model}", bm.residual_exp77))
    labels, vals = zip(*rows)
    colors = ["#8C2D2D" if v < 0 else "#4A6628" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("exp77 residual (µm)")
    ax.set_title("P8.3 — exp77 end-of-life residual by branch (best model)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "p8_3_exp77_residuals.png",
                                    bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("wrote 5 P8.3 figures to", OUT)


if __name__ == "__main__":
    main()
