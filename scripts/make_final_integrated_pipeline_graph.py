"""
make_final_integrated_pipeline_graph.py — final integrated PHM pipeline roadmap (P8.0).

Generates outputs/figures/final_integrated_phm_pipeline_graph.png:
central node "PINN-based VB estimation and RUL derivation for CNC tool wear" + 13 branches.

Run:  python scripts/make_final_integrated_pipeline_graph.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"

C_RAW = "#1F4E79"
C_FEAT = "#2E6F62"
C_DATA = "#3C6E47"
C_SYNTH = "#D98324"
C_MODEL = "#4A6628"
C_OUT = "#6B3FA0"
C_AUDIT = "#8C2D2D"
EDGE = "#777777"


def _box(ax, x, y, text, color, w=0.205, h=0.085, fs=8.0):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=color, edgecolor="white", linewidth=1.2,
                                alpha=0.95, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold", zorder=4, linespacing=1.15)


def main():
    branches = [
        ("Raw signals\nrotational / axial (50 kHz)", C_RAW),
        ("Full-contact segmentation\n(impact + tail, not just peak)", C_RAW),
        ("Features: time / freq /\nwavelet / cumulative / fusion", C_FEAT),
        ("Feature selection\nKendall / Spearman / MMI", C_FEAT),
        ("Canonical datasets\n(official VB, target-aware)", C_DATA),
        ("Synthetic research GATE\n(physics-informed, gated)", C_SYNTH),
        ("Baselines: ML / time-aware /\nTaylor-life-informed", C_MODEL),
        ("PINN ablation\n(data + mono + rate)", C_MODEL),
        ("VB prediction first\nVB(t,x) µm", C_MODEL),
        ("Health Index\nHI(t) / DI(t)", C_OUT),
        ("RUL by threshold crossing\n(220/250/300 µm)", C_OUT),
        ("SHAP audit\n(post-training, no leakage)", C_AUDIT),
        ("Anti-leakage validation\nLOEO now / LOTO later", C_AUDIT),
    ]
    fig, ax = plt.subplots(figsize=(13.6, 10.0), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cx, cy = 0.5, 0.52
    rx, ry = 0.38, 0.38
    n = len(branches)
    pos = []
    for i, (label, color) in enumerate(branches):
        ang = np.pi / 2 - 2 * np.pi * i / n
        x = cx + rx * np.cos(ang)
        y = cy + ry * np.sin(ang)
        pos.append((x, y))
        ax.add_patch(FancyArrowPatch((cx, cy), (x, y), arrowstyle="-",
                                     color=EDGE, linewidth=1.3, alpha=0.5, zorder=1))
    # sequence ring 1->...->13
    for i in range(n - 1):
        x0, y0 = pos[i]
        x1, y1 = pos[i + 1]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     connectionstyle="arc3,rad=0.22",
                                     arrowstyle="-|>", mutation_scale=9,
                                     color="#B0B0B0", linewidth=0.9, alpha=0.7, zorder=1))
    for (label, color), (x, y) in zip(branches, pos):
        _box(ax, x, y, label, color, w=0.19, h=0.078, fs=7.7)

    ax.add_patch(FancyBboxPatch((cx - 0.165, cy - 0.082), 0.33, 0.164,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor="#27496D", edgecolor="#142B40",
                                linewidth=2.4, zorder=5))
    ax.text(cx, cy, "PINN-based VB estimation\nand RUL derivation\nfor CNC tool wear",
            ha="center", va="center", fontsize=11.5, color="white",
            fontweight="bold", zorder=6, linespacing=1.3)

    ax.set_title("Final integrated PHM pipeline — official VB target (microscope_vb.csv); "
                 "predict VB first, derive RUL by threshold;\nsynthetic generation gated; "
                 "naive augmentation = Family-1 control; claims reserved for real unseen tools",
                 fontsize=11.5, pad=12)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=11,
                          markerfacecolor=c, markeredgecolor="none", label=l)
               for l, c in [("Raw / segmentation", C_RAW), ("Features / selection", C_FEAT),
                            ("Datasets", C_DATA), ("Synthetic gate", C_SYNTH),
                            ("Baselines / PINN / VB", C_MODEL), ("HI / RUL", C_OUT),
                            ("Audit / validation", C_AUDIT)]]
    ax.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
              fontsize=8.5, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    out = OUT / "final_integrated_phm_pipeline_graph.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
