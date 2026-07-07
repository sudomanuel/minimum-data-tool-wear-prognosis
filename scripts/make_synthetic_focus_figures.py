"""
make_synthetic_focus_figures.py — strategic-refocus figures (2026-06-11).

Generates:
  outputs/figures/synthetic_focus_project_graph.png
      Central node "Research, Select, and Validate the Best Synthetic Data
      Generation Method for T01 Tool Wear Prognosis" + 14 branches.
  outputs/figures/synthetic_focus_project_roadmap.png
      End-to-end roadmap: T01 real tool -> ... -> robust predictor.

Run:  python scripts/make_synthetic_focus_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette by role
C_CENTER = "#B3541E"   # burnt orange — synthetic core
C_REAL = "#1F4E79"     # blue — real tool / experimental context
C_INGEST = "#2E6F62"   # teal — ingestion / QA
C_SYNTH = "#D98324"    # orange — synthetic research/validation
C_MODEL = "#4A6628"    # green — models / training regimes
C_OUT = "#6B3FA0"      # purple — RUL / final validation / predictor
EDGE = "#777777"


def _wrap_box(ax, x, y, text, color, w=0.205, h=0.085, fs=8.6, lw=1.2):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.008,rounding_size=0.012",
                         facecolor=color, edgecolor="white", linewidth=lw,
                         alpha=0.95, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold", zorder=4, linespacing=1.15)
    return (x, y)


def make_project_graph():
    branches = [
        ("1. T01 Real Tool &\nExperimental Context", C_REAL),
        ("2. Machine /\nLaboratory Setup", C_REAL),
        ("3. Microscope VB\nMeasurements", C_REAL),
        ("4. Tool Damage\nProgression", C_REAL),
        ("5. Vibration Data\nIngestion & QA", C_INGEST),
        ("6. Synthetic Data\nGeneration Research", C_SYNTH),
        ("7. Synthetic Data\nValidation", C_SYNTH),
        ("8. Classical\nML Models", C_MODEL),
        ("9. Tuning / Augmentation /\nSynthetic-Assisted Training", C_MODEL),
        ("10. Temporal\nStress-Test", C_MODEL),
        ("11. PINN / Physical\nNeural Network", C_MODEL),
        ("12. RUL\nDerivation", C_OUT),
        ("13. Final Real Laboratory\nTool Testing", C_OUT),
        ("14. Robust Wear &\nRUL Predictor", C_OUT),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 10.0), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cx, cy = 0.5, 0.52
    rx, ry = 0.385, 0.385
    n = len(branches)
    positions = []
    for i, (label, color) in enumerate(branches):
        ang = np.pi / 2 - 2 * np.pi * i / n  # clockwise from top
        x = cx + rx * np.cos(ang)
        y = cy + ry * np.sin(ang)
        positions.append((x, y))
        ax.add_patch(FancyArrowPatch((cx, cy), (x, y),
                                     arrowstyle="-", color=EDGE,
                                     linewidth=1.3, alpha=0.55, zorder=1))
    # sequence arrows around the ring (1->2->...->14)
    for i in range(n - 1):
        x0, y0 = positions[i]
        x1, y1 = positions[i + 1]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     connectionstyle="arc3,rad=0.25",
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="#B0B0B0", linewidth=0.9,
                                     alpha=0.7, zorder=1))
    for (label, color), (x, y) in zip(branches, positions):
        _wrap_box(ax, x, y, label, color, w=0.185, h=0.075, fs=8.0)

    # central node
    box = FancyBboxPatch((cx - 0.165, cy - 0.085), 0.33, 0.17,
                         boxstyle="round,pad=0.012,rounding_size=0.02",
                         facecolor=C_CENTER, edgecolor="#5E2C0D",
                         linewidth=2.2, zorder=5)
    ax.add_patch(box)
    ax.text(cx, cy, "Research, Select, and Validate\nthe Best Synthetic Data\nGeneration Method for\nT01 Tool Wear Prognosis",
            ha="center", va="center", fontsize=11.5, color="white",
            fontweight="bold", zorder=6, linespacing=1.3)

    ax.set_title("Synthetic data generation is the methodological core connecting limited real data,\n"
                 "the models, and the final real-laboratory validation",
                 fontsize=12.5, pad=14)
    # legend
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=11,
                          markerfacecolor=c, markeredgecolor="none", label=l)
               for l, c in [("Real tool & physical context", C_REAL),
                            ("Data ingestion & QA", C_INGEST),
                            ("Synthetic research core", C_SYNTH),
                            ("Models & training regimes", C_MODEL),
                            ("RUL & final real validation", C_OUT)]]
    ax.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
              fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    out = OUT / "synthetic_focus_project_graph.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def make_roadmap():
    steps = [
        ("T01 real tool\n(1 tool, 10 experiments)", C_REAL),
        ("Machine / setup /\nphotos", C_REAL),
        ("Microscope VB\nmeasurements (85→280 µm)", C_REAL),
        ("Data ingestion + QA\n(manifest, signal/feature QA)", C_INGEST),
        ("Synthetic generation\nmethod research (6 families)", C_SYNTH),
        ("Synthetic data validation\n(clones / physics / utility)", C_SYNTH),
        ("ML benchmark\n(real-only baseline)", C_MODEL),
        ("Tuning / augmentation /\nsynthetic-assisted training", C_MODEL),
        ("Temporal stress-test\n(Linear(t), Poly2(t))", C_MODEL),
        ("PINN / Physical\nNeural Network", C_MODEL),
        ("RUL derivation\n(threshold crossing)", C_OUT),
        ("Final real unseen\nlab tools (LOTO)", C_OUT),
        ("Robust wear & RUL\npredictor", C_OUT),
    ]
    steps = [(f"{i + 1}. {label}", color) for i, (label, color) in enumerate(steps)]
    fig, ax = plt.subplots(figsize=(14.5, 7.2), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # serpentine: 3 rows (5 / 4 / 4), left->right, right->left, left->right
    rows = [steps[0:5], steps[5:9], steps[9:13]]
    ys = [0.80, 0.50, 0.20]
    coords = []
    for r, (row, y) in enumerate(zip(rows, ys)):
        k = len(row)
        xs = np.linspace(0.10, 0.90, k)
        if r == 1:
            xs = xs[::-1]
        for (label, color), x in zip(row, xs):
            coords.append((x, y, label, color))

    for i in range(len(coords) - 1):
        x0, y0, _, _ = coords[i]
        x1, y1, _, _ = coords[i + 1]
        rad = 0.0 if abs(y0 - y1) < 1e-9 else (-0.35 if x0 > 0.5 else 0.35)
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-|>", mutation_scale=20,
                                     color=EDGE, linewidth=1.6,
                                     shrinkA=30, shrinkB=26, zorder=1))
    for x, y, label, color in coords:
        _wrap_box(ax, x, y, label, color, w=0.165, h=0.115, fs=8.4)

    # highlight the synthetic core (steps 5-6, both at x=0.90)
    ax.add_patch(FancyBboxPatch((0.795, 0.40), 0.20, 0.495,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor="none", edgecolor=C_SYNTH,
                                linewidth=1.6, linestyle=(0, (4, 3)), zorder=0))
    ax.text(0.875, 0.655, "synthetic-data\nresearch core", color=C_SYNTH,
            fontsize=8.8, ha="center", style="italic", linespacing=1.2)

    ax.set_title("Project roadmap — synthetic generation research sits between rigorous real-data ingestion\n"
                 "and every model; the final claim is reserved for real unseen laboratory tools",
                 fontsize=12.5, pad=12)
    fig.tight_layout()
    out = OUT / "synthetic_focus_project_roadmap.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_project_graph()
    make_roadmap()
