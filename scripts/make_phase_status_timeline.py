"""
make_phase_status_timeline.py — project phase-status timeline (2026-06-13).
Visual, didactic map of the 16-phase framework with DONE / PARTIAL / TODO / GATED / FUTURE.

Run:  python scripts/make_phase_status_timeline.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"

STATUS_C = {"DONE": "#2E6F62", "PARTIAL": "#9C6B1E", "TODO": "#6B6B6B",
            "GATED": "#8C2D2D", "FUTURE": "#6B3FA0"}

PHASES = [
    ("0 · Project state audit", "DONE"),
    ("1 · Data QA & manifest", "DONE"),
    ("2 · Full-contact segmentation", "DONE"),
    ("3 · Multi-domain features", "DONE"),
    ("4 · Sensor branches A/R/Fusion", "PARTIAL"),
    ("5 · Selection K/S/MMI · SHAP · consensus", "PARTIAL"),
    ("6 · Classical baselines R0/R1/R2", "PARTIAL"),
    ("7 · Fold-safe augmentation", "PARTIAL"),
    ("8 · Time-aware + Taylor-life", "PARTIAL"),
    ("9 · Synthetic generation", "GATED"),
    ("10 · PINN ablation P0–P5", "PARTIAL"),
    ("11 · VB → HI → RUL", "PARTIAL"),
    ("12 · SHAP final audit", "TODO"),
    ("13 · Graphify per run", "DONE"),
    ("14 · Run ledger & Git", "PARTIAL"),
    ("15 · Real-lab LOTO validation", "FUTURE"),
]


def main():
    n = len(PHASES)
    fig, ax = plt.subplots(figsize=(12.5, 9.0), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.975, "PHM Framework — Phase Status",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.945,
            "Physics-informed synthetic-ready PHM framework for VB → HI → RUL "
            "under extreme single-tool data scarcity",
            ha="center", fontsize=9.5, style="italic", color="#333")

    y0, dy = 0.90, 0.054
    for i, (label, st) in enumerate(PHASES):
        y = y0 - i * dy
        c = STATUS_C[st]
        ax.add_patch(FancyBboxPatch((0.06, y - 0.021), 0.66, 0.042,
                                    boxstyle="round,pad=0.004,rounding_size=0.01",
                                    facecolor="#F2F2F2", edgecolor="#DDDDDD", zorder=1))
        ax.add_patch(FancyBboxPatch((0.06, y - 0.021), 0.012, 0.042,
                                    boxstyle="square,pad=0", facecolor=c, edgecolor="none", zorder=2))
        ax.text(0.09, y, label, va="center", fontsize=9.3, color="#1a1a1a", zorder=3)
        ax.add_patch(FancyBboxPatch((0.74, y - 0.018), 0.135, 0.036,
                                    boxstyle="round,pad=0.003,rounding_size=0.01",
                                    facecolor=c, edgecolor="white", zorder=2))
        ax.text(0.8075, y, st, va="center", ha="center", fontsize=8,
                color="white", fontweight="bold", zorder=3)

    # legend
    lx = 0.9
    ax.text(lx, 0.90, "Legend", fontsize=9, fontweight="bold", ha="left")
    for j, (k, c) in enumerate(STATUS_C.items()):
        yy = 0.86 - j * 0.045
        ax.add_patch(FancyBboxPatch((lx, yy - 0.013), 0.026, 0.026, boxstyle="square,pad=0",
                                    facecolor=c, edgecolor="none"))
        ax.text(lx + 0.035, yy, k, va="center", fontsize=8)

    ax.text(0.5, 0.045,
            "Current bar: Linear(t) 3.10 µm  ·  PINN_mono 6.65 µm  ·  sensor-only ~26 µm  "
            "(single-tool temporal degeneracy)\n"
            "Synthetic gate CLOSED  ·  claims reserved for real multi-tool LOTO validation",
            ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    out = OUT / "phase_status_timeline.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
