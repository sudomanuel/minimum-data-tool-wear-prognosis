"""
make_p8_2c_architecture.py — P8.2C two-branch segmentation architecture (2026-06-13).
per-contact files -> {full-contact original | active-window refined} -> QA compare -> P8.3 model
compare -> evidence-based choice.

Run:  python scripts/make_p8_2c_architecture.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"

C_RAW = "#1F4E79"
C_A = "#9C6B1E"     # full-contact original
C_B = "#2E6F62"     # active-window refined
C_QA = "#4A6628"
C_P83 = "#6B3FA0"
C_CHOICE = "#8C2D2D"
EDGE = "#777777"


def _box(ax, x, y, t, c, w, h, fs=8.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=c, edgecolor="white", linewidth=1.2, alpha=0.95, zorder=3))
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, color="white",
            fontweight="bold", zorder=4, linespacing=1.15)


def _arrow(ax, p0, p1, rad=0.0, sh=26):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=16, color=EDGE,
                                 linewidth=1.7, shrinkA=sh, shrinkB=sh, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(13.5, 7.8), dpi=200)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _box(ax, 0.5, 0.9, "Per-contact files (116, axial+rotational) — NOT moved/overwritten",
         C_RAW, 0.7, 0.09, fs=9.5)

    _box(ax, 0.26, 0.66, "Branch A — full_contact_original\nfeatures over the WHOLE file\n(existing project segmentation)",
         C_A, 0.40, 0.14, fs=8.6)
    _box(ax, 0.74, 0.66, "Branch B — active_window_refined\nfeatures over [start:end] active window\n(envelope+threshold+margins; ~89% kept)",
         C_B, 0.42, 0.14, fs=8.6)
    _arrow(ax, (0.42, 0.855), (0.30, 0.74), rad=0.12)
    _arrow(ax, (0.58, 0.855), (0.70, 0.74), rad=-0.12)

    _box(ax, 0.5, 0.42, "Feature QA comparison (P8.2C)\nRMS +4.8% | energy -0.4% | dominant_freq ~0% | both branch_candidate",
         C_QA, 0.74, 0.11, fs=8.8)
    _arrow(ax, (0.26, 0.59), (0.40, 0.475), rad=-0.1)
    _arrow(ax, (0.74, 0.59), (0.60, 0.475), rad=0.1)

    _box(ax, 0.5, 0.22, "P8.3 model comparison: segmentation_source x {B0..B4 feature branches}, leakage-safe LOEO",
         C_P83, 0.82, 0.085, fs=8.8)
    _arrow(ax, (0.5, 0.365), (0.5, 0.265), sh=8)

    _box(ax, 0.5, 0.06, "Evidence-based choice of segmentation source (NOT decided in P8.2)",
         C_CHOICE, 0.66, 0.075, fs=9)
    _arrow(ax, (0.5, 0.175), (0.5, 0.10), sh=8)

    ax.set_title("P8.2C — two segmentation/feature branches kept as candidates; "
                 "the source is chosen by performance in P8.3, not asserted in P8.2",
                 fontsize=11.5, pad=10)
    fig.tight_layout()
    out = OUT / "p8_2c_full_vs_active_window_architecture.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
