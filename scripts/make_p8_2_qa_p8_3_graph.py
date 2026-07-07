"""
make_p8_2_qa_p8_3_graph.py — P8.2B flow figure: per-contact files -> active-window refinement
-> multi-domain features -> QA -> P8.3 separated branches. (2026-06-13)

Run:  python scripts/make_p8_2_qa_p8_3_graph.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"

C_RAW = "#1F4E79"
C_SEG = "#2E6F62"
C_FEAT = "#3C6E47"
C_QA = "#9C6B1E"
C_SENSOR = "#4A6628"
C_TIME = "#8C2D2D"
C_PINN = "#6B3FA0"
EDGE = "#777777"


def _box(ax, x, y, t, c, w, h, fs=8.4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=c, edgecolor="white", linewidth=1.2, alpha=0.95, zorder=3))
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, color="white",
            fontweight="bold", zorder=4, linespacing=1.15)


def _arrow(ax, p0, p1, rad=0.0, sh=30):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=15, color=EDGE,
                                 linewidth=1.6, shrinkA=sh, shrinkB=sh, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(14.5, 8.4), dpi=200)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # top flow row
    _box(ax, 0.11, 0.86, "Per-contact files\n116 (axial+rotational)\nNOT moved", C_RAW, 0.18, 0.12)
    _box(ax, 0.345, 0.86, "Active-window refinement\nINSIDE each contact\n(envelope+adaptive thr)", C_SEG, 0.20, 0.12)
    _box(ax, 0.59, 0.86, "Multi-domain features\n51/segment: time+freq+wavelet\n326 exp-level cols", C_FEAT, 0.21, 0.12)
    _box(ax, 0.85, 0.86, "Feature QA\n0 NaN / 0 constant\n116/116 valid", C_QA, 0.18, 0.12)
    _arrow(ax, (0.11, 0.86), (0.345, 0.86), sh=66)
    _arrow(ax, (0.345, 0.86), (0.59, 0.86), sh=72)
    _arrow(ax, (0.59, 0.86), (0.85, 0.86), sh=68)

    # QA finding banner
    _box(ax, 0.5, 0.65,
         "QA FINDING: cumulative/degradation features correlate ~0.99 with VB = TIME DISGUISED;\n"
         "best genuine sensor feature ~0.88 -> P8.3 must keep sensor and time-like in SEPARATE branches",
         C_TIME, 0.92, 0.11, fs=9.5)
    _arrow(ax, (0.85, 0.80), (0.62, 0.71), rad=-0.1, sh=20)

    # split into the two feature groups
    _box(ax, 0.27, 0.45, "Sensor-derived (318)\nRMS, waveform length, kurtosis,\nspectral, band ratios, wavelet, A/R fusion",
         C_SENSOR, 0.42, 0.13, fs=8.4)
    _box(ax, 0.73, 0.45, "Time-like / degradation (8)\ncumulative order/count/RMS/energy\n= controlled branches only",
         C_TIME, 0.40, 0.13, fs=8.4)
    _arrow(ax, (0.40, 0.595), (0.30, 0.52), rad=0.1, sh=18)
    _arrow(ax, (0.60, 0.595), (0.70, 0.52), rad=-0.1, sh=18)

    # P8.3 branches
    branches = [
        (0.105, "B0 time-only\n(control 3.1)", C_TIME),
        (0.305, "B1 sensor-only\n(real claim)", C_SENSOR),
        (0.5, "B2 sensor +\ncontrolled time", C_FEAT),
        (0.695, "B3 reliability-\naware sensor", C_SENSOR),
        (0.895, "B4 PINN-ready\nminimal physics", C_PINN),
    ]
    for x, t, c in branches:
        _box(ax, x, 0.22, t, c, 0.175, 0.10, fs=7.8)
        _arrow(ax, (0.27 if x < 0.5 else 0.73, 0.385), (x, 0.27), rad=0.0, sh=20)
    _box(ax, 0.5, 0.06, "P8.3 — Kendall/Spearman/MMI selection (fold-safe) -> baselines + PINN ablation (official VB)  [NOT run yet]",
         C_PINN, 0.92, 0.065, fs=9)
    for x, _, _ in branches:
        _arrow(ax, (x, 0.17), (0.5, 0.095), rad=0.0, sh=14)

    ax.set_title("P8.2B feature QA + P8.3 plan — active-window refinement (not raw re-segmentation); "
                 "sensor vs time-like kept separate",
                 fontsize=12, pad=10)
    fig.tight_layout()
    out = OUT / "p8_2_feature_qa_and_p8_3_plan_graph.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
