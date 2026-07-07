"""
make_p7_2_lab_confirmed_map.py — P7.2 lab-confirmed data-truth map (2026-06-13).

Generates outputs/figures/p7_2_lab_confirmed_data_truth_map.png:
central "T01 Data Truth — LAB CONFIRMED" + branches showing every confirmation,
with the synthetic gate now "ready for controlled opening".

Run:  python scripts/make_p7_2_lab_confirmed_map.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"

C_CONF = "#2E6F62"     # confirmed
C_TARGET = "#1F4E79"   # target
C_LEGACY = "#7A6A2F"   # legacy/auxiliary
C_PARTIAL = "#9C6B1E"  # partial / caveat
C_OPEN = "#4A6628"     # gate ready
EDGE = "#777777"


def _box(ax, x, y, text, color, w=0.205, h=0.10, fs=8.4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=color, edgecolor="white", linewidth=1.2,
                                alpha=0.95, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold", zorder=4, linespacing=1.18)


def main():
    branches = [
        ("Official target = VB\nCONFIRMED", C_TARGET),
        ("\"VS\" = misnamed/legacy\nNOT a variable (auxiliary)", C_LEGACY),
        ("VB source reconciliation\n(confirm authoritative file)", C_LEGACY),
        ("71-72 performed,\nsignals not recorded\n(wear results exist)", C_PARTIAL),
        ("71-72 = target-only\n(trajectory analysis only)", C_PARTIAL),
        ("Exp 77 = 4 segmentable\ncontacts (p5/p6 not\nrecoverable, no imputation)", C_PARTIAL),
        ("Exp 77 reliability:\nrms OK / energy LOW", C_PARTIAL),
        ("Sampling = 50 kHz\nCONFIRMED", C_CONF),
        ("Origin = in-house\nlaboratory data CONFIRMED", C_CONF),
        ("Synthetic gate:\nREADY for controlled\nopening (flag pending)", C_OPEN),
    ]
    fig, ax = plt.subplots(figsize=(13.2, 9.8), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cx, cy = 0.5, 0.52
    rx, ry = 0.375, 0.375
    n = len(branches)
    for i, (label, color) in enumerate(branches):
        ang = np.pi / 2 - 2 * np.pi * i / n
        x = cx + rx * np.cos(ang)
        y = cy + ry * np.sin(ang)
        ax.add_patch(FancyArrowPatch((cx, cy), (x, y), arrowstyle="-",
                                     color=EDGE, linewidth=1.3, alpha=0.55, zorder=1))
        _box(ax, x, y, label, color, w=0.215, h=0.105, fs=8.0)

    ax.add_patch(FancyBboxPatch((cx - 0.15, cy - 0.075), 0.30, 0.15,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor="#27496D", edgecolor="#142B40",
                                linewidth=2.4, zorder=5))
    ax.text(cx, cy, "T01 Data Truth\nLAB CONFIRMED\n(2026-06-13)", ha="center", va="center",
            fontsize=13, color="white", fontweight="bold", zorder=6, linespacing=1.3)

    ax.set_title("P7.2 — T01 data truth, lab-confirmed: target = VB, in-house data, 50 kHz; "
                 "71-72 target-only; exp 77 partial;\n\"VS\" is a legacy mislabel; synthetic gate ready for controlled opening",
                 fontsize=11.5, pad=14)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=11,
                          markerfacecolor=c, markeredgecolor="none", label=l)
               for l, c in [("Confirmed", C_CONF), ("Official target", C_TARGET),
                            ("Legacy/auxiliary", C_LEGACY), ("Partial / caveat", C_PARTIAL),
                            ("Gate ready", C_OPEN)]]
    ax.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
              fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    out = OUT / "p7_2_lab_confirmed_data_truth_map.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
