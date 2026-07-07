"""
make_p7_0_maps.py — P7.0 inventory-truth figures (2026-06-12).

Generates:
  outputs/figures/p7_0_repository_data_map.png   (repo lineage with measured counts)
  outputs/figures/p7_0_t01_data_truth_map.png    (central T01 Data Truth Mapping + 10 branches)

Run:  python scripts/make_p7_0_maps.py
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

C_RAW = "#1F4E79"      # raw data
C_FEAT = "#2E6F62"     # features / processing
C_TGT = "#8C2D2D"      # targets (conflict)
C_MODEL = "#4A6628"    # models / results
C_DOC = "#6B3FA0"      # reports / paper / deck
C_GATE = "#B3541E"     # gates / blocked
EDGE = "#777777"


def _box(ax, x, y, text, color, w=0.20, h=0.10, fs=8.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=color, edgecolor="white", linewidth=1.2,
                                alpha=0.95, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold", zorder=4, linespacing=1.18)


def _arrow(ax, p0, p1, rad=0.0, color=EDGE, lw=1.6, shrink=30):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=18,
                                 color=color, linewidth=lw,
                                 shrinkA=shrink, shrinkB=shrink, zorder=1))


def repository_data_map():
    fig, ax = plt.subplots(figsize=(14.5, 8.2), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # row 1: raw layer
    _box(ax, 0.13, 0.84, "RAW SIGNALS\n136 files · 2.7 GB · 50 kHz\nunits undeclared", C_RAW, w=0.21, h=0.13)
    _box(ax, 0.40, 0.84, "segments/\n116 of 120 expected\n(A/R 77 p5-p6 missing)", C_RAW, w=0.20, h=0.13)
    _box(ax, 0.65, 0.84, "full_signals/\n20 files · 101-181 s", C_RAW, w=0.17, h=0.13)
    _box(ax, 0.88, 0.84, "acoustic/\nEMPTY (README\nclaims AE data)", C_GATE, w=0.16, h=0.13, fs=8.0)

    # row 2: features
    _box(ax, 0.26, 0.60, "contact_features.csv\n120 rows (4 NaN placeholders\nfor exp-77 p5/p6)", C_FEAT, w=0.23, h=0.13)
    _box(ax, 0.56, 0.60, "experiment_features.csv\n10 × 208 (207 numeric)\n64 cols NaN at exp 77", C_FEAT, w=0.24, h=0.13)

    # row 2b: targets
    _box(ax, 0.86, 0.60, "TARGETS — CONFLICT\nvb_targets: 10 exp, 85-280 (code)\nmicroscope_vs: 12 exp, 103-212 (unused)", C_TGT, w=0.25, h=0.13, fs=7.8)

    # row 3: models
    _box(ax, 0.30, 0.36, "MODELS (LOEO)\nP1 benchmark 19.07 · P2 Poly2(t) 4.96\nP3 PINN_mono 18.57 · P4 RUL · P5 UQ", C_MODEL, w=0.30, h=0.13, fs=8.2)
    _box(ax, 0.70, 0.36, "QA & AUDIT\nleakage 9 checks PASS\nsignal QA 136 files · P7.0 inventories", C_MODEL, w=0.26, h=0.13, fs=8.2)

    # row 4: docs
    _box(ax, 0.18, 0.12, "REPORTS\n24 md (P1-P7.0)", C_DOC, w=0.16, h=0.11)
    _box(ax, 0.42, 0.12, "FIGURES\n84 png", C_DOC, w=0.13, h=0.11)
    _box(ax, 0.66, 0.12, "PAPER\nIEEE light_v6 · 21 tex", C_DOC, w=0.17, h=0.11)
    _box(ax, 0.88, 0.12, "DECK\n7 actos ES\n(UNCOMMITTED)", C_GATE, w=0.15, h=0.11, fs=8.0)

    _arrow(ax, (0.13, 0.84), (0.40, 0.84), shrink=42)
    _arrow(ax, (0.40, 0.84), (0.26, 0.60), rad=0.15, shrink=40)
    _arrow(ax, (0.65, 0.84), (0.56, 0.60), rad=0.10, shrink=40)
    _arrow(ax, (0.26, 0.60), (0.56, 0.60), shrink=46)
    _arrow(ax, (0.86, 0.60), (0.56, 0.60), rad=-0.12, shrink=48)
    _arrow(ax, (0.56, 0.60), (0.30, 0.36), rad=0.12, shrink=42)
    _arrow(ax, (0.56, 0.60), (0.70, 0.36), rad=-0.10, shrink=42)
    _arrow(ax, (0.30, 0.36), (0.18, 0.12), rad=0.10, shrink=38)
    _arrow(ax, (0.30, 0.36), (0.42, 0.12), rad=-0.05, shrink=38)
    _arrow(ax, (0.30, 0.36), (0.66, 0.12), rad=-0.15, shrink=40)
    _arrow(ax, (0.66, 0.12), (0.88, 0.12), shrink=34)

    ax.set_title("P7.0 repository & data lineage map — measured inventory truth (504 files, 2.77 GB; "
                 "red boxes = inconsistencies/risks)", fontsize=12, pad=12)
    fig.tight_layout()
    out = OUT / "p7_0_repository_data_map.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def t01_data_truth_map():
    branches = [
        ("Physical experiments\n12 performed (66-77)", C_RAW),
        ("Recorded signals\n10 experiments · 136 files", C_RAW),
        ("Missing acquisitions\n71-72: PERFORMED BUT\nNOT RECORDED (lab)", C_GATE),
        ("VB targets\n10 exps · 85-280 µm\n(used by all code)", C_TGT),
        ("VS measurements\n12 exps · 103-212 µm\n(meaning unconfirmed)", C_TGT),
        ("Features\n207 numeric · p>>n\n64 cols imputed @77", C_FEAT),
        ("Model-ready rows\n10 (LOEO)", C_MODEL),
        ("Trajectory-only rows\n71-72 conditional on\nofficial target", C_MODEL),
        ("Blocked decisions\nVB vs VS · provenance\nexp-77 policy", C_GATE),
        ("Synthetic generation gate\nLevel 1 CLOSED\nLevel 2 OPEN (lab)", C_GATE),
    ]
    fig, ax = plt.subplots(figsize=(13.0, 9.6), dpi=200)
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
        _box(ax, x, y, label, color, w=0.21, h=0.10, fs=8.2)

    ax.add_patch(FancyBboxPatch((cx - 0.14, cy - 0.07), 0.28, 0.14,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor="#27496D", edgecolor="#142B40",
                                linewidth=2.2, zorder=5))
    ax.text(cx, cy, "T01 Data Truth\nMapping", ha="center", va="center",
            fontsize=14, color="white", fontweight="bold", zorder=6, linespacing=1.3)

    ax.set_title("P7.0 — T01 data truth map: 12 experiments performed, 10 recorded; "
                 "71-72 performed but not recorded;\ntarget officiality and provenance still blocked (gate Level 2)",
                 fontsize=12, pad=14)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=11,
                          markerfacecolor=c, markeredgecolor="none", label=l)
               for l, c in [("Raw / experiments", C_RAW), ("Targets (conflict)", C_TGT),
                            ("Features", C_FEAT), ("Model rows", C_MODEL),
                            ("Blocked / gates", C_GATE)]]
    ax.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
              fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    out = OUT / "p7_0_t01_data_truth_map.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    repository_data_map()
    t01_data_truth_map()
