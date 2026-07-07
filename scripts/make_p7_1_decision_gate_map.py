"""
make_p7_1_decision_gate_map.py — P7.1 decision-gate map (2026-06-12).

Generates outputs/figures/p7_1_decision_gate_map.png:
P7.0 inventory gate CLOSED -> P7.1 decision support ACTIVE -> synthetic generation BLOCKED,
with the 9 blocking decisions and the priority question highlighted.

Run:  python scripts/make_p7_1_decision_gate_map.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"

C_CLOSED = "#4A6628"   # done / closed
C_ACTIVE = "#1F4E79"   # active phase
C_BLOCK = "#8C2D2D"    # blocked
C_PRIO = "#B3541E"     # priority decision
EDGE = "#777777"


def _box(ax, x, y, text, color, w, h, fs=8.6, ec="white", lw=1.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=color, edgecolor=ec, linewidth=lw,
                                alpha=0.95, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold", zorder=4, linespacing=1.2)


def main():
    fig, ax = plt.subplots(figsize=(14.0, 8.6), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # priority banner
    _box(ax, 0.5, 0.93,
         "PRIORITY DECISION:  Which measurement is the official target — VB (vb_targets.csv),\n"
         "VS (microscope_vs.csv), or are they two different wear measurements?",
         C_PRIO, w=0.86, h=0.10, fs=10.5, ec="#5E2C0D", lw=2)

    # column 1: P7.0 closed
    _box(ax, 0.16, 0.74, "P7.0 — INVENTORY GATE\nCLOSED ✓", C_CLOSED, w=0.24, h=0.09, fs=10)
    items_closed = [
        "504 files / 2.77 GB inventoried",
        "Provisional manifest (12 exps)",
        "71-72 = performed, not recorded",
        "fs = 50 kHz measured ×136 files",
        "Target conflict documented",
        "Leakage contract for generators",
    ]
    for i, t in enumerate(items_closed):
        _box(ax, 0.16, 0.62 - i * 0.075, t, C_CLOSED, w=0.27, h=0.06, fs=8.0)

    # column 2: P7.1 active
    _box(ax, 0.50, 0.74, "P7.1 — DECISION GATE SUPPORT\nACTIVE", C_ACTIVE, w=0.27, h=0.09, fs=10)
    items_active = [
        "Supervisor decision packet (9 decisions)",
        "Message ready to send (ES)",
        "Exp-77 sensitivity DONE:\nenergy totals −20% biased; rms/ratios robust",
        "64 imputed p5/p6 cells: none reliable",
        "Generator contract (gate-enforced,\nnever executed)",
    ]
    heights = [0.06, 0.06, 0.085, 0.06, 0.085]
    y = 0.62
    for t, h in zip(items_active, heights):
        _box(ax, 0.50, y, t, C_ACTIVE, w=0.30, h=h, fs=8.0)
        y -= h + 0.018

    # column 3: blocked
    _box(ax, 0.84, 0.74, "SYNTHETIC GENERATION\nBLOCKED (gate Level 2)", C_BLOCK, w=0.26, h=0.09, fs=10)
    items_blocked = [
        "Official target VB vs VS (Q1-Q3)",
        "Provenance own/adapted (Q7)",
        "Exp-77 policy: recover p5/p6? (Q5)",
        "Formal fs + Value units (Q6, Q10)",
        "Photos + cutting conditions (Q8-Q9)\n(paper §2 only)",
    ]
    heights_b = [0.06, 0.06, 0.06, 0.06, 0.085]
    y = 0.62
    for t, h in zip(items_blocked, heights_b):
        _box(ax, 0.84, y, t, C_BLOCK, w=0.27, h=h, fs=8.0)
        y -= h + 0.018

    for x0, x1 in [(0.16, 0.50), (0.50, 0.84)]:
        ax.add_patch(FancyArrowPatch((x0, 0.74), (x1, 0.74),
                                     arrowstyle="-|>", mutation_scale=20,
                                     color=EDGE, linewidth=1.8, shrinkA=88, shrinkB=92))

    ax.text(0.5, 0.085,
            "Allowed meanwhile: QA, inventories, sensitivity diagnostics, contracts without execution, Graphify maps.\n"
            "Forbidden: synthetic datasets, training with synthetics, PINN-improvement claims, paper rewrite assuming targets confirmed.",
            ha="center", fontsize=9, color="#444444", style="italic")

    ax.set_title("P7.1 — Decision gate map: inventory closed, decision support active, synthetic generation blocked",
                 fontsize=12.5, pad=10)
    fig.tight_layout()
    out = OUT / "p7_1_decision_gate_map.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
