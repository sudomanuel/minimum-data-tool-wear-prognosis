"""
make_p7_3_vb_source_lock.py — P7.3 official VB source lock figure (2026-06-13).

Generates outputs/figures/p7_3_official_vb_source_lock.png:
official VB source locked to microscope_vb.csv (VB_um); vb_targets.csv = legacy subset;
the two trajectories overlaid to make the re-baseline consequence visible.

Run:  python scripts/make_p7_3_vb_source_lock.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"

C_OFF = "#1F4E79"      # official
C_LEG = "#9C6B1E"      # legacy
C_GATE = "#4A6628"     # gate
C_NOTE = "#8C2D2D"     # warning
EDGE = "#777777"

VB_OFFICIAL = {66: 103, 67: 108, 68: 121, 69: 136, 70: 148, 71: 159, 72: 170,
               73: 176, 74: 181, 75: 189, 76: 200, 77: 212}
VB_LEGACY = {66: 85, 67: 103, 68: 119, 69: 136, 70: 150, 73: 168, 74: 190, 75: 215, 76: 245, 77: 280}


def _box(ax, x, y, text, color, w, h, fs=8.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=color, edgecolor="white", linewidth=1.2,
                                alpha=0.95, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold", zorder=4, linespacing=1.2)


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 6.6), dpi=200,
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- left: lock diagram ----
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    axL.axis("off")
    _box(axL, 0.5, 0.92, "OFFICIAL TARGET = VB  (LOCKED P7.3)", C_OFF, 0.86, 0.10, fs=11)
    _box(axL, 0.27, 0.70, "Official source\nmicroscope_vb.csv\ncol VB_um · 12 exps · 103-212 µm", C_OFF, 0.42, 0.16, fs=8.6)
    _box(axL, 0.74, 0.70, "Legacy subset\nvb_targets.csv\n10 exps · 85-280 µm · P1-P6 only", C_LEG, 0.40, 0.16, fs=8.6)
    _box(axL, 0.27, 0.46, "VS column renamed -> VB_um\nbackup: microscope_vs_legacy.csv", C_LEG, 0.42, 0.12, fs=8.2)
    _box(axL, 0.74, 0.46, "71-72: official VB exists\ntarget-only (no signals)", C_OFF, 0.40, 0.12, fs=8.2)
    _box(axL, 0.5, 0.24, "config/targets.yaml + canonical manifest LOCKED\n50 kHz · in-house lab data · exp77 = 4 contacts (flags)", C_GATE, 0.86, 0.12, fs=8.6)
    _box(axL, 0.5, 0.07, "RE-BASELINE REQUIRED: P1-P6 = historical baseline; P8+ uses official VB", C_NOTE, 0.92, 0.075, fs=8.6)
    for y0 in (0.62,):
        axL.add_patch(FancyArrowPatch((0.5, 0.87), (0.5, 0.30), arrowstyle="-", color=EDGE, lw=1.2, alpha=0.4))

    # ---- right: trajectory overlay (the consequence) ----
    eo = sorted(VB_OFFICIAL)
    axR.plot(eo, [VB_OFFICIAL[e] for e in eo], "o-", color=C_OFF, lw=2.2,
             label="official VB (microscope_vb)")
    el = sorted(VB_LEGACY)
    axR.plot(el, [VB_LEGACY[e] for e in el], "s--", color=C_LEG, lw=2.0,
             label="legacy vb_targets (P1-P6)")
    for e in (71, 72):
        axR.plot([e], [VB_OFFICIAL[e]], "o", color=C_OFF, markersize=9,
                 markerfacecolor="white", markeredgewidth=2)
    axR.axhline(300, color=C_NOTE, ls=":", lw=1.4)
    axR.text(66.2, 304, "RUL threshold 300 µm (provisional)", color=C_NOTE, fontsize=8)
    axR.annotate("end of life:\n212 vs 280 µm", xy=(77, 212), xytext=(73.0, 250),
                 fontsize=8.5, color="#333333",
                 arrowprops=dict(arrowstyle="->", color="#555555"))
    axR.text(71.5, 150, "71-72\ntarget-only", fontsize=7.8, color=C_OFF, ha="center")
    axR.set_xlabel("experiment_id")
    axR.set_ylabel("VB (µm)")
    axR.set_title("Official vs legacy VB — why a re-baseline is required", fontsize=10.5)
    axR.legend(fontsize=8.5, loc="upper left")
    axR.grid(alpha=0.25)

    fig.suptitle("P7.3 — Official VB source lock: microscope_vb.csv (VB_um) is official; "
                 "vb_targets.csv is the legacy subset", fontsize=12.5, y=1.0)
    fig.tight_layout()
    out = OUT / "p7_3_official_vb_source_lock.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
