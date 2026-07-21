# -*- coding: utf-8 -*-
"""make_wear_microscopy_figure.py — illustrative optical-microscopy panel for Section 2 (what flank
wear and catastrophic chipping actually look like). Three real inspections of one tool
(Vc 55, f 0.08, dry): early flank wear -> developed wear -> the terminal edge fracture (chipping).
Source images: outputs/figures/microscopy_src/ (X100 front-flank views; the 300 um scale bar is
burned into each frame). Output: outputs/figures/wear_microscopy.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "outputs", "figures", "microscopy_src")
OUT = os.path.join(HERE, "..", "outputs", "figures", "wear_microscopy.png")

PANELS = [
    ("micro_a_early_133um.jpg", "(a) Early flank wear", "VB ≈ 133 µm · inspection 1", "#1f5fa8"),
    ("micro_b_developed_220um.jpg", "(b) Developed flank wear", "VB ≈ 220 µm · inspection 5", "#c87a00"),
    ("micro_c_chipping.jpg", "(c) Catastrophic chipping", "end of life · final inspection", "#b03030"),
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.2))
    for ax, (fn, head, sub, col) in zip(axes, PANELS):
        im = Image.open(os.path.join(SRC, fn))
        w, h = im.size
        # top-aligned SQUARE crop: keeps the cutting corner (top of frame) and the action,
        # trims the wide black margins -> less flattened panels
        # the cutting corner sits at the LEFT of these panoramas; keep it plus context,
        # in a 4:3 window so the three panels share one aspect and none looks flattened
        import numpy as _np
        arr = _np.asarray(im.convert("L"))
        # a letterbox band may cover only PART of a row, so test the dark FRACTION, not the mean
        dark_frac = (arr < 28).mean(axis=1)
        rows_ok = _np.where(dark_frac < 0.30)[0]
        y0, y1 = (int(rows_ok[0]), int(rows_ok[-1]) + 1) if len(rows_ok) else (0, h)
        im = im.crop((0, y0, w, y1)); h = y1 - y0
        side_h = h
        side_w = int(side_h * 4 / 3)
        x0 = 0 if side_w >= w else max(0, int(w * 0.02))
        im = im.crop((x0, 0, min(w, x0 + side_w), side_h))
        ax.imshow(im); ax.axis("off")
        ax.set_title(head, fontsize=12.5, fontweight="bold", color=col, pad=6)
        ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=11, color="#222222")
    # arrow to the fracture on panel (c)
    axc = axes[2]
    axc.annotate("", xy=(0.30, 0.66), xytext=(0.60, 0.30), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#b03030", lw=2.2))
    axc.text(0.63, 0.26, "fractured edge", transform=axc.transAxes, ha="center", va="top",
             fontsize=10.5, color="#b03030", fontweight="bold")
    fig.text(0.5, 0.02, "same cutting edge (Vc 55 m/min, f 0.08 mm/rev, dry) at three inspections of its life · "
             "Keyence VHX-7000 optical microscope; the 300 µm scale bar is burned into each frame",
             ha="center", fontsize=10, color="#5a6b7b", style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    from PIL import Image as I
    print("wrote", OUT, I.open(OUT).size)


if __name__ == "__main__":
    main()
