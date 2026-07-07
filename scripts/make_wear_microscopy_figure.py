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
    ("micro_a_early_133um.jpg", "(a) Early flank wear", "VB ≈ 133 µm", "#1f5fa8"),
    ("micro_b_developed_279um.jpg", "(b) Developed flank wear", "VB ≈ 279 µm", "#c87a00"),
    ("micro_c_chipping.jpg", "(c) Catastrophic chipping", "end of life", "#b03030"),
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 5.4))
    for ax, (fn, head, sub, col) in zip(axes, PANELS):
        im = Image.open(os.path.join(SRC, fn))
        w, h = im.size
        # top-aligned SQUARE crop: keeps the cutting corner (top of frame) and the action,
        # trims the wide black margins -> less flattened panels
        side = min(w, h)
        x0 = (w - side) // 2
        im = im.crop((x0, 0, x0 + side, side))
        ax.imshow(im); ax.axis("off")
        ax.set_title(head, fontsize=12.5, fontweight="bold", color=col, pad=6)
        ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=11, color="#222222")
    # arrow to the fracture on panel (c)
    axc = axes[2]
    axc.annotate("", xy=(0.66, 0.90), xytext=(0.40, 0.62), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#b03030", lw=2.2))
    axc.text(0.36, 0.55, "fractured\nedge", transform=axc.transAxes, ha="right", va="top",
             fontsize=10.5, color="#b03030", fontweight="bold")
    fig.text(0.5, 0.02, "same cutting edge (Vc 55 m/min, f 0.08 mm, dry) at three successive "
             "inspections · optical microscope; the labelled flank-wear width VB sets the scale",
             ha="center", fontsize=10, color="#5a6b7b", style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    from PIL import Image as I
    print("wrote", OUT, I.open(OUT).size)


if __name__ == "__main__":
    main()
