"""
build_paper_zips.py — reorganiza figuras y construye los dos ZIPs del paper.

1. figures/originals/  <- figuras v1 (espanol, trabajo) — preservadas.
2. figures/optional/   <- figuras no insertadas en el paper.
3. figures/*.png       <- SOLO las 9 v2 insertadas (full-res).
4. paper/phm_pinn_paper_overleaf_integrated.zip       (full-res v2)
5. paper/phm_pinn_paper_overleaf_integrated_light.zip (v2 optimizadas: PNG
   paleta 256 colores + optimize -> ~50-70% mas ligeras, sin perdida legible)
Reporta tamanos antes/despues por imagen.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SKEL = (ROOT / "paper" / "overleaf_extracted" / "phm_pinn_paper_scaffold_overleaf"
        / "phm_pinn_paper_skeleton")
FIGS = SKEL / "figures"
ZIP_FULL = ROOT / "paper" / "phm_pinn_paper_overleaf_integrated.zip"
ZIP_LIGHT = ROOT / "paper" / "phm_pinn_paper_overleaf_integrated_light.zip"

V2 = ["benchmark_branches_v2.png", "timeaware_comparison_v2.png",
      "pinn_comparison_mae_v2.png", "pinn_physical_metrics_v2.png",
      "pinn_vb_curve_v2.png", "rul_curves_v2.png",
      "uncertainty_vb_bands_v2.png", "uncertainty_structural_v2.png",
      "multitool_roadmap_v2.png"]
OPTIONAL = ["rul_disagreement.png", "uncertainty_tfail_dist.png"]
V1_TO_ORIGINALS = ["benchmark_branches.png", "timeaware_comparison.png",
                   "pinn_comparison_mae.png", "pinn_physical_metrics.png",
                   "pinn_vb_curve.png", "rul_curves.png",
                   "uncertainty_vb_bands.png", "uncertainty_structural.png",
                   "multitool_roadmap.png"]


def optimize_png(src: Path, dst: Path) -> tuple[int, int]:
    """Cuantiza a paleta 256 colores + optimize. Devuelve (antes, despues)."""
    before = src.stat().st_size
    img = Image.open(src).convert("RGB")
    img_q = img.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img_q.save(dst, "PNG", optimize=True)
    return before, dst.stat().st_size


def add_tree(zf: zipfile.ZipFile, base: Path, figdir: Path):
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        parts = rel.parts
        if p.name in ("main.pdf", ".gitkeep"):
            continue
        if parts[0] == "figures":
            continue  # las figuras se anaden aparte (full vs light)
        zf.write(p, str(rel))
    for f in V2:
        zf.write(figdir / f, f"figures/{f}")


def main() -> int:
    # 1-2. reorganizar
    (FIGS / "originals").mkdir(exist_ok=True)
    (FIGS / "optional").mkdir(exist_ok=True)
    for f in V1_TO_ORIGINALS:
        if (FIGS / f).exists():
            shutil.move(str(FIGS / f), str(FIGS / "originals" / f))
    for f in OPTIONAL:
        if (FIGS / f).exists():
            shutil.move(str(FIGS / f), str(FIGS / "optional" / f))
    print("[org] originals/:", len(list((FIGS / 'originals').glob('*.png'))),
          "| optional/:", len(list((FIGS / 'optional').glob('*.png'))),
          "| insertadas:", len(list(FIGS.glob('*.png'))))

    # 3. optimizar para light
    light_dir = FIGS / "_light_tmp"
    light_dir.mkdir(exist_ok=True)
    rows = []
    for f in V2:
        b, a = optimize_png(FIGS / f, light_dir / f)
        rows.append((f, b, a))
        print(f"[opt] {f:36s} {b/1024:7.0f} KB -> {a/1024:6.0f} KB "
              f"({100*(1-a/b):.0f}% menos)")

    # 4. ZIP full
    for z in (ZIP_FULL, ZIP_LIGHT):
        if z.exists():
            z.unlink()
    with zipfile.ZipFile(ZIP_FULL, "w", zipfile.ZIP_DEFLATED) as zf:
        add_tree(zf, SKEL, FIGS)
    # 5. ZIP light
    with zipfile.ZipFile(ZIP_LIGHT, "w", zipfile.ZIP_DEFLATED) as zf:
        add_tree(zf, SKEL, light_dir)
    shutil.rmtree(light_dir)

    print(f"[zip] full : {ZIP_FULL.name}  {ZIP_FULL.stat().st_size/1024:.0f} KB")
    print(f"[zip] light: {ZIP_LIGHT.name}  {ZIP_LIGHT.stat().st_size/1024:.0f} KB")
    total_b = sum(r[1] for r in rows)
    total_a = sum(r[2] for r in rows)
    print(f"[opt] figuras total: {total_b/1024:.0f} -> {total_a/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
