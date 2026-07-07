"""
finalize_visual_qa.py — P6 Step 3: contact sheet + dimensiones + ZIP light v3.

1. Mueve las v2 reemplazadas por v3 a figures/originals/ (nada se borra).
2. Genera paper/figure_mapping/figure_contact_sheet_p6.png (todas las
   figuras: nombre, px, single/double, PASS/REDESIGNED/OPTIONAL).
3. Genera paper/figure_mapping/figure_dimensions_p6.csv.
4. Reconstruye el ZIP full (sincronizado) y crea
   paper/phm_pinn_paper_overleaf_integrated_light_v3.zip
   (la light anterior NO se toca).
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SKEL = (ROOT / "paper" / "overleaf_extracted" / "phm_pinn_paper_scaffold_overleaf"
        / "phm_pinn_paper_skeleton")
FIGS = SKEL / "figures"
FMAP = ROOT / "paper" / "figure_mapping"
ZIP_FULL = ROOT / "paper" / "phm_pinn_paper_overleaf_integrated.zip"
ZIP_LIGHT3 = ROOT / "paper" / "phm_pinn_paper_overleaf_integrated_light_v3.zip"

# figuras INSERTADAS en el .tex (estado final tras Step 3)
INSERTED = {
    "timeaware_comparison_v3.png":   ("single", "REDESIGNED v3"),
    "pinn_comparison_mae_v3.png":    ("single", "REDESIGNED v3"),
    "pinn_vb_curve_v3.png":          ("single", "REDESIGNED v3"),
    "benchmark_branches_v3.png":     ("single", "REDESIGNED v3"),
    "uncertainty_structural_v3.png": ("single", "REDESIGNED v3"),
    "rul_curves_v3.png":             ("double", "REDESIGNED v3"),
    "uncertainty_vb_bands_v2.png":   ("double", "PASS (v2)"),
    "pinn_physical_metrics_v2.png":  ("double", "PASS (v2)"),
    "multitool_roadmap_v2.png":      ("double", "PASS (v2)"),
}
SUPERSEDED_V2 = ["timeaware_comparison_v2.png", "pinn_comparison_mae_v2.png",
                 "pinn_vb_curve_v2.png", "benchmark_branches_v2.png",
                 "uncertainty_structural_v2.png", "rul_curves_v2.png"]
OPTIONAL = ["rul_disagreement.png", "uncertainty_tfail_dist.png"]


def main() -> int:
    # 1. mover v2 reemplazadas
    (FIGS / "originals").mkdir(exist_ok=True)
    for f in SUPERSEDED_V2:
        if (FIGS / f).exists():
            shutil.move(str(FIGS / f), str(FIGS / "originals" / f))
    print(f"[org] insertadas en figures/: {sorted(p.name for p in FIGS.glob('*.png'))}")

    # 2-3. contact sheet + dimensiones
    entries = []
    for f, (env, status) in INSERTED.items():
        entries.append((FIGS / f, env, status))
    for f in OPTIONAL:
        p = FIGS / "optional" / f
        if p.exists():
            entries.append((p, "—", "OPTIONAL (not inserted)"))

    rows, thumbs = [], []
    TW = 640
    for p, env, status in entries:
        img = Image.open(p)
        w, h = img.size
        rows.append({
            "figure_file": p.name, "width_px": w, "height_px": h,
            "aspect_ratio": round(w / h, 2),
            "file_size_kb": round(p.stat().st_size / 1024),
            "recommended_latex_width": ("\\columnwidth" if env == "single"
                                        else "0.7-0.95\\textwidth" if env == "double"
                                        else "n/a"),
            "recommended_environment": ("figure" if env == "single"
                                        else "figure*" if env == "double"
                                        else "n/a"),
            "final_status": status,
        })
        th = img.resize((TW, int(h * TW / w)))
        thumbs.append((th, f"{p.name}  |  {w}x{h}px  |  "
                       f"{'figure' if env=='single' else 'figure*' if env=='double' else 'optional'}"
                       f"  |  {status}"))

    pd.DataFrame(rows).to_csv(FMAP / "figure_dimensions_p6.csv", index=False)
    print(f"[write] {FMAP / 'figure_dimensions_p6.csv'}")

    cols = 3
    label_h = 34
    rowsn = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.height for t, _ in thumbs) + label_h + 12
    sheet = Image.new("RGB", (cols * (TW + 14) + 14, rowsn * cell_h + 14), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (th, label) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = 14 + c * (TW + 14)
        y = 14 + r * cell_h
        sheet.paste(th, (x, y))
        color = ("#1B7F5A" if "PASS" in label or "REDESIGNED" in label
                 else "#7A7A7A")
        draw.rectangle([x, y + th.height + 2, x + TW, y + th.height + label_h],
                       outline=color, width=2)
        draw.text((x + 6, y + th.height + 9), label, fill="black")
    sheet.save(FMAP / "figure_contact_sheet_p6.png", optimize=True)
    print(f"[write] {FMAP / 'figure_contact_sheet_p6.png'} ({sheet.size})")

    # 4. ZIPs
    def optimize_png(src: Path, dst: Path):
        img = Image.open(src).convert("RGB")
        img.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE)\
           .save(dst, "PNG", optimize=True)

    def build_zip(zpath: Path, light: bool):
        tmp = FIGS / "_zip_tmp"
        if light:
            tmp.mkdir(exist_ok=True)
            for f in INSERTED:
                optimize_png(FIGS / f, tmp / f)
        if zpath.exists():
            zpath.unlink()
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(SKEL.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(SKEL)
                if p.name in ("main.pdf", ".gitkeep") or rel.parts[0] == "figures":
                    continue
                zf.write(p, str(rel))
            src = tmp if light else FIGS
            for f in INSERTED:
                zf.write(src / f, f"figures/{f}")
        if light:
            shutil.rmtree(tmp)
        print(f"[zip] {zpath.name}: {zpath.stat().st_size/1024:.0f} KB")

    build_zip(ZIP_FULL, light=False)
    build_zip(ZIP_LIGHT3, light=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
