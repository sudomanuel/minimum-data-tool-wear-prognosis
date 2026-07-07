"""
arch_diagrams.py — diagramas de arquitectura en cascada (bilingue ES/EN).

Niveles (validados para NO solaparse: cada uno expande UN bloque del anterior):
  L0  arquitectura general  (superficial: 5 bloques; unico punto paralelo)
  L1a extraccion de features + dimensionalidad
  L1b benchmark (LOEO -> 36 ramas -> modelos -> ranking -> SHAP)
  L1c PINN (validacion fisica -> f(x,t)+perdida -> LOEO)
El sub-detalle de L1b (las 36 ramas) es el arbol existente
`00_layered_flow_diagram_no_holdout.png` (L2).

Solo refleja lo que ESTA en uso. Lenguaje simple.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

from .config import FIGURE_DPI, FIGURE_FORMAT

NAVY, STEEL, RUST, GREEN, GRAY, INK = (
    "#1F3A5F", "#5B7A9C", "#D9742B", "#1B7F5A", "#9AA3AF", "#1F2937")
PAL = {
    "data":   dict(fc="#DCE4F0", ec=NAVY,  tc="#13263A"),
    "step":   dict(fc="#EEF2F8", ec=STEEL, tc="#22364B"),
    "bench":  dict(fc="#E6EDF5", ec=STEEL, tc="#22364B"),
    "phys":   dict(fc="#FBE6D4", ec=RUST,  tc="#7A3D10"),
    "report": dict(fc="#1F3A5F", ec=NAVY,  tc="#FFFFFF"),
}


def _canvas(title: str):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.add_patch(FancyBboxPatch((2.5, 3.0), 95, 92,
                 boxstyle="round,pad=0.2,rounding_size=1.4",
                 linewidth=2.2, edgecolor=NAVY, facecolor="none", zorder=1))
    ax.text(50, 97.6, title, ha="center", va="center",
            fontsize=18, weight="bold", color=INK)
    return fig, ax


def _box(ax, cx, cy, w, h, text, role="step", fs=11, dashed=False):
    p = PAL[role]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.25,rounding_size=0.8",
                 linewidth=1.8, edgecolor=p["ec"], facecolor=p["fc"], zorder=3,
                 linestyle="--" if dashed else "-"))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=p["tc"], weight="bold", zorder=4)
    return dict(top=(cx, cy + h / 2), bottom=(cx, cy - h / 2),
                left=(cx - w / 2, cy), right=(cx + w / 2, cy), cx=cx, cy=cy)


def _arrow(ax, a, b, color=STEEL, lw=2.2, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b,
                 arrowstyle="-|>,head_length=9,head_width=6",
                 linewidth=lw, color=color, shrinkA=1, shrinkB=3,
                 zorder=2, linestyle=ls))


def _line(ax, a, b, color=GRAY, lw=2.0):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
            solid_capstyle="round", zorder=1)


def _save(fig, target_dir, name):
    target_dir = Path(target_dir); target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _vchain(ax, nodes, role="step", x=50, y_top=86, y_bot=12, w=58, h=8, fs=11):
    """Cadena vertical lineal de cajas con flechas (de arriba a abajo)."""
    n = len(nodes)
    ys = [y_top - i * (y_top - y_bot) / max(1, n - 1) for i in range(n)]
    boxes = []
    for txt, yy in zip(nodes, ys):
        r = role(txt) if callable(role) else role
        boxes.append(_box(ax, x, yy, w, h, txt, r, fs=fs))
    for a, b in zip(boxes[:-1], boxes[1:]):
        _arrow(ax, a["bottom"], b["top"], color=NAVY)
    return boxes


# =============================================================================
# Textos (ES / EN)
# =============================================================================
T = {
    "L0": {
        "es": dict(title="Arquitectura general del proceso",
                   raw="Senales de vibracion\n(crudas)",
                   feat="Extraccion de features",
                   data="experiment_features.csv\n(1 fila por experimento)",
                   bench="BENCHMARK\ncomparar modelos · LOEO",
                   pinn="PINN\ninformada por fisica",
                   report="Reporte (diapositivas)",
                   par="en paralelo"),
        "en": dict(title="Overall process architecture",
                   raw="Vibration signals\n(raw)",
                   feat="Feature extraction",
                   data="experiment_features.csv\n(one row per experiment)",
                   bench="BENCHMARK\ncompare models · LOEO",
                   pinn="PINN\nphysics-informed",
                   report="Report (slides)",
                   par="in parallel"),
    },
    "FEAT": {
        "es": dict(title="Detalle — extraccion de features y dimensionalidad",
                   steps=["Senal cruda (por contacto)",
                          "Limpieza + centrado",
                          "16 descriptores  (13 tiempo + 3 frecuencia)",
                          "Agregar  →  1 fila por experimento",
                          "Dimensionalidad: quitar columnas\nmuertas/constantes y metadata",
                          "experiment_features.csv"]),
        "en": dict(title="Detail — feature extraction and dimensionality",
                   steps=["Raw signal (per contact)",
                          "Clean + center",
                          "16 descriptors  (13 time + 3 frequency)",
                          "Aggregate  →  one row per experiment",
                          "Dimensionality: drop dead/constant\ncolumns and metadata",
                          "experiment_features.csv"]),
    },
    "BENCH": {
        "es": dict(title="Detalle — benchmark (comparacion de modelos)",
                   nodes=["experiment_features.csv",
                          "Split LOEO  (10 folds, honesto)",
                          "36 ramas  (3 subsets × 2 datos × 3 tuning)",
                          "9 modelos:  Dummy · Ridge · Lasso · ElasticNet\nSVR · RandomForest · XGBoost · MLP · BNN",
                          "Ranking  (por MAE)",
                          "SHAP  (post-hoc, explica al ganador)"],
                   fan="abanico en paralelo"),
        "en": dict(title="Detail — benchmark (model comparison)",
                   nodes=["experiment_features.csv",
                          "LOEO split  (10 folds, honest)",
                          "36 branches  (3 subsets × 2 data × 3 tuning)",
                          "9 models:  Dummy · Ridge · Lasso · ElasticNet\nSVR · RandomForest · XGBoost · MLP · BNN",
                          "Ranking  (by MAE)",
                          "SHAP  (post-hoc, explains the winner)"],
                   fan="parallel fan-out"),
    },
    "PINN": {
        "es": dict(title="Detalle — PINN (informada por fisica)",
                   nodes=["experiment_features.csv  +  t (tiempo de corte)",
                          "Validacion fisica\n(monotonia rho=1.0 · tasa↔energia rot. rho=0.76)",
                          "PINN   VB = f(x, t)\nperdida = datos + lambda_mono(df/dt>=0)\n+ lambda_rate(df/dt = g(E_rot))   (autodiff)",
                          "Evaluacion LOEO  (run_pinn)"]),
        "en": dict(title="Detail — PINN (physics-informed)",
                   nodes=["experiment_features.csv  +  t (cutting time)",
                          "Physics validation\n(monotonic rho=1.0 · rate↔rot. energy rho=0.76)",
                          "PINN   VB = f(x, t)\nloss = data + lambda_mono(df/dt>=0)\n+ lambda_rate(df/dt = g(E_rot))   (autodiff)",
                          "LOEO evaluation  (run_pinn)"]),
    },
}


# =============================================================================
# L0 — arquitectura general
# =============================================================================
def plot_arch_general(target_dir, lang="es", filename=None) -> Path:
    t = T["L0"][lang]
    fig, ax = _canvas(t["title"])
    RAW = _box(ax, 50, 84, 46, 7, t["raw"], "data", fs=12)
    FEAT = _box(ax, 50, 70, 46, 7, t["feat"], "step", fs=12.5)
    DATA = _box(ax, 50, 56, 50, 7, t["data"], "data", fs=11.5)
    _arrow(ax, RAW["bottom"], FEAT["top"], color=NAVY)
    _arrow(ax, FEAT["bottom"], DATA["top"], color=NAVY)

    # split paralelo
    bus = 47.0
    _line(ax, DATA["bottom"], (50, bus), color=NAVY)
    _line(ax, (28, bus), (72, bus), color=NAVY)
    ax.text(74.5, bus, t["par"], ha="left", va="center", fontsize=10,
            style="italic", color=GRAY)
    BENCH = _box(ax, 28, 38, 38, 9, t["bench"], "bench", fs=12)
    PINN = _box(ax, 72, 38, 38, 9, t["pinn"], "phys", fs=12)
    _arrow(ax, (28, bus), BENCH["top"], color=STEEL)
    _arrow(ax, (72, bus), PINN["top"], color=RUST)

    REPORT = _box(ax, 50, 18, 44, 7, t["report"], "report", fs=12.5)
    cy = 28.5
    _line(ax, BENCH["bottom"], (28, cy), color=STEEL)
    _line(ax, PINN["bottom"], (72, cy), color=RUST)
    _line(ax, (28, cy), (72, cy), color=GRAY, lw=1.8)
    _arrow(ax, (50, cy), REPORT["top"], color=NAVY)
    return _save(fig, target_dir, filename or f"arch_general_{lang}")


# =============================================================================
# L1a — features + dimensionalidad
# =============================================================================
def plot_arch_features(target_dir, lang="es", filename=None) -> Path:
    t = T["FEAT"][lang]
    fig, ax = _canvas(t["title"])
    roles = ["data", "step", "step", "step", "step", "data"]
    nodes = t["steps"]
    n = len(nodes)
    ys = [88 - i * (88 - 12) / (n - 1) for i in range(n)]
    boxes = [_box(ax, 50, yy, 60, 8.5, txt, role, fs=11.5)
             for txt, yy, role in zip(nodes, ys, roles)]
    for a, b in zip(boxes[:-1], boxes[1:]):
        _arrow(ax, a["bottom"], b["top"], color=NAVY)
    return _save(fig, target_dir, filename or f"arch_features_{lang}")


# =============================================================================
# L1b — benchmark
# =============================================================================
def plot_arch_benchmark(target_dir, lang="es", filename=None) -> Path:
    t = T["BENCH"][lang]
    fig, ax = _canvas(t["title"])
    nodes = t["nodes"]
    roles = ["data", "bench", "bench", "bench", "bench", "phys"]
    n = len(nodes)
    ys = [88 - i * (88 - 12) / (n - 1) for i in range(n)]
    hs = [7, 7.5, 8, 9, 7, 7.5]
    boxes = [_box(ax, 50, yy, 66, hh, txt, role, fs=10.8)
             for txt, yy, hh, role in zip(nodes, ys, hs, roles)]
    for a, b in zip(boxes[:-1], boxes[1:]):
        _arrow(ax, a["bottom"], b["top"], color=STEEL)
    # nota de abanico en la rama "36 ramas"
    ax.text(85, boxes[2]["cy"], t["fan"], ha="left", va="center",
            fontsize=9.5, style="italic", color=GRAY)
    return _save(fig, target_dir, filename or f"arch_benchmark_{lang}")


# =============================================================================
# L1c — PINN
# =============================================================================
def plot_arch_pinn(target_dir, lang="es", filename=None) -> Path:
    t = T["PINN"][lang]
    fig, ax = _canvas(t["title"])
    nodes = t["nodes"]
    roles = ["data", "phys", "phys", "phys"]
    n = len(nodes)
    ys = [84 - i * (84 - 16) / (n - 1) for i in range(n)]
    hs = [7.5, 9, 12, 7.5]
    boxes = [_box(ax, 50, yy, 70, hh, txt, role, fs=11)
             for txt, yy, hh, role in zip(nodes, ys, hs, roles)]
    for a, b in zip(boxes[:-1], boxes[1:]):
        _arrow(ax, a["bottom"], b["top"], color=RUST)
    return _save(fig, target_dir, filename or f"arch_pinn_{lang}")


def generate_all(target_dir) -> list:
    """Genera los 4 diagramas en ES y EN (8 PNG)."""
    out = []
    for lang in ("es", "en"):
        out += [plot_arch_general(target_dir, lang),
                plot_arch_features(target_dir, lang),
                plot_arch_benchmark(target_dir, lang),
                plot_arch_pinn(target_dir, lang)]
    return out
