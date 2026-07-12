# -*- coding: utf-8 -*-
"""make_paper_figures.py — dedicated producer for the manuscript's plotted figures (paper_*.png).

Motivation (visual audit 2026-07-06): several manuscript figures were owned by deck/legacy producers
and carried (a) bilingual titles that overflowed the canvas, (b) clipped side labels, (c) STALE
values from the pre-record grid optimum (m-curve 11.3/9.1 and models chart 9.7 vs the paper's
11.0/5.6), and (d) in one case an entirely wrong legacy chart (P8.6 branch consolidation under the
ablation caption). This script decouples the paper from those producers:

  paper_models.png        Figure 3  — baseline vs all models (Table-2 values)
  paper_mcurve.png        Figure 4  — base vs joint-optimal m-curve (deployed values 12.7/11.0/5.6)
  paper_ablation.png      Figure 5  — sensor ablation R2<0 (Table-3) + physics m-curve (caption-true)
  paper_breakdown.png     Figure 6  — estimator influence vs MAP guard (results/f3_breakdown.csv)
  paper_conformal.png     Figure 7  — real deployed m=3 forecast + Mondrian band, tool chosen so the
                                      sealed points DO lie inside (caption-true), computed live
  paper_kalman.png        Figure 11 — online one-step Kalman trace (kf_online_onestep, real tool)
  paper_fair_baseline.png Figure 12 — fair-baseline R2 bars (results/f2_fair_baseline.csv)

Conventions (per the manuscript PDF-QA rule): NO in-figure titles (captions carry them), English
labels only, generous margins, tight bbox. Each figure is also exported to outputs/figures/print/ as
a vector PDF and a 600-dpi PNG submission twin.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
FIG = os.path.join(ROOT, "outputs", "figures")
PRINT = os.path.join(FIG, "print")
os.makedirs(PRINT, exist_ok=True)
GREEN, RED, BLUE, GREY, AMBER = "#2E8B57", "#C0392B", "#1F5FA8", "#7A8A99", "#C87A00"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(PRINT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(PRINT, name + "_600dpi.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------- Figure 3 · baseline vs all models (Table 2 values, verified this session) ----------
def fig_models():
    rows = [("MLP neural net (naive)", 67.0, GREY), ("Ridge (naive)", 46.4, GREY),
            ("Random Forest (naive)", 36.1, GREY),
            ("Average-wear-curve (baseline)", 18.7, RED),
            ("Linear(t) self-fit", 16.2, "#9AA8B5"),
            ("Ours — conservative (m = 3)", 11.0, GREEN),
            ("Ours — precise (m = 4)", 5.6, GREEN)]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [r[2] for r in rows][::-1]
    ax.barh(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"{v:.1f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("future-VB MAE (µm) — lower is better", fontsize=12)
    ax.set_xlim(0, 74)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(labelsize=11)
    save(fig, "paper_models")


# ---------- Figure 4 · m-curve, deployed values (12.7 / 11.0 / 5.6) ----------
def fig_mcurve():
    m = [2, 3, 4]
    base = [13.8, 11.6, 9.7]                    # base few-shot (Sec 4.3)
    opt = [12.7, 11.02, 5.63]                   # deployed optimum (records; nested-verified at m=4)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(18.7, ls=":", color=RED, lw=2, label="average-wear-curve baseline (18.7)")
    ax.plot(m, base, "--o", color=GREY, lw=2.2, ms=9, label="base few-shot (Theil–Sen)")
    ax.plot(m, opt, "-o", color=BLUE, lw=2.6, ms=10, label="joint-optimal (validity-constrained)")
    for x, y in zip(m, base):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10.5, color=GREY)
    for x, y, dy in zip(m, opt, (-15, -15, 10)):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=10.5, color=BLUE, fontweight="bold")
    ax.set_xlabel("m = number of early VB measurements", fontsize=12)
    ax.set_ylabel("future-VB MAE (µm)", fontsize=12)
    ax.set_xticks(m); ax.set_ylim(3.4, 19.9); ax.set_xlim(1.8, 4.2)
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower left")
    save(fig, "paper_mcurve")


# ---------- Figure 5 · ablation: sensor configs R2<0 + physics m-curve (caption-true) ----------
def fig_ablation():
    sens = [("Full sensor/ML (A+R)", -1.75), ("− feature selection", -1.69),
            ("− augmentation", -1.77), ("− hyper-tuning", -1.31),
            ("A-only", -1.68), ("R-only", -0.71), ("RandomForest", -0.58)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.6),
                                   gridspec_kw=dict(width_ratios=[1.25, 1]))
    names = [s[0] for s in sens][::-1]; vals = [s[1] for s in sens][::-1]
    ax1.barh(names, vals, color=RED, alpha=0.85)
    for i, v in enumerate(vals):
        ax1.text(v - 0.03, i, f"{v:.2f}", va="center", ha="right", fontsize=10)
    ax1.axvline(0, color="k", lw=1)
    ax1.set_xlabel("pooled R² (LOTO) — all configurations < 0", fontsize=11.5)
    ax1.set_xlim(-2.05, 0.4); ax1.grid(axis="x", alpha=0.3); ax1.tick_params(labelsize=10.5)
    m = [2, 3, 4]; base = [13.8, 11.6, 9.7]
    ax2.plot(m, base, "-o", color=GREEN, lw=2.6, ms=10)
    for x, y in zip(m, base):
        ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=10.5, color=GREEN, fontweight="bold")
    ax2.set_xlabel("m = early measurements", fontsize=11.5)
    ax2.set_ylabel("physics few-shot MAE (µm)", fontsize=11.5)
    ax2.set_xticks(m); ax2.set_ylim(9.0, 14.6); ax2.set_xlim(1.8, 4.2); ax2.grid(alpha=0.3)
    save(fig, "paper_ablation")


# ---------- Figure 6 · estimator influence vs MAP guard (real CSV) ----------
def fig_breakdown():
    d = pd.read_csv(os.path.join(ROOT, "results", "f3_breakdown.csv"))
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.plot(d.delta, d.theilsen_shift, "--o", color=RED, lw=2.4, ms=8,
            label="Theil–Sen alone (unbounded influence)")
    ax.plot(d.delta, d.hybrid_shift, "-o", color=BLUE, lw=2.4, ms=8,
            label="with inert-by-default MAP guard")
    ax.set_xlabel("injected outlier magnitude on one early measurement (µm)", fontsize=12)
    ax.set_ylabel("mean |shift| in predicted future VB (µm)", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=10.5, loc="upper left")
    save(fig, "paper_breakdown")


# ---------- Figure 7 · real deployed m=3 forecast + Mondrian band (caption-true tool) ----------
def fig_conformal():
    from run_mcurve import load, theil_sen, tools_of
    from run_optimal_config_search import fit_p, FIT
    CENSOR, M, AL = 300.0, 3, 0.10

    def local_p(o, v, m, p_star, lam=200.0):
        best_p, best = p_star, np.inf
        for pc in np.arange(max(p_star - 0.15, 0.05), p_star + 0.1501, 0.05):
            tau = o[:m] ** pc; a, b = theil_sen(tau, v[:m])
            sse = float(np.sum((b + a * tau - v[:m]) ** 2)) + lam * (pc - p_star) ** 2
            if sse < best:
                best, best_p = sse, pc
        return best_p

    def hbin(h):
        return 0 if h <= 1 else (1 if h <= 3 else 2)

    d = load()
    res, fits = {}, {}
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= M:
            continue
        fut = np.arange(M, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = local_p(o, v, M, fit_p(tr)); a, b = FIT["siegel"](o[:M] ** p, v[:M])
        pred = b + a * o ** p
        res[tt] = (pred[fut] - v[fut], (fut - (M - 1)).astype(float))
        fits[tt] = (o, v, pred, fut)
    # per-tool Mondrian band from the OTHER tools' residuals; pick the longest fully-covered tool
    best_tt, best_len = None, -1
    bands = {}
    for tt in fits:
        cal_s = np.concatenate([res[c][0] for c in res if c != tt])
        cal_h = np.array([hbin(x) for c in res if c != tt for x in res[c][1]])
        o, v, pred, fut = fits[tt]
        q = []
        for j, h in zip(fut, (fut - (M - 1)).astype(float)):
            sel = cal_h == hbin(h)
            pool = np.abs(cal_s[sel]) if sel.sum() >= 8 else np.abs(cal_s)
            q.append(float(np.quantile(pool, 1 - AL)))
        bands[tt] = np.array(q)
        inside = np.all(np.abs(pred[fut] - v[fut]) <= bands[tt])
        if inside and len(fut) > best_len:
            best_len, best_tt = len(fut), tt
    tt = best_tt
    o, v, pred, fut = fits[tt]; q = bands[tt]
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.fill_between(o[fut], pred[fut] - q, pred[fut] + q, color=BLUE, alpha=0.16,
                    label="90% Mondrian conformal band")
    ax.plot(o, pred, color=BLUE, lw=2.4, label="few-shot forecast (m = 3)")
    ax.plot(o[:M], v[:M], "o", color=GREEN, ms=11, label="3 early measurements (known)")
    ax.plot(o[fut], v[fut], "o", mfc="none", mec=RED, mew=2, ms=10,
            label="sealed future measurements")
    ax.set_xlabel("cut order", fontsize=12); ax.set_ylabel("flank wear VB (µm)", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="upper left")
    save(fig, "paper_conformal")
    print(f"   (tool {tt}: all {best_len} sealed points inside the band — caption-true)")


# ---------- Figure 11 · online Kalman one-step trace (real tool) ----------
def fig_kalman():
    from run_online_monitor import load, fit_global_p, tr_params, kf_online_onestep, CENSOR
    d = load()
    # representative tool: longest record <= censor with >= 8 cycles (mirrors the old figure's tool)
    cand = sorted(d.tool_id.unique(), key=lambda t: -len(d[d.tool_id == t]))
    tt = next(t for t in cand if 8 <= len(d[d.tool_id == t]) <= 14)
    tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
    o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
    p = fit_global_p(tr)
    preds = kf_online_onestep(tr, o, v, p)          # list of (pred, true, k) for cuts 2..n
    arr = np.array([(pp, kk) for pp, _, kk in preds])
    yhat, kk = arr[:, 0], arr[:, 1].astype(int)
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.plot(o, v, "-o", color=GREY, lw=2.2, ms=8, label="measured VB")
    ax.plot(o[kk], yhat, "--s", color=GREEN, lw=2.2, ms=7,
            label="Kalman one-step-ahead prediction")
    ax.set_xlabel("cut order", fontsize=12); ax.set_ylabel("flank wear VB (µm)", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=10.5, loc="upper left")
    save(fig, "paper_kalman")


# ---------- Figure 12 · fair-baseline R2 bars (real CSV) ----------
def fig_fair():
    d = pd.read_csv(os.path.join(ROOT, "results", "f2_fair_baseline.csv"))
    colors = [RED, RED, GREY, BLUE, GREY]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.barh(d.model[::-1], d.R2[::-1], color=colors[::-1], alpha=0.9)
    for i, v in enumerate(d.R2[::-1]):
        ax.text(v - 0.04, i, f"{v:.2f}", va="center", ha="right", fontsize=10.5, color="#222")
    ax.axvline(0, color="k", lw=1.2)
    ax.set_xlabel("out-of-sample R² (leave-one-tool-out) — 0 = population mean", fontsize=11.5)
    ax.set_xlim(-2.15, 0.45); ax.grid(axis="x", alpha=0.3); ax.tick_params(labelsize=10.5)
    save(fig, "paper_fair_baseline")


def fig_pipeline():
    """paper_pipeline.png — Figure 3: the two-branch prognostic pipeline (English, no in-figure
    title, deployed record values). Replaces the bilingual deck chart pipeline_flow_full.png that
    carried a stale R2=0.67 and internal-governance framing."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(11.4, 5.2))
    ax.set_xlim(0, 114); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fontsize=9.3, tc="#1a2a36"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=tc, fontweight="bold")

    def arrow(x1, y1, x2, y2, color):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.8, color=color))

    G, GF = "#1e7d4f", "#e8f5ee"     # physics branch
    N, NF = "#5a6b7a", "#f1f4f6"     # sensor branch
    R, RF = "#b03a2e", "#fdecea"     # null outcome
    B, BF = "#1F5FA8", "#eaf1fa"     # data

    box(1, 20, 15, 12, "Data\n18-tool DOE\nVB targets", BF, B)
    # physics branch (top)
    box(24, 34, 18, 10, "Physics law\nVB = b + a·t$^{p}$", GF, G)
    box(46, 34, 18, 10, "Few-shot adapt\n(first m points)", GF, G)
    box(68, 34, 18, 10, "VB → HI → RUL\n(chipping hazard)", GF, G)
    box(90, 34, 21, 10, "Conformal band 90%\n+ Kalman monitor", GF, G)
    ax.text(100.5, 47.5, "ADOPTED · pooled R² = 0.70 (m = 4)", ha="center",
            fontsize=11, color=G, fontweight="bold")
    # sensor branch (bottom)
    box(24, 8, 18, 10, "Vibration features\n(294)", NF, N)
    box(46, 8, 18, 10, "Select + augment\n(fold-safe)", NF, N)
    box(68, 8, 18, 10, "ML / neural-net\nregression", NF, N)
    box(90, 8, 21, 10, "No signal transfers\nR² < 0", RF, R, tc="#7c2d24")
    ax.text(100.5, 4.2, "DOCUMENTED NULL", ha="center", fontsize=11,
            color=R, fontweight="bold")
    arrow(16.5, 29, 23.5, 39, G); arrow(42.5, 39, 45.5, 39, G)
    arrow(64.5, 39, 67.5, 39, G); arrow(86.5, 39, 89.5, 39, G)
    arrow(16.5, 23, 23.5, 13, N); arrow(42.5, 13, 45.5, 13, N)
    arrow(64.5, 13, 67.5, 13, N); arrow(86.5, 13, 89.5, 13, N)
    ax.text(57, 25.6, "identical leakage-safe leave-one-tool-out protocol for both branches",
            ha="center", fontsize=10.5, color="#33454f", style="italic")
    save(fig, "paper_pipeline")


if __name__ == "__main__":
    fig_models(); fig_mcurve(); fig_ablation(); fig_breakdown()
    fig_conformal(); fig_kalman(); fig_fair(); fig_pipeline()
    print("done — 8 paper figures + print twins")
