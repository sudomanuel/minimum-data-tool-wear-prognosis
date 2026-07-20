# -*- coding: utf-8 -*-
"""make_paper3_figures.py — dedicated producer for the Paper-3 (PI-MAML) manuscript figures.

Same house conventions as the Paper-1 producer: NO in-figure titles (captions carry them),
English labels, self-explanatory names, deployed values only, print twins at 600 dpi.

  p3_task_construction.png  the change this paper makes: tools-as-tasks vs sub-window tasks
  p3_adaptation.png         error vs. inner gradient steps (K = 0 isolates the initialisation)
  p3_comparison.png         PI-MAML vs the deployed record and the fleet baseline
  p3_lolo.png               cross-condition generalisation, one factor level withheld per fold
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "outputs", "figures")
PRINT = os.path.join(FIG, "print")
os.makedirs(PRINT, exist_ok=True)

EMER, RED, BLUE, GREY, NAVY = "#0E7A4D", "#B03A2E", "#1F5FA8", "#7A8794", "#141E4F"
REC = {3: 11.0, 4: 5.6}
FLEET = 18.7


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(PRINT, name + "_600dpi.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _load(f):
    p = os.path.join(RES, f)
    return pd.read_csv(p) if os.path.exists(p) else None


def fig_task_construction():
    meta_p = os.path.join(RES, "p3_meta.json")
    if not os.path.exists(meta_p):
        print("skip p3_task_construction (no p3_meta.json)"); return
    meta = json.load(open(meta_p))
    a, b = meta["tasks_tools_as_tasks"], meta["tasks_subwindow"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(["tools as tasks\n(definition adjudicated\na non-starter)",
                   "sub-window tasks\n(this work, Sec. 3.1)"], [a, b],
                  color=[GREY, EMER], width=0.55)
    for r, v in zip(bars, [a, b]):
        ax.text(r.get_x() + r.get_width() / 2, v + max(b, 1) * 0.03, str(v),
                ha="center", fontsize=13, fontweight="bold")
    ax.set_ylabel("number of learning tasks", fontsize=12)
    ax.set_ylim(0, b * 1.18)
    ax.grid(axis="y", alpha=0.3); ax.tick_params(labelsize=10.5)
    ax.text(0.5, b * 1.10, f"×{b/max(a,1):.0f} from the same 172 inspections",
            ha="center", fontsize=11, style="italic", color=EMER)
    save(fig, "p3_task_construction")


def fig_adaptation():
    d = _load("p3_adaptation.csv")
    if d is None:
        print("skip p3_adaptation (no csv)"); return
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for m, col in [(3, BLUE), (4, EMER)]:
        s = d[d.m == m].sort_values("K")
        if not len(s):
            continue
        ax.plot(s.K, s.MAE, "-o", color=col, lw=2.4, ms=9, label=f"PI-MAML, m = {m}")
        for _, r in s.iterrows():
            ax.annotate(f"{r.MAE:.1f}", (r.K, r.MAE), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9.5, color=col)
        ax.axhline(REC[m], ls=":", color=col, lw=1.6, alpha=0.75)
        ax.text(3.05, REC[m], f" record {REC[m]}", va="center", fontsize=9, color=col)
    ax.set_xlabel("inner gradient steps K at deployment  (K = 0 = pure meta-initialisation)",
                  fontsize=11.5)
    ax.set_ylabel("future-VB MAE (µm)", fontsize=12)
    ax.set_xticks([0, 1, 2, 3]); ax.grid(alpha=0.3); ax.legend(fontsize=10)
    save(fig, "p3_adaptation")


def fig_comparison():
    d = _load("p3_main.csv")
    if d is None:
        print("skip p3_comparison (no csv)"); return
    rows, cols = [], []
    for m in (3, 4):
        s = d[d.m == m]
        if not len(s):
            continue
        rows.append((f"fleet-average baseline (m = {m})", FLEET, GREY))
        rows.append((f"physics few-shot record (m = {m})", REC[m], BLUE))
        rows.append((f"PI-MAML (m = {m})", float(s.iloc[0].MAE), EMER))
    if not rows:
        print("skip p3_comparison (empty)"); return
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [r[2] for r in rows][::-1]
    ax.barh(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.012, i, f"{v:.1f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("future-VB MAE (µm) — lower is better", fontsize=12)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.grid(axis="x", alpha=0.3); ax.tick_params(labelsize=10.5)
    save(fig, "p3_comparison")


def fig_lolo():
    d = _load("p3_lolo.csv")
    if d is None:
        print("skip p3_lolo (no csv)"); return
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for m, col, off in [(3, BLUE, -0.19), (4, EMER, 0.19)]:
        s = d[d.m == m]
        if not len(s):
            continue
        x = np.arange(len(s))
        ax.bar(x + off, s.MAE, width=0.36, color=col, label=f"m = {m}")
        for xi, v in zip(x + off, s.MAE):
            ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
        ax.axhline(REC[m], ls=":", color=col, lw=1.5, alpha=0.75)
        ax.set_xticks(np.arange(len(s))); ax.set_xticklabels(s.tag.tolist(), fontsize=9.5)
    ax.set_ylabel("future-VB MAE (µm)", fontsize=12)
    ax.set_xlabel("withheld factor level (dotted lines = the deployed record at each budget)",
                  fontsize=11)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=10)
    save(fig, "p3_lolo")


if __name__ == "__main__":
    fig_task_construction(); fig_adaptation(); fig_comparison(); fig_lolo()
    print("done — Paper 3 figures")
