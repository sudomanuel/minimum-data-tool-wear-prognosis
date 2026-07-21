# -*- coding: utf-8 -*-
"""make_paper3_results_figs.py — the four result figures of Paper 3.

House drawing rules enforced here (same rules as the schematic figures):
  * every legend is placed OUTSIDE the data area, so it can never cover a point or a bar;
  * no annotation is drawn on top of another element: label positions are computed from the data
    extent and the axes are padded to make room, never the other way round;
  * no decorative connector crosses a marker; the only lines drawn are the data themselves;
  * every number comes from results/*.csv at draw time, so a figure cannot outlive its experiment.

  F1  p3_tradeoff.png    the trade-off Paper 1 stopped at, and the point that escapes it
  F2  p3_horizon.png     where the gain is: error per horizon, and the weighting that produced it
  F3  p3_audit.png       nested selection and conformal validity, m = 3 and m = 4
  F4  p3_per_tool.png    per-tool effect: is the gain a fleet property or one lucky tool?
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "outputs", "figures")
PRINT = os.path.join(FIG, "print")
os.makedirs(PRINT, exist_ok=True)

INK, SLATE, GRID = "#1a2a36", "#5a6b7a", "#DBE2E8"
EMER, BLUE, RED, PUR, GOLD = "#0E7A4D", "#1F5FA8", "#B03A2E", "#5B21B6", "#B7791F"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": SLATE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SLATE, "ytick.color": SLATE, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def _csv(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} — run the experiment that produces it first")
    return pd.read_csv(p)


def save(fig, stem):
    fig.savefig(os.path.join(FIG, f"{stem}.png"), dpi=300, facecolor="white")
    fig.savefig(os.path.join(PRINT, f"{stem}_600dpi.png"), dpi=600, facecolor="white")
    plt.close(fig)
    print(f"  wrote outputs/figures/{stem}.png", flush=True)


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------------------- F1
def fig_tradeoff():
    """Paper 1 could lower MAE only by giving up pooled R^2. Both metrics on one plane makes the
    frontier visible, and makes it visible that the adopted point is NOT on it."""
    pr = _csv("p3_probe_estimator.csv")
    p4 = pr[pr.m == 4]
    glob = p4[p4.tag.str.startswith("global gamma=")].copy()
    glob["g"] = glob.tag.str.extract(r"=(\d+)").astype(float)
    glob = glob.sort_values("g")
    ada = p4[p4.tag.str.startswith("gamma(h)")].copy()

    fig = plt.figure(figsize=(12.6, 6.0))
    fig.subplots_adjust(left=0.065, right=0.975, top=0.80, bottom=0.20, wspace=0.30)
    axL = fig.add_subplot(1, 2, 1)
    axR = fig.add_subplot(1, 2, 2)

    # -- left: the two metrics against one global gamma, the way Paper 1 saw them --
    grid(axL)
    axL.plot(glob.g, glob.MAE, "-o", color=BLUE, lw=2.0, ms=6, label="MAE (left axis)")
    axL.set_xlabel("global weighting exponent  $\\gamma$")
    axL.set_ylabel("MAE  [$\\mu$m]", color=BLUE)
    axL.tick_params(axis="y", colors=BLUE)
    axL.set_xlim(0.4, 8.6)
    axL.set_ylim(4.2, 7.6)

    ax2 = axL.twinx()
    ax2.plot(glob.g, glob.pooled_R2, marker="s", color=RED, lw=2.0, ms=6, ls="--",
             label="pooled $R^2$ (right axis)")
    ax2.set_ylabel("pooled $R^2$", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.set_ylim(0.10, 0.80)
    for s in ("top",):
        ax2.spines[s].set_visible(False)

    axL.axvline(3, color=SLATE, lw=1.0, ls=":")
    # captions sit in empty regions of the panel; no leader line is drawn, so nothing can cross a curve
    axL.text(3.15, 7.40, "Paper 1 stops here  ($\\gamma=3$)", color=SLATE, fontsize=9.5,
             ha="left", va="center")
    axL.text(8.50, 4.40, "beyond it: lower error, but the pooled $R^2$ collapses",
             color=SLATE, fontsize=9.5, ha="right", va="center")
    axL.set_title("(a)  one global $\\gamma$: the two metrics pull apart",
                  fontsize=11, color=INK, pad=8, loc="left")
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.155),
               ncol=2, frameon=False, fontsize=9.5)

    # -- right: the same runs in the (MAE, R^2) plane; the frontier and the escape --
    grid(axR)
    axR.plot(glob.MAE, glob.pooled_R2, "-", color=BLUE, lw=1.6, zorder=2)
    axR.scatter(glob.MAE, glob.pooled_R2, s=52, color="white", edgecolor=BLUE, lw=1.8, zorder=3)
    for _, r in glob.iterrows():
        if r.g in (1, 2, 3):                      # top of the frontier: label above the marker
            axR.text(r.MAE, r.pooled_R2 + 0.030, f"$\\gamma$={int(r.g)}", fontsize=9,
                     color=BLUE, ha="center", va="bottom", zorder=4)
        else:                                     # the steep descent: label to the right, clear of
            axR.text(r.MAE + 0.13, r.pooled_R2, f"$\\gamma$={int(r.g)}", fontsize=9,
                     color=BLUE, ha="left", va="center", zorder=4)
    axR.scatter(ada.MAE, ada.pooled_R2, s=52, color="white", edgecolor=EMER, lw=1.8, zorder=3)
    best = ada.sort_values("MAE").iloc[0]
    axR.scatter([best.MAE], [best.pooled_R2], s=150, marker="*", color=EMER, zorder=5)
    axR.annotate("adopted  $\\gamma(h)=8/h$\n%.2f $\\mu$m,   $R^2$ %.3f" % (best.MAE, best.pooled_R2),
                 xy=(best.MAE, best.pooled_R2 - 0.012), xytext=(3.42, 0.455),
                 fontsize=9.5, color=EMER, ha="left", fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=EMER, lw=1.1,
                                 shrinkA=2, shrinkB=6))
    axR.axhline(float(glob[glob.g == 3].pooled_R2.iloc[0]), color=SLATE, lw=1.0, ls=":")
    axR.axvline(float(glob[glob.g == 3].MAE.iloc[0]), color=SLATE, lw=1.0, ls=":")
    axR.text(5.72, 0.155, "worse than the record →", fontsize=9, color=SLATE, ha="left")
    axR.text(3.42, 0.865, "better on both metrics than anything one global $\\gamma$ reaches",
             fontsize=9, color=EMER, ha="left", va="top")
    axR.set_xlabel("MAE  [$\\mu$m]   (lower is better →)")
    axR.set_ylabel("pooled $R^2$   (higher is better ↑)")
    axR.set_xlim(3.2, 7.6)
    axR.set_ylim(0.10, 0.88)
    axR.set_title("(b)  the same runs in one plane", fontsize=11, color=INK, pad=8, loc="left")
    axR.legend(handles=[
        Line2D([], [], marker="o", ls="-", color=BLUE, mfc="white", mew=1.8,
               label="one global $\\gamma$ (Paper-1 family)"),
        Line2D([], [], marker="o", ls="none", color=EMER, mfc="white", mew=1.8,
               label="horizon-adaptive $\\gamma(h)$"),
        Line2D([], [], marker="*", ls="none", color=EMER, ms=13, label="adopted"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=3, frameon=False, fontsize=9.5)

    fig.text(0.065, 0.925, "THE TRADE-OFF PAPER 1 STOPPED AT", fontsize=14.5,
             fontweight="bold", color=INK)
    fig.text(0.065, 0.873,
             "m = 4 support readings, leave-one-tool-out over 18 tools. Every point is a complete "
             "re-run of the protocol.", fontsize=10, color=SLATE)
    save(fig, "p3_tradeoff")


# ----------------------------------------------------------------------------------- F2
def fig_horizon(hmax=12):
    """The mechanism claim lives per horizon, so it is shown per horizon."""
    hz = _csv("p3_breakdown_horizon.csv")
    hz = hz[hz.horizon <= hmax]
    x = hz.horizon.to_numpy(float)
    w = 0.38

    fig = plt.figure(figsize=(12.6, 6.6))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.11, hspace=0.34)
    axT = fig.add_subplot(2, 1, 1)
    axB = fig.add_subplot(2, 1, 2, sharex=axT)

    grid(axT, axis="y")
    axT.bar(x - w / 2, hz.MAE_record, w, color=BLUE, label="Paper-1 record  ($\\gamma=3$)")
    axT.bar(x + w / 2, hz.MAE_adopted, w, color=EMER, label="this work  ($\\gamma(h)=8/h$)")
    top = float(max(hz.MAE_record.max(), hz.MAE_adopted.max()))
    for xi, g in zip(x, hz.gain_pct):
        axT.text(xi, top * 1.045, f"{g:+.0f}%", ha="center", fontsize=8.6,
                 color=EMER if g > 0 else RED)
    axT.text(x[0] - 0.95, top * 1.045, "gain", ha="right", fontsize=8.6, color=SLATE,
             fontstyle="italic")
    for xi, n in zip(x, hz.n):
        axT.text(xi, -top * 0.055, f"n={int(n)}", ha="center", va="top", fontsize=8.4, color=SLATE)
    axT.text(x[0] - 0.95, -top * 0.055, "readings", ha="right", va="top", fontsize=8.4,
             color=SLATE, fontstyle="italic")
    axT.set_ylim(-top * 0.16, top * 1.16)
    axT.set_ylabel("MAE  [$\\mu$m]")
    axT.tick_params(axis="x", labelbottom=False, length=0)
    axT.spines["bottom"].set_visible(False)
    axT.axhline(0, color=SLATE, lw=1.0)
    axT.set_title("(a)  error at each horizon, and the number of sealed readings behind it",
                  fontsize=11, color=INK, pad=8, loc="left")
    axT.legend(loc="upper left", bbox_to_anchor=(0.005, 0.88), ncol=1, frameon=False, fontsize=9.5)

    grid(axB, axis="both")
    axB.plot(x, hz.gamma_record, "-s", color=BLUE, lw=2.0, ms=6,
             label="Paper 1: one $\\gamma$ for every horizon")
    axB.plot(x, hz.gamma_adopted, "-o", color=EMER, lw=2.0, ms=6,
             label="this work: $\\gamma(h)=8/h$")
    axB.fill_between(x, hz.gamma_adopted, hz.gamma_record, where=hz.gamma_adopted >= hz.gamma_record,
                     color=EMER, alpha=0.10, interpolate=True)
    axB.fill_between(x, hz.gamma_adopted, hz.gamma_record, where=hz.gamma_adopted < hz.gamma_record,
                     color=BLUE, alpha=0.08, interpolate=True)
    axB.set_xlabel("horizon  $h$  [readings beyond the support window]")
    axB.set_ylabel("weighting exponent")
    axB.set_xticks(x)
    axB.set_ylim(0, 8.9)
    axB.annotate("sharper than the record:\ntrust the last support point", xy=(1, 8.0),
                 xytext=(1.9, 7.2), fontsize=9.3, color=EMER,
                 arrowprops=dict(arrowstyle="-", color=EMER, lw=0.9))
    # placed in the empty band above the constant-gamma line: no leader, so nothing crosses a curve
    axB.text(5.30, 4.35, "flatter than the record:\nuse the whole window as an anchor",
             fontsize=9.3, color=BLUE, ha="left", va="bottom")
    axB.set_title("(b)  the weighting that produced it", fontsize=11, color=INK, pad=8, loc="left")
    axB.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=1, frameon=False, fontsize=9.5)

    fig.text(0.075, 0.930, "WHERE THE GAIN COMES FROM", fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.075, 0.880,
             "One weighting per predicted horizon. The near horizon is sharpened; the far horizon is "
             "anchored — the two things one constant $\\gamma$ cannot do at once.",
             fontsize=10, color=SLATE)
    save(fig, "p3_horizon")


# ----------------------------------------------------------------------------------- F3
def fig_audit():
    """Nothing is claimed on an in-search score: the figure shows the blind score next to it."""
    au = _csv("p3_nested_audit.csv")
    fig = plt.figure(figsize=(12.6, 5.2))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.16, wspace=0.30)
    axA = fig.add_subplot(1, 3, 1)
    axB = fig.add_subplot(1, 3, 2)
    axC = fig.add_subplot(1, 3, 3)

    # (a) in-search vs nested against the record
    grid(axA, axis="y")
    xs = np.arange(len(au), dtype=float)
    w = 0.32
    axA.bar(xs - w / 2, au.in_search_MAE, w, color="#9EC0E8", label="in-search (scheme chosen on all folds)")
    axA.bar(xs + w / 2, au.nested_MAE, w, color=EMER, label="nested (scheme re-chosen blind per fold)")
    for i, r in au.iterrows():
        axA.plot([i - 0.46, i + 0.46], [r.record, r.record], color=RED, lw=1.8, ls="--", zorder=4)
        axA.text(i + 0.48, r.record, f" record {r.record:.2f}", color=RED, fontsize=8.8,
                 va="center", ha="left")
        axA.text(i + w / 2, r.nested_MAE + 0.45, f"{r.nested_MAE:.2f}", ha="center",
                 fontsize=9, color=EMER, fontweight="bold")
    axA.set_xticks(xs)
    axA.set_xticklabels([f"m = {int(m)}" for m in au.m])
    axA.set_ylabel("MAE  [$\\mu$m]")
    axA.set_xlim(-0.75, len(au) - 0.05)
    axA.set_ylim(0, 15.5)
    axA.set_title("(a)  selection honesty", fontsize=11, color=INK, pad=8, loc="left")
    axA.legend(loc="upper left", bbox_to_anchor=(0.0, -0.11), ncol=1, frameon=False, fontsize=8.8)

    # (b) optimism: the gap between the two bars, signed
    grid(axB, axis="y")
    cols = [EMER if o <= 0 else RED for o in au.optimism]
    axB.bar(xs, au.optimism, 0.42, color=cols)
    axB.axhline(0, color=SLATE, lw=1.2)
    for i, r in au.iterrows():
        va = "bottom" if r.optimism > 0 else "top"
        off = 0.03 if r.optimism > 0 else -0.03
        axB.text(i, r.optimism + off, f"{r.optimism:+.2f} $\\mu$m", ha="center", va=va,
                 fontsize=9.5, color=EMER if r.optimism <= 0 else RED, fontweight="bold")
        axB.text(i, -0.46, f"winner held in\n{r.nested_winner_share*100:.0f}% of folds",
                 ha="center", va="top", fontsize=8.8, color=SLATE)
    axB.set_xticks(xs)
    axB.set_xticklabels([f"m = {int(m)}" for m in au.m])
    axB.set_ylabel("nested − in-search  [$\\mu$m]")
    axB.set_ylim(-0.80, 0.72)
    axB.set_title("(b)  selection optimism", fontsize=11, color=INK, pad=8, loc="left")

    # (c) conformal coverage against the pre-stated gate
    grid(axC, axis="y")
    axC.bar(xs, au.coverage_pct, 0.70, color=[PUR] * len(au))
    axC.axhline(90, color=SLATE, lw=1.2, ls=":")
    axC.axhline(88, color=RED, lw=1.6, ls="--")
    axC.text(1.44, 90.12, "nominal 90%", fontsize=8.8, color=SLATE, ha="left", va="bottom")
    axC.text(1.44, 87.86, "adoption\ngate 88%", fontsize=8.8, color=RED, ha="left", va="top")
    for i, r in au.iterrows():
        # inside the bar: the 90% reference line runs just above it, so an outside label would touch
        axC.text(i, r.coverage_pct - 0.16, f"{r.coverage_pct:.1f}%", ha="center", va="top",
                 fontsize=9.5, color="white", fontweight="bold")
        axC.text(i, 83.70, f"{r.mean_width_um:.0f} $\\mu$m wide", ha="center", va="bottom",
                 fontsize=8.8, color="white")
    axC.set_xticks(xs)
    axC.set_xticklabels([f"m = {int(r.m)}" for _, r in au.iterrows()])
    axC.set_ylabel("empirical coverage  [%]")
    axC.set_xlim(-0.62, 2.30)
    axC.set_ylim(83.5, 91.2)
    axC.set_title("(c)  interval validity", fontsize=11, color=INK, pad=8, loc="left")

    fig.text(0.07, 0.920, "THE TWO CHECKS THAT DECIDE ADOPTION", fontsize=14.5,
             fontweight="bold", color=INK)
    fig.text(0.07, 0.865,
             "Adopted only if the blind score beats the record AND the interval keeps its coverage. "
             "m = 4 passes both; m = 3 fails the first and is reported as such.",
             fontsize=10, color=SLATE)
    save(fig, "p3_audit")


# ----------------------------------------------------------------------------------- F4
def fig_per_tool():
    """A fleet mean can be carried by one tool. This shows every tool, including the two that lose."""
    tl = _csv("p3_breakdown_tool.csv").sort_values("MAE_record").reset_index(drop=True)
    y = np.arange(len(tl), dtype=float)

    fig = plt.figure(figsize=(12.6, 6.0))
    fig.subplots_adjust(left=0.085, right=0.72, top=0.80, bottom=0.10)
    ax = fig.add_subplot(1, 1, 1)
    grid(ax, axis="x")

    for i, r in tl.iterrows():
        better = r.MAE_adopted < r.MAE_record
        ax.plot([r.MAE_record, r.MAE_adopted], [i, i],
                color=EMER if better else RED, lw=2.4, alpha=0.55, zorder=2,
                solid_capstyle="butt")
    ax.scatter(tl.MAE_record, y, s=46, color="white", edgecolor=BLUE, lw=1.8, zorder=3)
    ax.scatter(tl.MAE_adopted, y, s=46, color=EMER, zorder=4)
    worse = tl[tl.MAE_adopted > tl.MAE_record]
    ax.scatter(worse.MAE_adopted, worse.index.to_numpy(float), s=46, color=RED, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}   (n={int(n)})" for t, n in zip(tl.tool, tl.n)], fontsize=9.2)
    ax.set_xlabel("per-tool MAE over its sealed readings  [$\\mu$m]")
    ax.set_xlim(-0.9, 21.5)
    ax.set_ylim(-0.9, len(tl) - 0.1)
    ax.invert_yaxis()

    mr, ma = float(tl.MAE_record.mean()), float(tl.MAE_adopted.mean())
    ax.axvline(mr, color=BLUE, lw=1.2, ls=":")
    ax.axvline(ma, color=EMER, lw=1.2, ls=":")
    ax.text(mr + 0.20, -0.45, f"fleet mean {mr:.2f}", color=BLUE, fontsize=9, va="center")
    ax.text(ma - 0.22, 0.55, f"fleet mean {ma:.2f}", color=EMER, fontsize=9, va="center",
            ha="right")

    n_imp = int((tl.MAE_adopted < tl.MAE_record).sum())
    n_tie = int(np.isclose(tl.MAE_adopted, tl.MAE_record).sum())
    n_wor = len(tl) - n_imp - n_tie
    side = ("\n".join([
        "READ IT AS:",
        "",
        f"improved   {n_imp} tools",
        f"unchanged  {n_tie} tool",
        f"worse      {n_wor} tools",
        "",
        "The two that lose are the two",
        "longest tools: their sealed part",
        "reaches far horizons, where the",
        "adaptive weight is deliberately",
        "flatter than the record's.",
        "",
        "n = sealed readings the tool",
        "contributes. Most tools stop",
        "shortly after the 4th reading,",
        "so they contribute one.",
    ]))
    fig.text(0.735, 0.63, side, fontsize=9.4, color=SLATE, va="top", linespacing=1.55)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=BLUE, mfc="white", mew=1.8,
               label="Paper-1 record  ($\\gamma=3$)"),
        Line2D([], [], marker="o", ls="none", color=EMER, label="this work — improved"),
        Line2D([], [], marker="o", ls="none", color=RED, label="this work — degraded"),
    ], loc="upper left", bbox_to_anchor=(0.735, 0.80), frameon=False, fontsize=9.4)

    fig.text(0.085, 0.925, "EVERY TOOL, INCLUDING THE ONES THAT LOSE", fontsize=14.5,
             fontweight="bold", color=INK)
    fig.text(0.085, 0.873,
             "Each tool is held out in turn; the line is the change from the record to this work.",
             fontsize=10, color=SLATE)
    save(fig, "p3_per_tool")


if __name__ == "__main__":
    print("Paper-3 result figures", flush=True)
    fig_tradeoff()
    fig_horizon()
    fig_audit()
    fig_per_tool()
    print("done.", flush=True)
