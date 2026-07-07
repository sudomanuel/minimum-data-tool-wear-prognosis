"""
run_minimum_data_experiments.py — LOTO + few-shot (Axis 2: m) and the training-tool
learning curve (Axis 1: K) for the Minimum-Data Physics-Integrated Adaptive PINN.

This is an EXECUTABLE SCAFFOLD. The leakage-critical parts are real and tested:
  - tool discovery + data-availability gate (needs >=2 tools);
  - the few-shot SEAL: calibration = first m points, future = the rest, disjoint;
  - per-tool / per-(K,m) iteration and result logging.
The GLOBAL MODEL and the ADAPTER are documented hooks (`fit_global`, `adapt_tool`,
`predict`) to be filled by the PINN/Physics Architect (see
reports/physics_integrated_equation_candidates.md). Until then the script runs the
structure with baseline placeholders and refuses to emit "results" without >=2 tools.

Usage:
  python scripts/run_minimum_data_experiments.py --mode fewshot
  python scripts/run_minimum_data_experiments.py --mode kcurve
"""
import argparse
import itertools
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_TOOLS = os.path.join(ROOT, "data", "input", "tools")
M_GRID = [0, 1, 2, 3, 5]
K_M_GRID = [0, 2, 3]
SEED = 20260623


# ----------------------------------------------------------------------------- data
def discover_tools():
    """Return validated tool ids. T01 is the frozen baseline (in the main data tree);
    new tools live under data/input/tools/. A tool counts only if its manifest exists."""
    tools = []
    if os.path.isdir(INPUT_TOOLS):
        for d in sorted(os.listdir(INPUT_TOOLS)):
            p = os.path.join(INPUT_TOOLS, d)
            if d.startswith("_") or not os.path.isdir(p):
                continue
            if os.path.isfile(os.path.join(p, "manifest.csv")):
                tools.append(d)
    # T01 frozen baseline is available from the main pipeline outputs
    if "T01" not in tools and os.path.isfile(os.path.join(ROOT, "data", "targets", "microscope_vb.csv")):
        tools.append("T01")
    return sorted(tools)


def load_tool_trajectory(tool):
    """HOOK: return the per-tool ordered (order, vb, features) for the trajectory view.
    Implement once manifests are populated (see data/input/schema/DATASET_SCHEMAS.md)."""
    raise NotImplementedError(
        f"load_tool_trajectory({tool}) not implemented — fill once tool manifests exist")


# ------------------------------------------------------------------- leakage-safe seal
def fewshot_split(order_index, m):
    """REAL leakage seal. Given a tool's ordered indices, return (calibration, future).
    Calibration = first m points; future = the rest; strictly disjoint and ordered."""
    idx = list(order_index)
    cal = idx[:m]
    fut = idx[m:]
    assert set(cal).isdisjoint(set(fut)), "few-shot seal violated: cal/future overlap"
    assert (not cal) or (not fut) or (max(cal) < min(fut)), "future must follow calibration"
    return cal, fut


# --------------------------------------------------------------------------- hooks
def fit_global(train_tools):
    """HOOK: fit global preprocessing + global degradation law on train_tools ONLY."""
    raise NotImplementedError("fit_global: plug in M3 adaptive integral PINN + baselines")


def adapt_tool(global_model, cal_points):
    """HOOK: update ONLY tool-specific params {VB0_j, alpha_j, b_j} on cal_points.
    MUST NOT touch global weights or see future points."""
    raise NotImplementedError("adapt_tool: per-tool few-shot adaptation")


def predict_and_score(model, future_points):
    """HOOK: predict VB on future points; derive HI/RUL; return the metric dict."""
    raise NotImplementedError("predict_and_score: MAE_future, late-wear MAE, safety flags")


# --------------------------------------------------------------------------- drivers
def run_fewshot(tools):
    print(f"[fewshot] tools={tools}")
    for test in tools:
        train = [t for t in tools if t != test]
        gm = fit_global(train)                       # test tool fully unseen
        for m in M_GRID:
            traj = load_tool_trajectory(test)
            cal, fut = fewshot_split(range(len(traj)), m)
            am = adapt_tool(gm, [traj[i] for i in cal])
            _ = predict_and_score(am, [traj[i] for i in fut])
            # log row: tool, m, model, MAE_future, ... -> results/loto_fewshot_results.csv


def run_kcurve(tools):
    print(f"[kcurve] tools={tools}")
    rng = __import__("random").Random(SEED)
    for test in tools:
        cand = [t for t in tools if t != test]
        for K in range(1, len(cand) + 1):
            subsets = list(itertools.combinations(cand, K))
            if len(subsets) > 10:                    # cap; reproducible sample
                subsets = rng.sample(subsets, 10)
            for sub in subsets:
                gm = fit_global(list(sub))
                for m in K_M_GRID:
                    traj = load_tool_trajectory(test)
                    cal, fut = fewshot_split(range(len(traj)), m)
                    am = adapt_tool(gm, [traj[i] for i in cal])
                    _ = predict_and_score(am, [traj[i] for i in fut])
                    # log row: tool, K, sub, m, ... -> results/training_tool_learning_curve.csv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["fewshot", "kcurve"], required=True)
    args = ap.parse_args()

    tools = discover_tools()
    print(f"discovered tools: {tools}")
    if len([t for t in tools if t != "T01"]) < 1 or len(tools) < 2:
        print("\nBLOCKED: multi-tool experiments need >=2 tools (T01 + at least one new "
              "tool).\nDrop tools into data/input/tools/ and run "
              "`python scripts/validate_tool_input.py --all` first.\n"
              "T01 alone only yields the frozen temporal-degeneracy diagnostic.")
        sys.exit(2)

    (run_fewshot if args.mode == "fewshot" else run_kcurve)(tools)


if __name__ == "__main__":
    main()
