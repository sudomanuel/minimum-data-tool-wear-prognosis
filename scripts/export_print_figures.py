"""export_print_figures.py — Reviewer-response F1: submission-grade figure set.

MDPI production asks for vector figures or >=300 dpi raster. The 13 embedded figures are produced by
deterministic scripts that save screen-dpi PNGs. Rather than editing every producer, this script
monkey-patches matplotlib's Figure.savefig so that whenever a producer saves one of the 13 target
PNGs, a vector PDF twin and a 600-dpi PNG twin are ALSO written to outputs/figures/print/. The
producers themselves are executed unchanged (they are deterministic; re-running them was validated
bit-identical on the result CSVs).

Output: outputs/figures/print/<name>.pdf + <name>_600dpi.png for the 13 manuscript figures.
"""
import os, sys, runpy, traceback
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

PRINT_DIR = os.path.join(ROOT, "outputs", "figures", "print")
os.makedirs(PRINT_DIR, exist_ok=True)

TARGETS = {"context_wear", "pipeline_flow_full", "all_models_comparison", "mcurve",
           "sensor_branch_performance", "f3_breakdown", "conformal_demo", "f1_rul_windows",
           "r3_chipping_hazard", "r1_envelope", "kalman_online", "f2_fair_baseline", "f4_multirate"}

# producer -> figures it owns (run order chosen so the OFFICIAL producer writes last where duplicated)
PRODUCERS = ["make_mgmt_diagrams.py", "make_headline_figures.py", "make_mgmt_figures.py",
             "run_optimal_config_final_metrics.py", "make_deck_figures.py",
             "run_f3_map_estimator.py", "run_f1_multithreshold_rul.py", "run_r3_chipping_rul.py",
             "run_r1_twophase_law.py", "run_online_monitor.py", "run_f2_fair_baseline.py",
             "run_f4_multirate_kalman.py"]

_orig = Figure.savefig
exported = {}


def patched(self, fname, *a, **k):
    out = _orig(self, fname, *a, **k)
    try:
        name = os.path.splitext(os.path.basename(str(fname)))[0]
        if str(fname).endswith(".png") and name in TARGETS:
            _orig(self, os.path.join(PRINT_DIR, name + ".pdf"), bbox_inches=k.get("bbox_inches"))
            _orig(self, os.path.join(PRINT_DIR, name + "_600dpi.png"), dpi=600,
                  bbox_inches=k.get("bbox_inches"))
            exported[name] = exported.get(name, 0) + 1
    except Exception as e:
        print(f"  [print-twin FAILED for {fname}: {e}]")
    return out


Figure.savefig = patched


def main():
    print(f"F1: PRINT FIGURE EXPORT — {len(TARGETS)} targets via {len(PRODUCERS)} producers\n")
    for p in PRODUCERS:
        path = os.path.join(ROOT, "scripts", p)
        try:
            print(f"-- running {p}")
            runpy.run_path(path, run_name="__main__")
        except SystemExit:
            pass
        except Exception:
            print(f"   PRODUCER FAILED: {p}")
            traceback.print_exc(limit=1)
    missing = TARGETS - set(exported)
    print(f"\nexported print twins: {len(exported)}/{len(TARGETS)}")
    for n in sorted(exported):
        print(f"  {n}: pdf + 600dpi png")
    if missing:
        print("MISSING:", sorted(missing))
    else:
        print("ALL 13 manuscript figures now have vector PDF + 600 dpi twins in outputs/figures/print/")


if __name__ == "__main__":
    main()
