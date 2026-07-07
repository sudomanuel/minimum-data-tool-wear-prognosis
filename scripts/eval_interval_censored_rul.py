"""
eval_interval_censored_rul.py — breakage-based, interval-censored RUL evaluation.

Turns model VB-threshold crossings into an honest, breakage-anchored RUL report.
Single-tool T01 is RIGHT-censored (breakage order T_R not recorded); a parametric
window is used only for sensitivity. Multi-tool ready: add a block to
config/failure_events.yaml (and, when available, a real breakage_order) and re-run.

Reads:
  config/failure_events.yaml
  results/hi_rul_threshold_summary.csv   (model, vb_failure_um, t_failure, ...)
Writes:
  results/interval_censored_rul.csv
"""
import csv, os, sys
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from phm.rul import interval_censored_failure


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "failure_events.yaml")))
    with open(os.path.join(ROOT, "results", "hi_rul_threshold_summary.csv")) as f:
        summ = list(csv.DictReader(f))

    out = []
    for tool, spec in cfg["tools"].items():
        TL = spec.get("last_measured_order")
        TR = spec.get("breakage_order")
        win = spec.get("breakage_window_assumed")
        # Reporting T_R: explicit if recorded, else upper end of the assumed window
        # (interval sensitivity); the right-censored view (T_R=None) is always kept too.
        tr_report = TR if TR is not None else (win[1] if win else None)
        for r in summ:
            tf = r["t_failure"]
            tf = None if tf == "no_crossing_within_horizon" else float(tf)
            m = interval_censored_failure(tf, TL, tr_report)
            rc = interval_censored_failure(tf, TL, None)  # right-censored view
            out.append({
                "tool": tool,
                "model": r["model"],
                "vb_failure_um": r["vb_failure_um"],
                "predicted_T_fail": m["predicted_T_fail"],
                "T_L": m["T_L"],
                "T_R": m["T_R"],
                "censoring": "interval" if TR is not None else "right(parametric_TR)",
                "failure_mode": m["failure_mode"],
                "interval_hit": m["interval_hit"],
                "distance_to_interval": m["distance_to_interval"],
                "unsafe_overestimation": m["unsafe_overestimation"],
                "conservative_underestimation": m["conservative_underestimation"],
                "interval_width": m["interval_width"],
                "rc_after_TL": (None if tf is None
                                else (not rc["conservative_underestimation"])),
            })

    outpath = os.path.join(ROOT, "results", "interval_censored_rul.csv")
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {outpath}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
