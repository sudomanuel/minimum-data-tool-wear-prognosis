#!/usr/bin/env python3
"""
rebaseline_official_vb.py — P8.0 re-baseline of time-aware baselines + RUL on the OFFICIAL
VB target (data/targets/microscope_vb.csv), replacing the legacy vb_targets.csv subset.

Self-contained and READ-ONLY w.r.t. the frozen P2/P4 code: it does NOT touch
src/phm/time_aware.py or scripts/derive_rul.py (those stay reproducible for the P1-P6
historical baseline). Outputs go to results/ (the new P8+ results area), separate from the
historical outputs/metrics/.

Time-aware baselines: Linear(t) and Poly2(t) fit VB(t); LOEO MAE.
RUL: threshold crossing on the Poly2(t) full-data fit for several thresholds.

Uso:
    python run.py rebaseline-vb
    python run.py rebaseline-vb --repo-root <repo-root>
"""
import argparse
import json
from pathlib import Path

import numpy as np

WT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = WT_ROOT / "results"

# Legacy P2 reference numbers (vb_targets.csv subset, recorded-order t) for honest comparison.
LEGACY_REF = {"Linear(t)": 9.93, "Poly2(t)": 4.96, "max_vb": 280, "n": 10,
              "t_failure_poly2_300": 10.77}


def _read_yaml_source(targets_yaml: Path) -> str:
    src = "data/targets/microscope_vb.csv"
    try:
        import yaml
        cfg = yaml.safe_load(targets_yaml.read_text(encoding="utf-8"))
        src = cfg["official_target"]["source"]
    except Exception:
        for line in targets_yaml.read_text(encoding="utf-8").splitlines():
            if "source:" in line and "microscope" in line:
                src = line.split("source:")[1].strip()
                break
    return src


def _load_official_vb(repo_root: Path, targets_yaml: Path):
    src_rel = _read_yaml_source(targets_yaml)
    path = (repo_root / src_rel)
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not ln.strip():
            continue
        eid, vb = ln.replace(";", ",").split(",")[:2]
        rows.append((int(eid), float(vb)))
    rows.sort()
    return rows, src_rel


def _loeo_mae(t, vb, degree):
    t = np.asarray(t, float)
    vb = np.asarray(vb, float)
    errs = []
    for i in range(len(t)):
        mask = np.arange(len(t)) != i
        coef = np.polyfit(t[mask], vb[mask], degree)
        pred = np.polyval(coef, t[i])
        errs.append(abs(pred - vb[i]))
    return float(np.mean(errs)), float(np.max(errs))


def _threshold_crossing(coef, thr, t_lo, t_max, step):
    grid = np.arange(t_lo, t_max + step, step)
    vals = np.polyval(coef, grid)
    hit = np.where(vals >= thr)[0]
    if len(hit) == 0:
        return None
    return float(grid[hit[0]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    targets_yaml = WT_ROOT / "config" / "targets.yaml"

    rows, src_rel = _load_official_vb(args.repo_root, targets_yaml)
    eids = [e for e, _ in rows]
    vb = [v for _, v in rows]
    t_phys = list(range(1, len(rows) + 1))  # physical order 1..12 (incl. 71-72)
    recorded = {66, 67, 68, 69, 70, 73, 74, 75, 76, 77}
    rec_idx = [i for i, e in enumerate(eids) if e in recorded]

    print(f"official source: {src_rel}")
    print(f"official VB: {dict(rows)}")
    print(f"max VB official = {max(vb)} (legacy subset max = {LEGACY_REF['max_vb']})")

    # ---- time-aware baselines on official VB ----
    ta_rows = []
    # full 12-point physical trajectory (incl. 71-72 target-only points)
    for name, deg in [("Linear(t)", 1), ("Poly2(t)", 2)]:
        mae12, mx12 = _loeo_mae(t_phys, vb, deg)
        ta_rows.append({"baseline": name, "trajectory": "official_12pt_physical",
                        "n": len(rows), "loeo_mae": round(mae12, 3),
                        "loeo_max_err": round(mx12, 3),
                        "legacy_ref_mae": LEGACY_REF.get(name)})
    # recorded-only (apples-to-apples with P2's 10 points), t = physical order with gaps
    t_rec = [t_phys[i] for i in rec_idx]
    vb_rec = [vb[i] for i in rec_idx]
    for name, deg in [("Linear(t)", 1), ("Poly2(t)", 2)]:
        mae, mx = _loeo_mae(t_rec, vb_rec, deg)
        ta_rows.append({"baseline": name, "trajectory": "official_10pt_recorded_gapped",
                        "n": len(t_rec), "loeo_mae": round(mae, 3),
                        "loeo_max_err": round(mx, 3),
                        "legacy_ref_mae": LEGACY_REF.get(name)})

    ta_csv = RESULTS / "p8_0_rebaseline_timeaware.csv"
    _write_csv(ta_csv, ta_rows)

    # ---- RUL threshold crossing on official Poly2 full-data fit (12pt) ----
    coef2 = np.polyfit(t_phys, vb, 2)
    thresholds = [220, 250, 300, 600]
    t_last = t_phys[-1]
    rul_rows = []
    for thr in thresholds:
        tf = _threshold_crossing(coef2, thr, t_phys[0], 24.0, 0.01)
        crossed_in_range = (tf is not None and tf <= t_last)
        rul_rows.append({
            "threshold_um": thr, "official_max_vb": max(vb),
            "t_failure": round(tf, 3) if tf is not None else "no_crossing<=t24",
            "rul_at_last_obs": round(tf - t_last, 3) if tf is not None else "n/a",
            "rul_extrapolated": (not crossed_in_range),
            "legacy_t_failure_300": LEGACY_REF["t_failure_poly2_300"] if thr == 300 else "",
        })
    rul_csv = RESULTS / "p8_0_rebaseline_rul.csv"
    _write_csv(rul_csv, rul_rows)

    # ---- machine-readable summary ----
    summary = {
        "official_source": src_rel, "official_vb_range": [min(vb), max(vb)],
        "legacy_vb_range": [85, 280], "n_official_points": len(rows),
        "time_aware": ta_rows, "rul": rul_rows,
        "all_rul_extrapolated": all(r["rul_extrapolated"] for r in rul_rows),
    }
    (RESULTS / "p8_0_rebaseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n-> {ta_csv}")
    for r in ta_rows:
        print(f"   {r['baseline']:<10} {r['trajectory']:<28} LOEO MAE={r['loeo_mae']:<6} "
              f"(legacy ref {r['legacy_ref_mae']})")
    print(f"\n-> {rul_csv}")
    for r in rul_rows:
        print(f"   thr {r['threshold_um']:>3}um: t_failure={r['t_failure']} "
              f"extrapolated={r['rul_extrapolated']}")
    print(f"\nALL RUL extrapolated (official max {max(vb)} < every threshold): "
          f"{summary['all_rul_extrapolated']}")
    return 0


def _write_csv(path, rows):
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
