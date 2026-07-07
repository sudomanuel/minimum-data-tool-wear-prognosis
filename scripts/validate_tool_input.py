"""
validate_tool_input.py — check a dropped-in tool folder before it enters validation.

Usage:
    python scripts/validate_tool_input.py --tool T02
    python scripts/validate_tool_input.py --all

Checks (PASS/FAIL): folder structure, manifest schema, VB trajectory, referenced
signal files, and reports usable-row counts for the three dataset views
(sensor-based / trajectory-based VB / RUL-censoring). Read-only: never modifies data.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT = os.path.join(ROOT, "data", "input", "tools")
MIN_CONTACTS = 5
REQUIRED_MANIFEST_COLS = {
    "tool_id", "experiment_id", "within_tool_order", "vb_um", "has_signal",
    "n_valid_contacts", "signal_file", "usable_sensor", "usable_trajectory",
}


def _read_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_tool(tool):
    base = os.path.join(INPUT, tool)
    fails, warns, info = [], [], []

    if not os.path.isdir(base):
        return [f"tool folder not found: {base}"], [], []

    for sub in ("signals", "vb", "photos", "meta"):
        if not os.path.isdir(os.path.join(base, sub)):
            warns.append(f"missing subfolder: {sub}/")

    man_path = os.path.join(base, "manifest.csv")
    if not os.path.isfile(man_path):
        fails.append("manifest.csv missing (copy from templates/tool_manifest_template.csv)")
        return fails, warns, info
    rows = _read_csv(man_path)
    if not rows:
        fails.append("manifest.csv is empty")
        return fails, warns, info

    cols = set(rows[0].keys())
    missing = REQUIRED_MANIFEST_COLS - cols
    if missing:
        fails.append(f"manifest missing columns: {sorted(missing)}")

    # VB trajectory file
    vb_path = os.path.join(base, "vb", "vb_measurements.csv")
    if not os.path.isfile(vb_path):
        warns.append("vb/vb_measurements.csv missing (trajectory may come from manifest vb_um)")

    # per-row integrity + view counts
    def truthy(v):
        return str(v).strip().lower() in ("true", "1", "yes")

    n_sensor = n_traj = 0
    orders = []
    for r in rows:
        oid = r.get("within_tool_order", "")
        try:
            orders.append(int(float(oid)))
        except (TypeError, ValueError):
            fails.append(f"experiment {r.get('experiment_id')}: bad within_tool_order {oid!r}")
        vb = str(r.get("vb_um", "")).strip()
        has_vb = vb not in ("", "nan", "None")
        if has_vb:
            n_traj += 1
        # sensor usability
        sig = str(r.get("signal_file", "")).strip()
        nvc = str(r.get("n_valid_contacts", "0")).strip() or "0"
        try:
            nvc_i = int(float(nvc))
        except ValueError:
            nvc_i = 0
        if truthy(r.get("has_signal")) and nvc_i >= MIN_CONTACTS:
            n_sensor += 1
            if sig and not os.path.isfile(os.path.join(base, sig)):
                fails.append(f"experiment {r.get('experiment_id')}: signal_file not found: {sig}")
        # consistency of declared flags
        if truthy(r.get("usable_sensor")) and not (truthy(r.get("has_signal")) and nvc_i >= MIN_CONTACTS):
            warns.append(f"experiment {r.get('experiment_id')}: usable_sensor=true but signal/contacts insufficient")
        if truthy(r.get("usable_trajectory")) and not has_vb:
            fails.append(f"experiment {r.get('experiment_id')}: usable_trajectory=true but vb_um empty")

    if orders and sorted(orders) != list(range(min(orders), min(orders) + len(orders))):
        warns.append(f"within_tool_order not a contiguous sequence: {sorted(orders)}")

    info.append(f"experiments: {len(rows)}")
    info.append(f"sensor-based usable: {n_sensor}")
    info.append(f"trajectory-based VB usable: {n_traj}")
    fe = os.path.join(ROOT, "config", "failure_events.yaml")
    info.append(f"failure_events registry: {'present' if os.path.isfile(fe) else 'MISSING'} "
                f"(add a block for {tool} when breakage is known)")
    return fails, warns, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tool", help="tool id, e.g. T02")
    ap.add_argument("--all", action="store_true", help="validate every tools/T* folder")
    args = ap.parse_args()

    if args.all:
        tools = sorted(d for d in os.listdir(INPUT)
                       if os.path.isdir(os.path.join(INPUT, d)) and not d.startswith("_"))
    elif args.tool:
        tools = [args.tool]
    else:
        ap.error("pass --tool T0X or --all")

    overall_ok = True
    for t in tools:
        fails, warns, info = validate_tool(t)
        print(f"\n=== {t} ===")
        for i in info:
            print(f"  info: {i}")
        for w in warns:
            print(f"  WARN: {w}")
        for fl in fails:
            print(f"  FAIL: {fl}")
        verdict = "PASS" if not fails else "FAIL"
        overall_ok = overall_ok and not fails
        print(f"  -> {verdict}")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
