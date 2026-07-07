"""
targets.py — target-aware VB loading driven by config/targets.yaml (P8.0).

Single place that knows which VB file/column is OFFICIAL vs LEGACY. New P8+ pipelines call
`load_official_vb()`; the historical P1-P6 code keeps using `config.TARGET_FILE`
(vb_targets.csv) untouched for reproducibility.

Policy (P7.3):
  official: data/targets/microscope_vb.csv, column VB_um, 12 experiments (full trajectory)
  legacy:   data/raw/targets/vb_targets.csv, 10 experiments (recorded-signal subset; P1-P6)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PROJECT_ROOT

TARGETS_YAML = PROJECT_ROOT / "config" / "targets.yaml"
EXPERIMENT_ID_COL = "experiment_id"
TARGET_COLUMN = "VB_um"

# Experiments with recorded vibration signals (sensor-modeling eligible). 71-72 are
# target-only (performed, no signals). Kept here so callers don't re-derive it.
RECORDED_EXPERIMENTS = {66, 67, 68, 69, 70, 73, 74, 75, 76, 77}
TARGET_ONLY_EXPERIMENTS = {71, 72}


@dataclass
class TargetSpec:
    name: str
    source: Path
    column: str
    status: str


def _parse_targets_yaml(path: Path = TARGETS_YAML) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        # minimal fallback parser for the official_target block (yaml unavailable)
        block = {}
        in_official = False
        for ln in path.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if s.startswith("official_target:"):
                in_official = True
                continue
            if in_official:
                if s and not ln.startswith((" ", "\t")):
                    break
                if ":" in s:
                    k, v = s.split(":", 1)
                    block[k.strip()] = v.strip()
        return {"official_target": block}


def official_target_spec() -> TargetSpec:
    cfg = _parse_targets_yaml()
    ot = cfg["official_target"]
    return TargetSpec(name=ot.get("name", "VB"),
                      source=PROJECT_ROOT / ot["source"],
                      column=ot.get("column", TARGET_COLUMN),
                      status=ot.get("status", "confirmed"))


def load_official_vb(recorded_only: bool = False,
                     data_root: Optional[Path] = None) -> pd.DataFrame:
    """Official VB trajectory as a DataFrame [experiment_id, VB_um, has_signal, row_type].

    recorded_only=True -> drop the 71-72 target-only rows (use for sensor-based modeling).
    Full 12-point trajectory (default) -> use for trajectory/RUL work.
    """
    spec = official_target_spec()
    src = spec.source if data_root is None else (data_root / spec.source.name)
    raw = pd.read_csv(src, sep=None, engine="python")  # sniff , or ;
    # normalize a possibly-legacy VS column name
    col = spec.column
    if col not in raw.columns:
        for c in raw.columns:
            if c.lower() in ("vs", "vs_final_um", "vb_um", "vb"):
                col = c
                break
    df = raw[[EXPERIMENT_ID_COL, col]].rename(columns={col: TARGET_COLUMN})
    df[EXPERIMENT_ID_COL] = df[EXPERIMENT_ID_COL].astype(int)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).sort_values(EXPERIMENT_ID_COL).reset_index(drop=True)
    df["has_signal"] = df[EXPERIMENT_ID_COL].isin(RECORDED_EXPERIMENTS)
    df["row_type"] = df[EXPERIMENT_ID_COL].apply(
        lambda e: "target_only_missing_signal" if e in TARGET_ONLY_EXPERIMENTS
        else "recorded")
    df["physical_experiment_order"] = range(1, len(df) + 1)
    if recorded_only:
        df = df[df["has_signal"]].reset_index(drop=True)
    return df


def attach_official_vb(features_df: pd.DataFrame,
                       data_root: Optional[Path] = None) -> pd.DataFrame:
    """Join the official VB onto a sensor feature table (recorded experiments only).

    Use this in P8+ instead of reading vb_targets.csv. Sensor models must NOT receive the
    71-72 target-only rows (they have no features).
    """
    vb = load_official_vb(recorded_only=True, data_root=data_root)[[EXPERIMENT_ID_COL, TARGET_COLUMN]]
    out = features_df.drop(columns=[c for c in features_df.columns if c == TARGET_COLUMN],
                           errors="ignore")
    return out.merge(vb, on=EXPERIMENT_ID_COL, how="inner")
