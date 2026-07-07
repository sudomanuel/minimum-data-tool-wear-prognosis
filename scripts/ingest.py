#!/usr/bin/env python3
"""
ingest.py — build experiment_features.csv from raw multi-cutter signals.

Flexible front-end for the multi-cutter ingestion engine
(`phm.ingestion.build_dataset_multitool`). Everything is configurable so the
same command serves any input: a different segments folder, a different target
file, a custom output path, cache on/off, and the number of workers.

Examples:
    python run.py ingest                          # defaults (data/raw/segments)
    python scripts/ingest.py --segments data/raw/segments --out data/processed/experiment_features.csv
    python scripts/ingest.py --no-cache --jobs 1  # clean, sequential
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from phm.config import (
    SEGMENTS_DIR, TARGET_FILE, METADATA_FILE, PROCESSED_DATASET, N_INGEST_WORKERS,
)
from phm.ingestion import build_dataset_multitool


def main():
    ap = argparse.ArgumentParser(description="Multi-cutter ingestion to experiment_features.csv")
    ap.add_argument("--segments", default=str(SEGMENTS_DIR),
                    help="folder with raw segments {CH}{cutter}_{exp}_p{part}.txt")
    ap.add_argument("--target", default=str(TARGET_FILE), help="targets CSV (tool_id, experiment_id, VB_um, ...)")
    ap.add_argument("--metadata", default=str(METADATA_FILE), help="optional metadata CSV")
    ap.add_argument("--out", default=str(PROCESSED_DATASET), help="output CSV path")
    ap.add_argument("--no-cache", action="store_true", help="disable the per-file feature cache")
    ap.add_argument("--jobs", type=int, default=N_INGEST_WORKERS, help="parallel workers (-1 = all)")
    args = ap.parse_args()

    print(f"[INGEST] segments={args.segments}\n[INGEST] out={args.out}")
    if Path(args.out) == PROCESSED_DATASET:
        print("[INGEST] NOTA: escribe el dataset canonico (esquema aggregate-over-parts).")

    df = build_dataset_multitool(
        segments_dir=Path(args.segments),
        target_file=Path(args.target),
        metadata_file=Path(args.metadata) if args.metadata else None,
        out_path=Path(args.out),
        use_cache=not args.no_cache,
        n_jobs=args.jobs,
        verbose=True,
    )
    print(f"[INGEST] OK — {df.shape[0]} experimentos, {df.shape[1]} columnas.")


if __name__ == "__main__":
    main()
