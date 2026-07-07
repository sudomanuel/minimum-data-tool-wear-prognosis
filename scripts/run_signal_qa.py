#!/usr/bin/env python3
"""
run_signal_qa.py — QA de señales raw T01 (P7 gate; audit plan §4).

Produce outputs/metrics/t01_signal_qa.csv (una fila por archivo raw) e imprime
el resumen de consistencia del sampling rate estimado (FINDING 4).

Uso:
    python run.py signal-qa
    python run.py signal-qa --data-root D:/otra/copia/de/data
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phm.signal_qa import run_signal_qa, summarize_fs, SIGNAL_QA_CSV  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=None,
                    help="raíz data/ alternativa (usa data/raw/segments y data/raw/full_signals)")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"CSV de salida (default {SIGNAL_QA_CSV})")
    args = ap.parse_args()

    seg_dir = full_dir = None
    if args.data_root is not None:
        seg_dir = args.data_root / "raw" / "segments"
        full_dir = args.data_root / "raw" / "full_signals"

    df = run_signal_qa(segments_dir=seg_dir, full_dir=full_dir, out_csv=args.out)
    if df.empty:
        print("ERROR: no se encontraron archivos raw (¿--data-root correcto?)")
        return 1

    s = summarize_fs(df)
    print(f"QA de {s['n_files']} archivos ({s['n_readable']} legibles) -> "
          f"{args.out or SIGNAL_QA_CSV}")
    print(f"fs estimado [Hz]: min={s['fs_min']:.2f}  mediana={s['fs_median']:.2f}  "
          f"max={s['fs_max']:.2f}  spread relativo={s['fs_rel_spread']:.2e}")
    print(f"no-monotonicos={s['n_non_monotonic']}  con timestamps duplicados="
          f"{s['n_with_dup_timestamps']}  peor dt_offgrid_frac={s['worst_dt_offgrid_frac']:.2e}")

    problems = df[(~df['readable']) | (df.get('nan_frac', 0) > 0)]
    if len(problems):
        print(f"ATENCION: {len(problems)} archivos con problemas (ver CSV)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
