"""
signal_qa.py — QA de señales raw (P7 gate; plan §4 de reports/t01_data_ingestion_audit_plan.md).

Cada archivo raw (segmentos {A,R}{exp}_p{X}.txt y full signals
{A,R}{exp}_parsed_modified.txt) produce una fila de métricas en
outputs/metrics/t01_signal_qa.csv. El resumen evalúa la consistencia del
sampling rate estimado entre archivos (FINDING 4: no hay fs nominal
documentado; aquí se cuantifica el estimado real).

Solo LECTURA de datos reales: este módulo no genera ni modifica señales.
"""
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import SEGMENTS_DIR, RAW_DIR, METRICS_DIR
from .filename_parser import parse_segment_name

FULL_SIGNALS_DIR = RAW_DIR / "full_signals"
SIGNAL_QA_CSV = METRICS_DIR / "t01_signal_qa.csv"

# {CH}{exp}_parsed_modified  ej. A66_parsed_modified
_FULL_PATTERN = re.compile(r'^(?P<ch>[A-Za-z]+)(?P<exp>\d+)_parsed_modified$',
                           re.IGNORECASE)


def _parse_full_signal_name(filename: str):
    m = _FULL_PATTERN.match(Path(filename).stem)
    if m is None:
        return None
    return m.group('ch').upper(), int(m.group('exp'))


def qa_one_file(path: Path, kind: str, direction: str, experiment_id: int,
                contact_id: Optional[int] = None) -> dict:
    """Métricas de QA para un archivo Time,Value. Nunca lanza: errores -> fila con error."""
    row = {
        'file': str(path.name), 'kind': kind, 'direction': direction,
        'experiment_id': experiment_id, 'contact_id': contact_id,
        'readable': True, 'error': '',
    }
    try:
        df = pd.read_csv(path, usecols=[0, 1], engine='c')
        df.columns = ['t', 'v']
    except Exception as exc:
        row.update({'readable': False, 'error': str(exc)[:120]})
        return row

    t = df['t'].to_numpy(dtype=float)
    v = df['v'].to_numpy(dtype=float)
    n = len(df)
    row['n_samples'] = n
    row['nan_frac'] = float(np.isnan(v).mean()) if n else np.nan
    if n < 2:
        row['error'] = 'menos de 2 muestras'
        return row

    dt = np.diff(t)
    pos = dt[dt > 0]
    med_dt = float(np.median(pos)) if len(pos) else np.nan
    row['duration_s'] = float(t[-1] - t[0])
    row['fs_est_hz'] = float(1.0 / med_dt) if med_dt and med_dt > 0 else np.nan
    row['monotonic_time'] = bool(np.all(dt >= 0))
    row['n_dup_timestamps'] = int((dt == 0).sum())
    # jitter: fracción de deltas que se desvían >1% de la mediana
    row['dt_offgrid_frac'] = float((np.abs(pos - med_dt) > 0.01 * med_dt).mean()) if len(pos) else np.nan

    vv = v[~np.isnan(v)]
    if len(vv):
        rms = float(np.sqrt(np.mean(vv ** 2)))
        sd = float(vv.std())
        row.update({
            'value_rms': rms, 'value_min': float(vv.min()), 'value_max': float(vv.max()),
            'spike_frac_z6': float((np.abs(vv - vv.mean()) > 6 * sd).mean()) if sd > 0 else 0.0,
        })
        # tramo constante más largo (saturación / canal muerto)
        if len(vv) > 1:
            changes = np.flatnonzero(np.diff(vv) != 0)
            bounds = np.concatenate(([-1], changes, [len(vv) - 1]))
            row['max_const_run'] = int(np.max(np.diff(bounds)))
        else:
            row['max_const_run'] = 1
    return row


def run_signal_qa(segments_dir: Path = None, full_dir: Path = None,
                  out_csv: Path = None) -> pd.DataFrame:
    """Corre el QA sobre segmentos + full signals y guarda el CSV. Devuelve el DataFrame."""
    segments_dir = Path(segments_dir or SEGMENTS_DIR)
    full_dir = Path(full_dir or FULL_SIGNALS_DIR)
    out_csv = Path(out_csv or SIGNAL_QA_CSV)

    rows = []
    if segments_dir.exists():
        for f in sorted(segments_dir.glob('*.txt')):
            meta = parse_segment_name(f.name)
            if meta is None:
                continue
            rows.append(qa_one_file(f, 'segment', meta.direction_code,
                                    meta.experiment_id, meta.contact_id))
    if full_dir.exists():
        for f in sorted(full_dir.glob('*.txt')):
            parsed = _parse_full_signal_name(f.name)
            if parsed is None:
                continue
            ch, exp = parsed
            rows.append(qa_one_file(f, 'full_signal', ch, exp))

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def summarize_fs(df: pd.DataFrame) -> dict:
    """Resumen de consistencia del sampling rate estimado (para el reporte P7)."""
    ok = df[df['readable'] & df['fs_est_hz'].notna()]
    fs = ok['fs_est_hz']
    return {
        'n_files': int(len(df)),
        'n_readable': int(df['readable'].sum()),
        'fs_min': float(fs.min()) if len(fs) else np.nan,
        'fs_median': float(fs.median()) if len(fs) else np.nan,
        'fs_max': float(fs.max()) if len(fs) else np.nan,
        'fs_rel_spread': float((fs.max() - fs.min()) / fs.median()) if len(fs) else np.nan,
        'n_non_monotonic': int((~ok['monotonic_time']).sum()) if len(ok) else 0,
        'n_with_dup_timestamps': int((ok['n_dup_timestamps'] > 0).sum()) if len(ok) else 0,
        'worst_dt_offgrid_frac': float(ok['dt_offgrid_frac'].max()) if len(ok) else np.nan,
    }
