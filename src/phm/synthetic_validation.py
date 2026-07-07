"""
synthetic_validation.py — primitivas de validación de datos sintéticos (skeleton P7).

GATE P7 VIGENTE: en este repo NO existe todavía ningún generador sintético, por
decisión (reports/p7_t01_data_truth_reconciliation.md). Este módulo provee:

1. El CONTRATO de auditoría que todo generador futuro debe cumplir:
   llamar `log_generation_event(...)` en cada generación, declarando el scope
   (fold LOEO/LOTO o full-data) y los experimentos de TRAIN usados. El check
   `leakage_audit.check_synthetic_from_train_fold_only` audita ese log.
2. Métricas del protocolo de validación (reports/synthetic_data_validation_protocol.md):
   bloque A (similarity-but-not-duplication) y bloque B (physical plausibility).
   Operan sobre arrays/DataFrames ya existentes; NO generan datos.

Regla binding: el test es siempre real; el generador se ajusta SOLO con el
fold de entrenamiento; sin labels RUL sintéticos jamás.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import METRICS_DIR

SYNTH_GENERATION_LOG = METRICS_DIR / "synthetic_generation_log.csv"

# scopes permitidos para eventos de generación
SCOPE_LOEO_FOLD = "loeo_fold"    # generación dentro de un fold (requiere fold_test_experiment_id)
SCOPE_LOTO_FOLD = "loto_fold"    # ídem a nivel herramienta (futuro multi-tool)
SCOPE_FULL_DATA = "full_data"    # SOLO diagnóstico post-hoc, nunca métricas LOEO/LOTO
VALID_SCOPES = {SCOPE_LOEO_FOLD, SCOPE_LOTO_FOLD, SCOPE_FULL_DATA}


# -----------------------------------------------------------------------------
# Contrato de auditoría (lo consume leakage_audit.check_synthetic_from_train_fold_only)
# -----------------------------------------------------------------------------
def log_generation_event(method: str, family: str, scope: str,
                         train_experiment_ids: Iterable[int],
                         n_real_rows: int, n_synthetic_rows: int,
                         fold_test_experiment_id: Optional[int] = None,
                         seed: Optional[int] = None, notes: str = "") -> dict:
    """Registra un evento de generación sintética en el log auditable.

    Todo generador DEBE llamar esto una vez por (fold, método). El log es
    append-only; el check de leakage lo lee completo.
    """
    if scope not in VALID_SCOPES:
        raise ValueError(f"scope invalido {scope!r}; usar uno de {sorted(VALID_SCOPES)}")
    if scope in (SCOPE_LOEO_FOLD, SCOPE_LOTO_FOLD) and fold_test_experiment_id is None:
        raise ValueError(f"scope {scope} requiere fold_test_experiment_id")
    train_ids = sorted(int(e) for e in train_experiment_ids)
    if fold_test_experiment_id is not None and int(fold_test_experiment_id) in train_ids:
        raise ValueError(
            f"LEAKAGE: el experimento held-out {fold_test_experiment_id} aparece "
            f"en los train_experiment_ids del generador")
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": method, "family": family, "scope": scope,
        "fold_test_experiment_id": fold_test_experiment_id,
        "train_experiment_ids": ";".join(str(e) for e in train_ids),
        "n_train_experiments": len(train_ids),
        "n_real_rows": int(n_real_rows), "n_synthetic_rows": int(n_synthetic_rows),
        "seed": seed, "notes": notes,
    }
    SYNTH_GENERATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    header = not SYNTH_GENERATION_LOG.exists()
    pd.DataFrame([row]).to_csv(SYNTH_GENERATION_LOG, mode="a", header=header, index=False)
    return row


def read_generation_log() -> pd.DataFrame:
    """Log completo de eventos de generación (vacío si no hay generador aún)."""
    if not SYNTH_GENERATION_LOG.exists():
        return pd.DataFrame()
    return pd.read_csv(SYNTH_GENERATION_LOG)


# -----------------------------------------------------------------------------
# Bloque A — similarity but not duplication
# -----------------------------------------------------------------------------
def nn_distance_ratio(real_X: np.ndarray, synth_X: np.ndarray) -> dict:
    """Distancia al vecino real más cercano, normalizada por la escala real-real.

    ratio mediano < 0.3 => alarma de clon (sintéticos pegados a los reales).
    """
    real_X = np.asarray(real_X, dtype=float)
    synth_X = np.asarray(synth_X, dtype=float)
    d_rr = _pairwise_min_dist(real_X, real_X, exclude_self=True)
    d_sr = _pairwise_min_dist(synth_X, real_X, exclude_self=False)
    ref = float(np.median(d_rr)) if len(d_rr) else np.nan
    ratios = d_sr / ref if ref and ref > 0 else np.full(len(d_sr), np.nan)
    return {
        "median_nn_ratio": float(np.median(ratios)),
        "p05_nn_ratio": float(np.percentile(ratios, 5)),
        "clone_alarm": bool(np.median(ratios) < 0.3),
    }


def duplicate_fraction(real_X: np.ndarray, synth_X: np.ndarray, tol: float = 1e-8) -> float:
    """Fracción de sintéticos que son duplicados (casi) exactos de alguna fila real."""
    d = _pairwise_min_dist(np.asarray(synth_X, float), np.asarray(real_X, float))
    return float((d <= tol).mean()) if len(d) else np.nan


def ks_distance_per_feature(real_X: pd.DataFrame, synth_X: pd.DataFrame) -> pd.Series:
    """Distancia KS por feature (columnas comunes). Reportar la DISTRIBUCIÓN completa."""
    cols = [c for c in real_X.columns if c in synth_X.columns]
    out = {}
    for c in cols:
        a = pd.to_numeric(real_X[c], errors="coerce").dropna()
        b = pd.to_numeric(synth_X[c], errors="coerce").dropna()
        if len(a) > 1 and len(b) > 1:
            out[c] = float(stats.ks_2samp(a, b).statistic)
    return pd.Series(out, name="ks_distance")


def correlation_matrix_distance(real_X: pd.DataFrame, synth_X: pd.DataFrame) -> dict:
    """‖Corr_real − Corr_synth‖_F y conteo de signos invertidos (pares válidos)."""
    cols = [c for c in real_X.columns if c in synth_X.columns]
    cr = real_X[cols].corr().to_numpy()
    cs = synth_X[cols].corr().to_numpy()
    mask = ~(np.isnan(cr) | np.isnan(cs))
    diff = np.where(mask, cr - cs, 0.0)
    iu = np.triu_indices_from(cr, k=1)
    valid = mask[iu]
    flips = int(np.sum((np.sign(cr[iu]) != np.sign(cs[iu])) & valid))
    return {"frobenius": float(np.linalg.norm(diff)),
            "sign_flips": flips, "n_pairs": int(valid.sum())}


def _pairwise_min_dist(A: np.ndarray, B: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    if len(A) == 0 or len(B) == 0:
        return np.array([])
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    if exclude_self and A.shape == B.shape and np.allclose(A, B):
        np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min(axis=1))


# -----------------------------------------------------------------------------
# Bloque B — physical plausibility (trayectorias VB)
# -----------------------------------------------------------------------------
def monotonicity_violations(vb_seq: Sequence[float]) -> int:
    """Nº de pasos con ΔVB < 0 (el desgaste no decrece)."""
    d = np.diff(np.asarray(vb_seq, dtype=float))
    return int((d < 0).sum())


def negative_rate_fraction(vb_seq: Sequence[float]) -> float:
    """Fracción de incrementos negativos (target: 0)."""
    d = np.diff(np.asarray(vb_seq, dtype=float))
    return float((d < 0).mean()) if len(d) else np.nan


def smoothness_second_diff(vb_seq: Sequence[float]) -> float:
    """Media de |Δ²VB| — suave pero NO plana (referencia OOF P3: 20.6)."""
    a = np.asarray(vb_seq, dtype=float)
    return float(np.abs(np.diff(a, n=2)).mean()) if len(a) > 2 else np.nan


def wear_increment_range_check(vb_seq: Sequence[float],
                               lo: float = 0.0, hi: float = 50.0) -> dict:
    """Incrementos dentro de un rango físico declarado (T01 real: 14–35 µm/paso)."""
    d = np.diff(np.asarray(vb_seq, dtype=float))
    return {"min_increment": float(d.min()) if len(d) else np.nan,
            "max_increment": float(d.max()) if len(d) else np.nan,
            "frac_out_of_range": float(((d < lo) | (d > hi)).mean()) if len(d) else np.nan}


def rate_energy_spearman(rates: Sequence[float], energies: Sequence[float]) -> float:
    """ρ de Spearman entre tasa de desgaste y energía (referencia T01 ≈ 0.68)."""
    r = np.asarray(rates, dtype=float)
    e = np.asarray(energies, dtype=float)
    ok = ~(np.isnan(r) | np.isnan(e))
    if ok.sum() < 3:
        return np.nan
    return float(stats.spearmanr(r[ok], e[ok]).statistic)
