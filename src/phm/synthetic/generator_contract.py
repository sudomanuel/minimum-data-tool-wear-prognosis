"""
generator_contract.py — contrato obligatorio para TODO generador sintético futuro.

P7.1 (2026-06-12): SOLO interfaces. Ningún generador está implementado ni puede
ejecutarse: `assert_gate_open()` falla mientras el Nivel 2 del gate de verdad de
datos siga abierto (reports/p7_0_synthetic_generation_gate.md). El gate se abre
EXCLUSIVAMENTE creando el archivo-marcador
`data/manifest/synthetic_gate_level2_closed.flag` tras las respuestas del
laboratorio (target oficial, procedencia, política exp-77, fs formal) — ese
archivo se crea a mano, nunca desde código.

Reglas que este contrato hace ejecutables:
1. El generador se ajusta SOLO con experimentos del fold de entrenamiento
   (`assert_train_fold_only`); el held-out jamás entra al generador.
2. Todo evento de generación queda registrado con metadata completa
   (`log_generation_event`) — auditable por
   `leakage_audit.check_synthetic_from_train_fold_only`.
3. Sin labels RUL sintéticos. Los experimentos 71-72 (performed_but_not_recorded)
   jamás reciben señales sintéticas presentadas como reales.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from ..config import METRICS_DIR, DATA_DIR
from .. import synthetic_validation as _sv

GATE_FLAG = DATA_DIR / "manifest" / "synthetic_gate_level2_closed.flag"
GENERATOR_METADATA_LOG = METRICS_DIR / "synthetic_generator_metadata.csv"

VALID_STATUS = {
    "draft_never_used",      # código escrito, jamás ejecutado para producir datos
    "validation_only",       # muestras usadas solo en bloques A/B del protocolo
    "training_eligible",     # aprobado por el protocolo para regímenes C (post-gate)
}


class GateClosedError(RuntimeError):
    """La generación sintética está bloqueada por el gate P7 (Nivel 2 abierto)."""


def assert_gate_open() -> None:
    """Falla mientras el Nivel 2 del gate siga abierto. Llamar ANTES de generar nada."""
    if not GATE_FLAG.exists():
        raise GateClosedError(
            "SYNTHETIC GATE CLOSED: falta confirmar target oficial (VB/VS), procedencia, "
            "política exp-77 y fs nominal (reports/p7_0_synthetic_generation_gate.md). "
            f"El gate se abre creando manualmente {GATE_FLAG.as_posix()} tras las "
            "respuestas del laboratorio.")


def assert_train_fold_only(source_experiments: Iterable[int],
                           excluded_test_experiments: Iterable[int]) -> None:
    """Garantiza que el generador no ve el/los held-out del fold."""
    src = {int(e) for e in source_experiments}
    excl = {int(e) for e in excluded_test_experiments}
    if not src:
        raise ValueError("source_experiments vacío: el generador debe declarar sus fuentes")
    if not excl:
        raise ValueError(
            "excluded_test_experiments vacío: todo evento de fold debe declarar qué "
            "experimento(s) held-out excluye (usar scope full_data solo para diagnóstico)")
    leak = src & excl
    if leak:
        raise ValueError(f"LEAKAGE: experimentos held-out {sorted(leak)} dentro de las "
                         f"fuentes del generador")


@dataclass
class GenerationMetadata:
    """Metadata OBLIGATORIA de todo evento de generación sintética."""
    generator_name: str
    source_experiments: Sequence[int]
    excluded_test_experiments: Sequence[int]
    target_source: str               # oficial P7.3: 'data/targets/microscope_vb.csv' (columna VB_um)
    physical_constraints_used: str   # p.ej. 'monotone VB; rate=softplus(a+bE); VB in [50,400]um'
    random_seed: int
    synthetic_ratio: float           # n_synthetic / n_real del fold
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    synthetic_data_status: str = "draft_never_used"

    def validate(self) -> None:
        if not self.generator_name:
            raise ValueError("generator_name obligatorio")
        if self.synthetic_data_status not in VALID_STATUS:
            raise ValueError(f"synthetic_data_status {self.synthetic_data_status!r} "
                             f"invalido; usar uno de {sorted(VALID_STATUS)}")
        if "vs" in Path(self.target_source).stem.lower() and "vb" in Path(self.target_source).stem.lower():
            raise ValueError("target_source ambiguo: declarar UN archivo de target")
        if self.synthetic_ratio < 0:
            raise ValueError("synthetic_ratio negativo")
        assert_train_fold_only(self.source_experiments, self.excluded_test_experiments)


def log_generation_event(meta: GenerationMetadata, scope: str,
                         n_real_rows: int, n_synthetic_rows: int,
                         fold_test_experiment_id: Optional[int] = None,
                         notes: str = "") -> dict:
    """Registro doble: log auditado (leakage check) + metadata completa del generador.

    NO genera datos: solo registra. La generación misma además requiere
    `assert_gate_open()` (ver SyntheticGeneratorBase.generate).
    """
    meta.validate()
    core = _sv.log_generation_event(
        method=meta.generator_name, family=meta.physical_constraints_used[:40] or "unspecified",
        scope=scope, train_experiment_ids=meta.source_experiments,
        n_real_rows=n_real_rows, n_synthetic_rows=n_synthetic_rows,
        fold_test_experiment_id=fold_test_experiment_id,
        seed=meta.random_seed, notes=notes,
    )
    row = {**asdict(meta), "scope": scope,
           "fold_test_experiment_id": fold_test_experiment_id,
           "n_real_rows": n_real_rows, "n_synthetic_rows": n_synthetic_rows,
           "source_experiments": ";".join(str(int(e)) for e in meta.source_experiments),
           "excluded_test_experiments": ";".join(str(int(e)) for e in meta.excluded_test_experiments),
           "notes": notes}
    GENERATOR_METADATA_LOG.parent.mkdir(parents=True, exist_ok=True)
    header = not GENERATOR_METADATA_LOG.exists()
    pd.DataFrame([row]).to_csv(GENERATOR_METADATA_LOG, mode="a", header=header, index=False)
    return core


class SyntheticGeneratorBase(ABC):
    """Interfaz base de generadores. Las subclases implementan _fit/_generate;
    la base impone gate + contrato y NO puede esquivarse sin dejar rastro."""

    def __init__(self, meta: GenerationMetadata):
        meta.validate()
        self.meta = meta
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "SyntheticGeneratorBase":
        """Ajuste SOLO con filas del fold de entrenamiento (verificado contra metadata)."""
        seen = {int(e) for e in train_df["experiment_id"].unique()}
        declared = {int(e) for e in self.meta.source_experiments}
        if not seen.issubset(declared):
            raise ValueError(f"fit() ve experimentos no declarados: {sorted(seen - declared)}")
        forbidden = seen & {int(e) for e in self.meta.excluded_test_experiments}
        if forbidden:
            raise ValueError(f"LEAKAGE en fit(): held-out {sorted(forbidden)} presente")
        self._fit(train_df)
        self._fitted = True
        return self

    def generate(self, n_samples: int, scope: str,
                 fold_test_experiment_id: Optional[int] = None) -> pd.DataFrame:
        """Genera muestras sintéticas. BLOQUEADO mientras el gate esté cerrado."""
        assert_gate_open()
        if not self._fitted:
            raise RuntimeError("generate() antes de fit()")
        out = self._generate(n_samples)
        log_generation_event(self.meta, scope=scope,
                             n_real_rows=len(self.meta.source_experiments),
                             n_synthetic_rows=len(out),
                             fold_test_experiment_id=fold_test_experiment_id)
        return out

    @abstractmethod
    def _fit(self, train_df: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def _generate(self, n_samples: int) -> pd.DataFrame:
        ...
