"""
phm.synthetic — subpaquete de generación sintética (SOLO contratos por ahora).

GATE P7/P7.1: ningún generador puede ejecutarse hasta cerrar el Nivel 2 del gate
(reports/p7_0_synthetic_generation_gate.md). Este subpaquete contiene únicamente
las interfaces y el contrato de auditoría que todo generador futuro debe cumplir.
"""
from .generator_contract import (  # noqa: F401
    GenerationMetadata,
    SyntheticGeneratorBase,
    GateClosedError,
    assert_gate_open,
    assert_train_fold_only,
    log_generation_event,
)
