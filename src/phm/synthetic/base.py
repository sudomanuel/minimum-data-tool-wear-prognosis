"""
base.py — Stage-6 public interface surface for synthetic generation.

The real contract lives in `generator_contract.py`; this module is the stable import path
referenced by the integrated pipeline plan (Stage 6). Importing from here keeps call-sites
decoupled from the contract file's internal name.

GATE: generation stays blocked until `data/manifest/synthetic_gate_level2_closed.flag` exists
(see `assert_gate_open`). Naive augmentation (feature_noise/scaling) is NOT here — it is the
Family-1 baseline control, kept in `src/phm/augmentation.py`.
"""
from .generator_contract import (  # noqa: F401
    SyntheticGeneratorBase,
    GenerationMetadata,
    GateClosedError,
    assert_gate_open,
    assert_train_fold_only,
    log_generation_event,
    GATE_FLAG,
    VALID_STATUS,
)

__all__ = [
    "SyntheticGeneratorBase",
    "GenerationMetadata",
    "GateClosedError",
    "assert_gate_open",
    "assert_train_fold_only",
    "log_generation_event",
    "GATE_FLAG",
    "VALID_STATUS",
]
