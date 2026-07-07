"""
filename_parser.py — parsea segmentos de vibracion/acustica.

Patron unificado (retro-compatible):
    NUEVO   : {CH}{cuchilla}_{exp}_p{X}     ej. A1_5_p3  (axial, tool 1, exp 5, parte 3)
    LEGACY  : {CH}{exp}_p{X}                ej. A66_p1   (tool -> DEFAULT_TOOL_ID)

Se distinguen por el nº de bloques numericos: si hay dos (..._..._pX) el
primero es la cuchilla y el segundo el experimento; si hay uno, es el
experimento (formato T01 antiguo, sin cuchilla).

`CH` (canal) se resuelve via CHANNEL_TOKENS: A=axial, R=rotacional,
AE=acustica. Tokens desconocidos se conservan en minuscula como nombre.
"""
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

from .config import CHANNEL_TOKENS, DEFAULT_TOOL_ID

# {CH}{n1}[_{n2}]_p{part}
_PATTERN = re.compile(
    r'^(?P<ch>[A-Za-z]+)(?P<n1>\d+)(?:_(?P<n2>\d+))?_p(?P<part>\d+)$',
    re.IGNORECASE,
)


@dataclass
class SegmentMeta:
    filename: str
    direction_code: str            # token crudo del canal: 'A' | 'R' | 'AE' ...
    channel: str                   # nombre resuelto: 'axial' | 'rotacional' | ...
    experiment_id: int
    contact_id: int                # nº de parte (pX)
    tool_id: Optional[Union[int, str]] = None  # None en legacy (se resuelve al escanear)


def parse_segment_name(filename: str) -> Optional[SegmentMeta]:
    """Devuelve metadata si el nombre coincide con el patron, sino None."""
    m = _PATTERN.match(Path(filename).stem)
    if m is None:
        return None
    token = m.group('ch').upper()
    n1 = int(m.group('n1'))
    n2 = m.group('n2')
    if n2 is None:                          # legacy: {CH}{exp}_p{X}
        tool_id, exp_id = None, n1
    else:                                   # nuevo: {CH}{tool}_{exp}_p{X}
        tool_id, exp_id = n1, int(n2)
    return SegmentMeta(
        filename=filename,
        direction_code=token,
        channel=CHANNEL_TOKENS.get(token, token.lower()),
        experiment_id=exp_id,
        contact_id=int(m.group('part')),
        tool_id=tool_id,
    )


def scan_segments(directory: Path) -> dict:
    """LEGACY: devuelve {experiment_id: {(direction_code, contact_id): Path}}.

    Mantiene la firma usada por el `dataset_builder` actual (T01). No usar
    para la ingesta multi-cuchilla; ahi se usa `scan_experiments`.
    """
    directory = Path(directory)
    out: dict = {}
    if not directory.exists():
        return out
    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.suffix.lower() != '.txt':
            continue
        meta = parse_segment_name(f.name)
        if meta is None:
            continue
        out.setdefault(meta.experiment_id, {})[(meta.direction_code, meta.contact_id)] = f
    return out


def scan_experiments(directory: Path,
                     default_tool_id: Union[int, str] = DEFAULT_TOOL_ID) -> dict:
    """Ingesta multi-cuchilla. Devuelve:

        {(tool_id, experiment_id): {channel: {part_id: Path}}}

    - `tool_id` legacy (None) se resuelve a `default_tool_id`.
    - Robusto a nº de partes variable (run-to-failure): no asume p1..p6.
    """
    directory = Path(directory)
    out: dict = {}
    if not directory.exists():
        return out
    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.suffix.lower() != '.txt':
            continue
        meta = parse_segment_name(f.name)
        if meta is None:
            continue
        tool = meta.tool_id if meta.tool_id is not None else default_tool_id
        key = (tool, meta.experiment_id)
        out.setdefault(key, {}).setdefault(meta.channel, {})[meta.contact_id] = f
    return out
