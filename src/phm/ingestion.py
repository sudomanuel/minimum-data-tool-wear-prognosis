"""
ingestion.py — motor de ingesta multi-cuchilla (flexible, cacheado, paralelo).

Convierte senales crudas `{canal}{cuchilla}_{exp}_p{parte}.txt` en un
`experiment_features.csv` (1 fila por experimento), preparado para CUALQUIER
entrada y robusto a:
  - multiples cuchillas (tool_id),
  - numero de partes VARIABLE por experimento (run-to-failure),
  - canales presentes variables (axial / rotacional / acustica),
  - naming legacy de T01 (sin cuchilla).

Etapas (ver reports/data_ingestion_architecture.md):
  [1] scan     filename_parser.scan_experiments -> {(tool,exp):{canal:{parte:path}}}
  [2] extract  features por ARCHIVO, CACHEADAS por hash, en PARALELO
  [3] aggregate sobre PARTES -> {mean, std, slope} por (canal x feature base)
                + energia total por canal (driver fisico) + ratios cruzados
  [4] assemble fila por experimento (ancho FIJO, independiente del nº de partes)
  [5] join     targets / metadata -> experiment_features.csv

El esquema de salida es el contrato del proyecto (tool_id, experiment_id,
experiment_order, VB_um, + features numericas), asi que el harness y la PINN
lo consumen sin cambios. Solo cambian los NOMBRES de las features respecto al
builder legacy por-contacto: aqui son `{A|R|AE}_{base}_{mean|std|slope}`, lo
que mantiene los filtros de subset (`A_`, `R_`).
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    SEGMENTS_DIR, TARGET_FILE, METADATA_FILE, PROCESSED_DIR, PROCESSED_DATASET,
    EXPERIMENT_ID_COL, TOOL_ID_COL, EXP_ORDER_COL, TARGET_COLUMN,
    CHANNEL_TOKENS, AGG_OVER_PARTS, FEATURE_CACHE_DIR, N_INGEST_WORKERS,
    ENABLE_FREQUENCY_FEATURES, MIN_SAMPLES_FOR_FFT, METRICS_DIR,
    DATASET_SCHEMA_MULTITOOL,
)
from .filename_parser import scan_experiments
from .data_loader import load_signal, load_target_csv
from .preprocessing import preprocess_signal
from .feature_extraction import extract_all_features, all_feature_names
from .dataset_builder import write_dataset_meta, read_dataset_meta

# canal (nombre resuelto) -> token/prefijo para nombrar features (A, R, AE).
_CHANNEL_PREFIX = {name: tok for tok, name in CHANNEL_TOKENS.items()}


def _channel_prefix(channel_name: str) -> str:
    return _CHANNEL_PREFIX.get(channel_name, str(channel_name)[:2].upper())


# =============================================================================
# [2] extraccion por archivo, con cache por hash (ruta + size + mtime)
# =============================================================================
def _cache_key(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def extract_file_features(path, enable_frequency: bool = True,
                          use_cache: bool = True) -> dict:
    """Features de una senal (un archivo). Cacheadas por hash; NaN si ilegible."""
    path = Path(path)
    names = all_feature_names(enable_frequency)
    cf = None
    if use_cache:
        FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cf = FEATURE_CACHE_DIR / f"{_cache_key(path)}.json"
        if cf.exists():
            try:
                cached = json.loads(cf.read_text(encoding="utf-8"))
                return {k: (float("nan") if v is None else v) for k, v in cached.items()}
            except Exception:
                pass

    feats = None
    df = load_signal(path)
    if df is not None:
        df_c, fs = preprocess_signal(df, center=True)
        if df_c is not None and not df_c.empty:
            feats = extract_all_features(
                df_c["vibration_value"].values, df_c["timestamp"].values,
                sampling_rate_hz=fs, enable_frequency=enable_frequency,
                min_samples_fft=MIN_SAMPLES_FOR_FFT,
            )
    if feats is None:
        feats = {f: float("nan") for f in names}

    if cf is not None:
        try:
            cf.write_text(json.dumps(
                {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                 for k, v in feats.items()}), encoding="utf-8")
        except Exception:
            pass
    return feats


# =============================================================================
# [3] agregacion sobre partes (ancho fijo, robusto al nº de partes)
# =============================================================================
def _slope(values) -> float:
    """Pendiente de la regresion lineal de `values` vs indice de parte."""
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    if mask.sum() < 2:
        return float("nan")
    x = np.arange(len(v))[mask]
    return float(np.polyfit(x, v[mask], 1)[0])


def _aggregate(values, aggs) -> dict:
    v = np.asarray(values, dtype=float)
    has = np.isfinite(v).any()
    out = {}
    for a in aggs:
        if a == "mean":
            out["mean"] = float(np.nanmean(v)) if has else float("nan")
        elif a == "std":
            out["std"] = float(np.nanstd(v)) if has else float("nan")
        elif a == "slope":
            out["slope"] = _slope(v)
        else:
            raise ValueError(f"agg desconocido: {a!r}")
    return out


# =============================================================================
# [4] fila por experimento
# =============================================================================
def build_experiment_features(key, channels: dict, enable_frequency: bool,
                              aggs, use_cache: bool = True) -> dict:
    """Una fila ancho-fijo para `(tool, exp)` agregando sobre las partes."""
    tool_id, exp_id = key
    row = {EXPERIMENT_ID_COL: int(exp_id), TOOL_ID_COL: tool_id}
    base_names = all_feature_names(enable_frequency)
    rms_mean_by_ch, erg_mean_by_ch = {}, {}

    for channel_name, parts in channels.items():
        prefix = _channel_prefix(channel_name)
        part_ids = sorted(parts.keys())
        per_part = [extract_file_features(parts[pid], enable_frequency, use_cache)
                    for pid in part_ids]

        for base in base_names:
            vals = [pf.get(base, float("nan")) for pf in per_part]
            for a, val in _aggregate(vals, aggs).items():
                row[f"{prefix}_{base}_{a}"] = val

        erg = np.asarray([pf.get("energy", float("nan")) for pf in per_part], float)
        rms = np.asarray([pf.get("rms", float("nan")) for pf in per_part], float)
        row[f"{prefix}_energy_total"] = float(np.nansum(erg)) if np.isfinite(erg).any() else float("nan")
        erg_mean_by_ch[prefix] = float(np.nanmean(erg)) if np.isfinite(erg).any() else float("nan")
        rms_mean_by_ch[prefix] = float(np.nanmean(rms)) if np.isfinite(rms).any() else float("nan")

    # ratios cruzados axial / rotacional (si ambos canales presentes)
    def _ratio(num, den):
        return float(num / den) if (den and np.isfinite(den) and den != 0
                                    and np.isfinite(num)) else float("nan")
    if "A" in rms_mean_by_ch and "R" in rms_mean_by_ch:
        row["A_to_R_rms_ratio"] = _ratio(rms_mean_by_ch["A"], rms_mean_by_ch["R"])
    if "A" in erg_mean_by_ch and "R" in erg_mean_by_ch:
        row["A_to_R_energy_ratio"] = _ratio(erg_mean_by_ch["A"], erg_mean_by_ch["R"])
    return row


# =============================================================================
# inventario / data-quality (no rompe; informa)
# =============================================================================
def build_inventory(seg_index: dict) -> pd.DataFrame:
    rows = []
    for (tool, exp), channels in sorted(seg_index.items(), key=lambda kv: (str(kv[0][0]), int(kv[0][1]))):
        rows.append({
            TOOL_ID_COL: tool, EXPERIMENT_ID_COL: int(exp),
            "channels": ";".join(sorted(channels)),
            "n_channels": len(channels),
            "n_parts_max": max((len(p) for p in channels.values()), default=0),
            "parts_per_channel": ";".join(f"{c}:{len(p)}" for c, p in sorted(channels.items())),
        })
    return pd.DataFrame(rows)


# =============================================================================
# [1]+[5] dataset completo
# =============================================================================
def build_dataset_multitool(segments_dir: Path = SEGMENTS_DIR,
                            target_file: Path = TARGET_FILE,
                            metadata_file: Optional[Path] = METADATA_FILE,
                            out_path: Path = PROCESSED_DATASET,
                            enable_frequency: bool = ENABLE_FREQUENCY_FEATURES,
                            aggs=None, use_cache: bool = True,
                            n_jobs: int = N_INGEST_WORKERS,
                            verbose: bool = True) -> pd.DataFrame:
    aggs = list(aggs) if aggs is not None else list(AGG_OVER_PARTS)
    segments_dir = Path(segments_dir)
    seg_index = scan_experiments(segments_dir)
    if not seg_index:
        raise ValueError(f"No hay segmentos parseables en {segments_dir}")

    # inventario (siempre, util para auditar entradas nuevas)
    inv = build_inventory(seg_index)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    inv.to_csv(METRICS_DIR / "ingest_inventory.csv", index=False)

    target_df = load_target_csv(target_file)
    target_df.columns = [str(c).strip() for c in target_df.columns]
    if TARGET_COLUMN not in target_df.columns:
        raise ValueError(f"target file no tiene columna {TARGET_COLUMN}")
    target_df[EXPERIMENT_ID_COL] = pd.to_numeric(
        target_df[EXPERIMENT_ID_COL], errors="coerce").astype("Int64")
    target_df[TARGET_COLUMN] = pd.to_numeric(target_df[TARGET_COLUMN], errors="coerce")
    target_df = target_df.dropna(subset=[EXPERIMENT_ID_COL, TARGET_COLUMN])
    target_df[EXPERIMENT_ID_COL] = target_df[EXPERIMENT_ID_COL].astype(int)
    # tool_id autoritativo desde el filename -> evitamos colision en el merge
    if TOOL_ID_COL in target_df.columns:
        target_df = target_df.drop(columns=[TOOL_ID_COL])

    target_ids = set(target_df[EXPERIMENT_ID_COL].tolist())

    # H2 — descartes silenciosos del inner-join: reportar y avisar.
    seg_ids = {int(k[1]) for k in seg_index}
    seg_no_target = sorted(seg_ids - target_ids)
    target_no_seg = sorted(target_ids - seg_ids)
    if seg_no_target:
        warnings.warn(f"[INGEST/H2] {len(seg_no_target)} experimentos con SENAL pero SIN "
                      f"target (se descartan): {seg_no_target[:10]}")
    if target_no_seg:
        warnings.warn(f"[INGEST/H2] {len(target_no_seg)} targets SIN senal (se descartan): "
                      f"{target_no_seg[:10]}")

    keys = sorted([k for k in seg_index if int(k[1]) in target_ids],
                  key=lambda k: (str(k[0]), int(k[1])))
    if not keys:
        raise ValueError("Ningun (tool, exp) tiene a la vez segmentos y target.")

    # H1 — completitud de canales: avisar de experimentos con canal faltante/vacio.
    all_channels = sorted({c for k in keys for c in seg_index[k]})
    incomplete = []
    for k in keys:
        missing = set(all_channels) - set(seg_index[k])
        empty = [c for c, parts in seg_index[k].items() if len(parts) == 0]
        if missing or empty:
            incomplete.append({
                TOOL_ID_COL: k[0], EXPERIMENT_ID_COL: int(k[1]),
                "missing_channels": ";".join(sorted(missing)),
                "empty_channels": ";".join(sorted(empty)),
            })
    if incomplete:
        warnings.warn(f"[INGEST/H1] {len(incomplete)} experimento(s) con CANAL faltante/vacio "
                      f"(features de ese canal seran NaN/imputadas). Ver ingest_incomplete.csv")
        pd.DataFrame(incomplete).to_csv(METRICS_DIR / "ingest_incomplete.csv", index=False)

    if verbose:
        chans = sorted({c for k in keys for c in seg_index[k]})
        n_tools = len({k[0] for k in keys})
        print(f"[INGEST] experimentos={len(keys)}  cuchillas={n_tools}  "
              f"canales={chans}  aggs={aggs}  cache={use_cache}")

    def _one(k):
        return build_experiment_features(k, seg_index[k], enable_frequency, aggs, use_cache)

    rows = None
    if n_jobs and n_jobs != 1 and len(keys) > 1:
        try:
            from joblib import Parallel, delayed
            rows = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_one)(k) for k in keys)
        except Exception as exc:
            warnings.warn(f"[INGEST] joblib no disponible ({exc}); secuencial.")
    if rows is None:
        rows = [_one(k) for k in keys]

    feats_df = pd.DataFrame(rows)
    dataset = feats_df.merge(target_df, on=EXPERIMENT_ID_COL, how="inner")

    # experiment_order: derivar por cuchilla si no viene del target
    if EXP_ORDER_COL not in dataset.columns:
        dataset = dataset.sort_values([TOOL_ID_COL, EXPERIMENT_ID_COL])
        dataset[EXP_ORDER_COL] = dataset.groupby(TOOL_ID_COL).cumcount() + 1

    # metadata opcional (no duplica columnas existentes)
    if metadata_file is not None and Path(metadata_file).exists():
        try:
            meta = pd.read_csv(metadata_file)
            meta.columns = [str(c).strip() for c in meta.columns]
            if EXPERIMENT_ID_COL in meta.columns:
                meta[EXPERIMENT_ID_COL] = pd.to_numeric(
                    meta[EXPERIMENT_ID_COL], errors="coerce").astype("Int64")
                keep = [c for c in meta.columns
                        if c == EXPERIMENT_ID_COL or c not in dataset.columns]
                dataset = dataset.merge(meta[keep], on=EXPERIMENT_ID_COL, how="left")
        except Exception as exc:
            warnings.warn(f"[INGEST] metadata no mergeada: {exc}")

    # orden de columnas: ids, features, target al final
    front = [c for c in (TOOL_ID_COL, EXPERIMENT_ID_COL, EXP_ORDER_COL) if c in dataset.columns]
    back = [TARGET_COLUMN]
    other = [c for c in dataset.columns if c not in front + back]
    dataset = dataset[front + other + back]
    dataset = dataset.sort_values([TOOL_ID_COL, EXP_ORDER_COL]).reset_index(drop=True)

    # C1 — provenance: avisar si sobrescribimos un dataset de OTRO builder.
    prev = read_dataset_meta(out_path)
    if prev and prev.get("builder") and prev["builder"] != "build_dataset_multitool":
        warnings.warn(f"[INGEST/C1] sobrescribiendo dataset de '{prev['builder']}' "
                      f"(schema {prev.get('schema')}). El benchmark/PINN cambian de esquema.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_path, index=False)
    write_dataset_meta(out_path, "build_dataset_multitool", DATASET_SCHEMA_MULTITOOL, dataset)

    # H3 — VB debe ser no-decreciente con experiment_order por cuchilla.
    vb_viol = []
    for tool, sub in dataset.groupby(TOOL_ID_COL):
        vb = sub.sort_values(EXP_ORDER_COL)[TARGET_COLUMN].values.astype(float)
        n_dec = int((np.diff(vb) < 0).sum())
        if n_dec > 0:
            vb_viol.append((tool, n_dec))
    if vb_viol:
        warnings.warn(f"[INGEST/H3] VB NO monotona con experiment_order en {len(vb_viol)} "
                      f"cuchilla(s): {vb_viol[:5]}. ¿experiment_id no es temporal? "
                      f"Considera proveer experiment_order en el target.")

    if verbose:
        n_feat = len([c for c in other if c not in (TOOL_ID_COL, EXPERIMENT_ID_COL, EXP_ORDER_COL)])
        print(f"[INGEST] guardado {out_path}  shape={dataset.shape}  (~{n_feat} features)")
    return dataset
