# T01 Manifests — lineage

Single source of truth (as of 2026-06-13, after P7.3 official VB source lock):

- **`t01_canonical_manifest.csv`** — **MASTER**. Experiment-level (12 rows). Official target =
  **VB** (confirmed); **official source = `data/targets/microscope_vb.csv`, column `VB_um`**
  (12 experiments, full physical trajectory, range 103–212 µm); origin = in-house laboratory
  data; sampling = 50 kHz; 71–72 = target-only (performed, signals not recorded, VB result
  exists); exp 77 = end-of-life with 4 valid contacts. The legacy `vb_targets.csv` value
  (range 85–280, 10 exps) is kept per-row as `VB_um_legacy_vb_targets` for traceability and is
  marked `legacy_model_ready_subset`.

> **P7.3 lock:** the file once named `microscope_vs.csv` was a VB measurement of all performed
> experiments with a misnamed `VS` column. It is now `microscope_vb.csv` with column `VB_um`
> and is the OFFICIAL VB source. The pre-cleanup file is backed up at
> `data/targets/legacy/microscope_vs_legacy.csv`. `config/targets.yaml` is the machine-readable
> lock. Because P1–P6 used `vb_targets.csv` (different values), they are historical baseline
> only — a re-baseline on the official source is required for P8+.

Superseded snapshots (kept for history, do NOT consume):

- `t01_provisional_manifest.csv` — P7.0 pre-confirmation snapshot (status fields say
  "unconfirmed"; superseded by canonical).
- `t01_canonical_definition.csv` — P7 experiment-level draft (pre-confirmation).
- `t01_manifest.csv` — P7 per-contact manifest (120 rows); still the per-contact reference,
  but target/origin status superseded by the canonical experiment-level file.

Naming note: a file named `microscope_vs.csv` exists with column `VS_final_um`. **"VS" is a
misnamed/legacy label, not a distinct physical variable.** It conceptually holds microscope
wear readings (same quantity family as VB). It is auxiliary/non-official and must NOT be used
as a modeling target or introduced as "VS" in the paper. The open item is which file is the
authoritative VB SOURCE (values differ from `vb_targets.csv`), not a choice between two targets.
