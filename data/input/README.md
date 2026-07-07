# `data/input/` — multi-tool ingestion folder

Drop new cutting tools here (T02, T03, …) so they can enter the **multi-tool,
leakage-safe** validation (LOTO + few-shot) without touching code. This folder is
the single entry point for the new research focus:

> **Minimum-Data Physics-Integrated Adaptive PINN for VB→HI→RUL Prognostics** —
> learn a global wear law from **K** historical tools, adapt to an unseen tool
> with **m** early VB labels, predict future VB, derive HI/RUL.

T01 is the **frozen single-tool baseline** (temporal-degeneracy diagnostic); it is
already processed and lives in the main `data/` tree. Do **not** mix new tools into
T01's results — they enter the *new* multi-tool experiments only.

---

## How to add a tool (3 steps)

1. **Copy the template folder** `tools/_TEMPLATE_Txx/` to `tools/T0X/` (use the real
   tool id, e.g. `T02`).
2. **Drop the files** into its subfolders:
   - `signals/` — raw axial + rotational vibration recordings (one file per
     experiment, or as acquired). Keep the original filenames; record the mapping
     in the manifest.
   - `vb/vb_measurements.csv` — the microscope VB trajectory (see template).
   - `photos/` — microscope photographs (optional; used only for target provenance
     / qualitative wear-stage audit, **not** as a deep-learning image source).
   - `meta/` — optional cutting conditions and the failure/breakage record.
3. **Fill the manifest** `tools/T0X/manifest.csv` (copy from
   `templates/tool_manifest_template.csv`) and **validate**:
   ```
   python scripts/validate_tool_input.py --tool T0X
   ```
   The validator reports missing files, schema violations, and the usable-row
   counts per dataset view. Fix any FAIL before the tool enters validation.

When a tool has a recorded breakage, also add a block to
[`config/failure_events.yaml`](../../config/failure_events.yaml) (T_L, and T_R if
the breakage experiment order is known; otherwise leave T_R null → right-censored).

---

## The three dataset views (kept separate — see `schema/DATASET_SCHEMAS.md`)

A single experiment can be usable for some views and not others. The manifest
flags this per experiment so the pipeline never silently mixes them:

| View | What it is | Used for |
|---|---|---|
| **sensor-based** | experiments with valid signal + valid contacts | classical sensor models, sensor-driven wear-rate |
| **trajectory-based VB** | every experiment with a measured VB (even signal-less) | the VB(t) curve, PINN integral, HI |
| **RUL / breakage / censoring** | tools with an observed breakage or threshold crossing | interval-/right-censored RUL evaluation |

---

## Non-negotiable rules (enforced by the validator + leakage audit)

- A held-out **test tool never** participates in global preprocessing, feature
  selection, tuning, augmentation, or synthetic generation.
- In few-shot, **only the first `m`** VB-labelled points of the test tool may be
  used for adaptation; **all later points stay sealed** for evaluation.
- Augmentation/synthetic rows are **train-fold only**, never test evidence.
- No random per-experiment split as a headline result — splits are **per tool**.
- Two scarcity axes are kept distinct everywhere:
  **K** = number of training tools (global law); **m** = early test-tool labels (adaptation).

See [`reports/multitool_manifest_requirements.md`](../../reports/multitool_manifest_requirements.md)
for the full field spec and [`reports/minimum_data_protocol.md`](../../reports/minimum_data_protocol.md)
for the K×m experimental design.
