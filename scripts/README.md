# scripts/

Entry points for the project. Prefer `python run.py <task>` from the repo
root; the table below is the underlying map.

## Canonical flow

| Order | Script | Role | `run.py` task |
|------:|--------|------|---------------|
| 1 | `ingest.py` | multi-cutter raw signals to `experiment_features.csv` (cached, parallel, aggregate-over-parts) | `ingest` |
| 1b | `build_dataset.py` | legacy T01 builder (per-contact features) | `dataset` |
| 2 | `run_layered_pipeline.py` | 36-branch LOEO benchmark; rankings, figures, SHAP | `benchmark` |
| 3 | `build_pipeline_presentation.py` / `_en.py` | rebuild the decks from tracked CSVs | `report` |

## Physics / PINN track

| Script | Role | `run.py` task |
|--------|------|---------------|
| `validate_physics_assumptions.py` | check monotonicity, acceleration, rate-energy | `physics` |
| `run_pinn.py` | train + evaluate the wear-curve PINN under LOEO (config search) | `pinn` |
| `validate_bnn_entrant.py` | quick BNN vs ElasticNet vs Dummy check | — |

The earlier Bayesian-PINN calibration scripts (`validate_c0`, `calibrate_c1`,
`calibrate_c2`) are in `legacy/`: the approach moved to a deterministic PINN.

## Utilities

| Script | Role | `run.py` task |
|--------|------|---------------|
| `build_architecture_deck.py` | architecture diagrams + PPTX (in-use components) | `arch` |
| `check_feature_redundancy.py` | feature overlap (correlation + VIF) | — |
| `audit_data.py` | raw-data inventory and quality plots | — |
| `audit_outputs.py` | audit generated figures and CSVs | — |
| `regen_modified_figures.py` | regenerate only the layered-pipeline figures | — |

## Subfolders

- `legacy/` — the deprecated linear (hold-out) pipeline, superseded by
  `run_layered_pipeline.py`. Kept for provenance.
- `experiments/` — exploratory one-off analyses, not part of any flow.
