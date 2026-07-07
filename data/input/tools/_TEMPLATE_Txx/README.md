# Tool template — copy this folder to `tools/T0X/` and fill it

```
tools/T0X/
  manifest.csv          # copy from data/input/templates/tool_manifest_template.csv
  signals/              # raw axial+rotational vibration (one file per experiment)
  vb/vb_measurements.csv# microscope VB trajectory (template provided)
  photos/               # microscope photographs (optional)
  meta/                 # optional: cutting_conditions.csv, failure_events row
```

Then run: `python scripts/validate_tool_input.py --tool T0X`

Rules: see `../../README.md` and `../../schema/DATASET_SCHEMAS.md`.
Do not delete `_TEMPLATE_Txx/`; copy it.
