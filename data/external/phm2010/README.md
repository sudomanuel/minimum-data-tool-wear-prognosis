# PHM2010 Data Challenge — per-cut wear labels (external validation set)

Public benchmark: 2010 PHM Society Conference Data Challenge (high-speed CNC milling, ball-nose
cutter, 315 cuts per tool; flank wear measured ex situ after every cut with a LEICA microscope,
reported per flute in 10^-3 mm = µm). Cutters c1, c4, c6 are the labeled records.

## Files
- `phm2010_wear_percut.csv` — tidy per-cut wear: `cutter, cut (1..315), max_wear (µm, max over the
  three flutes)`. 945 rows. This is the ONLY input the external check uses (no signals needed).
- `c4_wear_flutes.csv`, `c6_wear_flutes.csv` — original per-flute tables kept for provenance.

## Provenance & authentication (2026-07-05)
- `max_wear` series obtained from the processed table in
  https://github.com/TemilolaG/phm2010-tool-wear-monitoring (data/processed/train_final.csv).
- Cross-validated EXACTLY (max abs diff 0.000000 µm, 315/315 cuts) for c4 and c6 against the
  original per-flute tables mirrored at https://github.com/qingluM/CDAR (01Tool wear/data/), taking
  the max over flutes.
- c1 cross-validated against an independent second source
  (https://github.com/katulu-io/uniwear-dataset, data/uniwear.csv, stepwise `tool_wear` for
  dataset_tag=phm2010): values match exactly (e.g., cut 1 = 48.893 µm, cut 2 = 49.571 µm).
- Sanity: 315 cuts per cutter; wear monotone non-decreasing (0 reversals); ranges 48.9–172.7 (c1),
  31.4–210.9 (c4), 62.8–234.7 (c6) µm.
- Original challenge page: https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/

## Role in the paper
External sanity check ONLY (Appendix): the paper's few-shot machinery is run ON this campaign as an
independent dataset (LOTO across the 3 cutters, sparse inspection cadence subsampled from the dense
record). It is NOT used to train, calibrate, or tune anything reported on the 18-tool campaign —
out-of-domain transfer INTO our model remains ruled out.
