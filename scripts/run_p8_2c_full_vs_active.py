#!/usr/bin/env python3
"""
run_p8_2c_full_vs_active.py — P8.2C: build the full-contact-original feature branch and
compare it against the active-window-refined branch. NO modeling, NO synthetic data.

Two segmentation sources (both branch_candidate until P8.3 decides on performance):
  A) full_contact_original : features over the WHOLE per-contact file (no internal crop).
  B) active_window_refined : features over [segment_start_sample : segment_end_sample]
                             (the P8.2 envelope/threshold/margin window).

Reads raw signals from --repo-root; reuses the active-window boundaries already in
data/manifest/segments_manifest.csv. Writes branch tables + a comparison CSV.

Uso:  python run.py p8-2c-compare  [--repo-root D:/KSF/PHM/phm_tool_wear]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "src"))
from phm.segmentation_p8 import load_signal, segment_features, FS  # noqa: E402
from phm.targets import load_official_vb  # noqa: E402

MANIFEST = WT / "data" / "manifest"
FEATURES = WT / "data" / "features"
AUDIT = WT / "outputs" / "audit"
RECORDED = [66, 67, 68, 69, 70, 73, 74, 75, 76, 77]
PHYS_ORDER = {e: i + 1 for i, e in enumerate([66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77])}
REC_ORDER = {e: i + 1 for i, e in enumerate(RECORDED)}
CHPRE = {"axial": "A", "rotational": "R"}


def aggregate(feat_df, vb_map, source):
    """Aggregate segment features to experiment level (mean/std/median + fusion + degradation)."""
    base = [c for c in feat_df.columns if c not in
            ("experiment_id", "contact_id", "channel", "valid_segment", "quality_flag",
             "segmentation_source")]
    rows = []
    cum_rms = {"axial": 0.0, "rotational": 0.0}
    cum_e = {"axial": 0.0, "rotational": 0.0}
    cum_contacts = 0
    for exp in RECORDED:
        row = {"experiment_id": exp, "physical_experiment_order": PHYS_ORDER[exp],
               "recorded_signal_order": REC_ORDER[exp], "segmentation_source": source}
        per_ch, vc = {}, {}
        for ch in ("axial", "rotational"):
            sub = feat_df[(feat_df.experiment_id == exp) & (feat_df.channel == ch)
                          & feat_df.valid_segment]
            pre = CHPRE[ch]
            vc[ch] = len(sub)
            if not len(sub):
                continue
            means = {}
            for c in base:
                v = sub[c].to_numpy(float)
                row[f"{pre}_{c}_mean"] = float(np.nanmean(v))
                row[f"{pre}_{c}_std"] = float(np.nanstd(v))
                row[f"{pre}_{c}_median"] = float(np.nanmedian(v))
                means[c] = float(np.nanmean(v))
            per_ch[ch] = means
            row[f"{pre}_contact_count_valid"] = len(sub)
            cum_rms[ch] += means.get("rms", 0.0)
            cum_e[ch] += means.get("energy", 0.0)
            row[f"{pre}_cumulative_rms"] = cum_rms[ch]
            row[f"{pre}_cumulative_energy"] = cum_e[ch]
        if "axial" in per_ch and "rotational" in per_ch:
            a, r = per_ch["axial"], per_ch["rotational"]
            for k in ("rms", "energy", "mav", "kurtosis", "spectral_centroid"):
                if k in a and k in r:
                    row[f"AR_{k}_ratio"] = a[k] / (r[k] + 1e-12)
                    row[f"AR_{k}_diff"] = a[k] - r[k]
            row["combined_rms"] = np.sqrt(a.get("rms", 0) ** 2 + r.get("rms", 0) ** 2)
            row["combined_energy"] = a.get("energy", 0) + r.get("energy", 0)
        cum_contacts += sum(vc.values())
        row["cumulative_contact_count"] = cum_contacts
        row["cumulative_experiment_order"] = PHYS_ORDER[exp]
        v = min(vc.get("axial", 0), vc.get("rotational", 0))
        row["contact_count_valid"] = v
        row["missing_contact_count"] = max(0, 6 - v)
        row["feature_reliability"] = "partial" if exp == 77 else "full"
        row["energy_total_reliability"] = "low" if exp == 77 else "acceptable"
        row["rms_mean_reliability"] = "acceptable"
        row["VB_um"] = vb_map.get(exp, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("physical_experiment_order").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path("D:/KSF/PHM/phm_tool_wear"))
    args = ap.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)
    vb = load_official_vb(data_root=args.repo_root / "data" / "targets")
    vb_map = dict(zip(vb.experiment_id.astype(int), vb.VB_um.astype(float)))

    sm = pd.read_csv(MANIFEST / "segments_manifest.csv")
    full_rows, active_rows, cmp_rows = [], [], []
    for _, r in sm.iterrows():
        path = args.repo_root / r.file_path
        t, v = load_signal(path)
        n = len(v)
        s, e = int(r.segment_start_sample), int(r.segment_end_sample)
        ff = segment_features(v, 0, n - 1, FS)          # full contact
        af = segment_features(v, s, e, FS)              # active window
        meta = dict(experiment_id=int(r.experiment_id), contact_id=int(r.contact_id),
                    channel=r.channel, valid_segment=bool(r.get("valid_segment", True))
                    if "valid_segment" in r else True, quality_flag=r.get("quality_flag", ""))
        full_rows.append({**ff, **meta, "segmentation_source": "full_contact_original"})
        active_rows.append({**af, **meta, "segmentation_source": "active_window_refined"})
        cmp_rows.append(dict(
            experiment_id=int(r.experiment_id), contact_id=int(r.contact_id), channel=r.channel,
            original_n_samples=n, active_window_n_samples=(e - s + 1),
            retained_fraction=round((e - s + 1) / n, 4),
            original_duration_s=round(n / FS, 4), active_duration_s=round((e - s + 1) / FS, 4),
            rms_full=round(ff["rms"], 4), rms_active=round(af["rms"], 4),
            energy_full=ff["energy"], energy_active=af["energy"],
            waveform_length_full=round(ff["waveform_length"], 2),
            waveform_length_active=round(af["waveform_length"], 2),
            dominant_frequency_full=round(ff["dominant_freq"], 2),
            dominant_frequency_active=round(af["dominant_freq"], 2),
            notes="" if r.get("quality_flag", "") == "valid" else str(r.get("quality_flag", "")),
        ))
        print(f"  {Path(r.file_path).name:12s} retained={(e-s+1)/n:5.1%} "
              f"rms {ff['rms']:6.1f}->{af['rms']:6.1f}", flush=True)

    full_seg = pd.DataFrame(full_rows)
    active_seg = pd.DataFrame(active_rows)
    full_seg.to_csv(FEATURES / "p8_2_features_segments_full_contact.csv", index=False)
    # active-window copy WITH segmentation_source (alias of the existing table)
    active_seg.to_csv(FEATURES / "p8_2_features_segments_active_window.csv", index=False)

    full_exp = aggregate(full_seg, vb_map, "full_contact_original")
    active_exp = aggregate(active_seg, vb_map, "active_window_refined")
    full_exp.to_csv(FEATURES / "p8_2_features_experiment_full_contact.csv", index=False)
    active_exp.to_csv(FEATURES / "p8_2_features_experiment_active_window.csv", index=False)

    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(AUDIT / "p8_2_full_vs_active_window_comparison.csv", index=False)

    # summary stats for the report
    print("\n=== SUMMARY ===")
    print(f"mean retained fraction: {cmp.retained_fraction.mean():.1%} "
          f"(min {cmp.retained_fraction.min():.1%}, max {cmp.retained_fraction.max():.1%})")
    print(f"mean original dur {cmp.original_duration_s.mean():.2f}s -> active {cmp.active_duration_s.mean():.2f}s")
    for col_f, col_a, name in [("rms_full", "rms_active", "RMS"),
                               ("energy_full", "energy_active", "energy"),
                               ("waveform_length_full", "waveform_length_active", "waveform_length"),
                               ("dominant_frequency_full", "dominant_frequency_active", "dominant_freq")]:
        rel = (cmp[col_a] - cmp[col_f]) / (cmp[col_f].abs() + 1e-12)
        print(f"  {name:16s} median rel change active vs full: {rel.median():+.1%} "
              f"(p10 {rel.quantile(.1):+.1%}, p90 {rel.quantile(.9):+.1%})")
    print(f"\nbranches written: full_contact + active_window (segments & experiment). "
          f"comparison -> {AUDIT/'p8_2_full_vs_active_window_comparison.csv'}")


if __name__ == "__main__":
    main()
