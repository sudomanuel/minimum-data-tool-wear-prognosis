#!/usr/bin/env python3
"""
run_p8_2_segmentation_features.py — P8.2 orchestrator.

Builds: data/manifest/raw_signal_manifest.csv, data/manifest/segments_manifest.csv,
        data/features/p8_2_features_segments.csv, data/features/p8_2_features_experiment.csv,
        overlay figures in outputs/figures/p8_2_segmentation_overlays/.

Reads raw signals from the main repo (--repo-root); writes outputs into the worktree.
No signals moved; no synthetic data; no model training. exp77 kept at 4 contacts (no
imputation); 71-72 are target-only rows in the raw_signal_manifest (no signals).

Uso:  python run.py p8-2-features  [--repo-root <repo-root>]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "src"))
from phm.segmentation_p8 import load_signal, detect_contact, segment_features, FS  # noqa: E402
from phm.targets import load_official_vb  # noqa: E402

MANIFEST = WT / "data" / "manifest"
FEATURES = WT / "data" / "features"
OVERLAYS = WT / "outputs" / "figures" / "p8_2_segmentation_overlays"
RECORDED = [66, 67, 68, 69, 70, 73, 74, 75, 76, 77]
TARGET_ONLY = [71, 72]
PHYS_ORDER = {e: i + 1 for i, e in enumerate([66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77])}
REC_ORDER = {e: i + 1 for i, e in enumerate(RECORDED)}
SEG_RE = re.compile(r"^(?P<ch>[AR])(?P<exp>\d+)_p(?P<c>\d+)\.txt$", re.I)
CH = {"A": "axial", "R": "rotational"}


def build_raw_manifest(repo_root, vb_map):
    seg_dir = repo_root / "data" / "raw" / "segments"
    rows = []
    present = {}
    for f in sorted(seg_dir.glob("*.txt")):
        m = SEG_RE.match(f.name)
        if not m:
            continue
        ch, exp, c = m["ch"].upper(), int(m["exp"]), int(m["c"])
        present.setdefault(exp, set()).add((ch, c))
        rows.append(dict(
            tool_id="T01", experiment_id=exp,
            physical_experiment_order=PHYS_ORDER[exp], recorded_signal_order=REC_ORDER.get(exp, ""),
            contact_id=c, channel=CH[ch], file_path=f"data/raw/segments/{f.name}",
            sampling_rate_hz=50000, has_signal=True, has_official_vb=exp in vb_map,
            VB_um=vb_map.get(exp, ""),
            valid_signal=True, missing_signal=False, target_only=False,
            incomplete_contact=False, valid_contact=True,
            exp77_four_contact_case=(exp == 77),
            quality_flag="recorded",
        ))
    # exp77 missing p5/p6 (both channels) — record as incomplete, no file
    for ch in ("A", "R"):
        for c in (5, 6):
            rows.append(dict(
                tool_id="T01", experiment_id=77, physical_experiment_order=PHYS_ORDER[77],
                recorded_signal_order=REC_ORDER[77], contact_id=c, channel=CH[ch], file_path="",
                sampling_rate_hz=50000, has_signal=False, has_official_vb=True, VB_um=vb_map.get(77, ""),
                valid_signal=False, missing_signal=True, target_only=False,
                incomplete_contact=True, valid_contact=False, exp77_four_contact_case=True,
                quality_flag="only_4_segmentable_peaks",
            ))
    # 71-72 target-only (no signals)
    for exp in TARGET_ONLY:
        rows.append(dict(
            tool_id="T01", experiment_id=exp, physical_experiment_order=PHYS_ORDER[exp],
            recorded_signal_order="", contact_id="", channel="", file_path="",
            sampling_rate_hz="", has_signal=False, has_official_vb=exp in vb_map,
            VB_um=vb_map.get(exp, ""), valid_signal=False, missing_signal=True, target_only=True,
            incomplete_contact=False, valid_contact=False, exp77_four_contact_case=False,
            quality_flag="target_only_missing_signal",
        ))
    df = pd.DataFrame(rows)
    MANIFEST.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST / "raw_signal_manifest.csv", index=False)
    return df, seg_dir


def segment_and_extract(raw_df, seg_dir):
    seg_rows, feat_rows = [], []
    sig_files = raw_df[raw_df.has_signal & (raw_df.file_path != "")]
    for _, r in sig_files.iterrows():
        path = seg_dir.parent.parent / Path(r.file_path).relative_to("data")
        # resolve against repo: file_path is repo-relative
        path = (seg_dir.parent.parent / r.file_path).resolve() if not path.exists() else path
        if not path.exists():
            path = seg_dir / Path(r.file_path).name
        t, v = load_signal(path)
        det = detect_contact(v, FS)
        seg_rows.append(dict(
            tool_id="T01", experiment_id=int(r.experiment_id), contact_id=int(r.contact_id),
            channel=r.channel, file_path=r.file_path,
            segment_start_sample=det["start"], segment_end_sample=det["end"],
            segment_duration_s=round((det["end"] - det["start"]) / FS, 4),
            peak_sample=det["peak_sample"], threshold_used=round(det["threshold"], 5),
            active_frac=round(det["active_frac"], 4),
            valid_segment=det["flag"] != "no_active_region",
            quality_flag=det["flag"], notes="full-contact window (envelope+adaptive thr+margin)",
        ))
        f = segment_features(v, det["start"], det["end"], FS)
        f.update(dict(experiment_id=int(r.experiment_id), contact_id=int(r.contact_id),
                      channel=r.channel, valid_segment=det["flag"] != "no_active_region",
                      quality_flag=det["flag"]))
        feat_rows.append(f)
        print(f"  {Path(r.file_path).name:12s} [{det['start']:>6}:{det['end']:>6}] "
              f"dur={ (det['end']-det['start'])/FS:5.2f}s flag={det['flag']}", flush=True)
    seg_df = pd.DataFrame(seg_rows)
    feat_df = pd.DataFrame(feat_rows)
    seg_df.to_csv(MANIFEST / "segments_manifest.csv", index=False)
    FEATURES.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(FEATURES / "p8_2_features_segments.csv", index=False)
    return seg_df, feat_df


def aggregate_experiment(feat_df, vb_map):
    base_cols = [c for c in feat_df.columns
                 if c not in ("experiment_id", "contact_id", "channel",
                              "valid_segment", "quality_flag")]
    rows = []
    cum_rms = {"axial": 0.0, "rotational": 0.0}
    cum_energy = {"axial": 0.0, "rotational": 0.0}
    cum_contacts = 0
    for exp in RECORDED:
        row = {"experiment_id": exp, "physical_experiment_order": PHYS_ORDER[exp],
               "recorded_signal_order": REC_ORDER[exp]}
        per_ch_means = {}
        valid_counts = {}
        for ch in ("axial", "rotational"):
            sub = feat_df[(feat_df.experiment_id == exp) & (feat_df.channel == ch)
                          & feat_df.valid_segment]
            pre = "A" if ch == "axial" else "R"
            valid_counts[ch] = len(sub)
            if len(sub) == 0:
                continue
            means = {}
            for c in base_cols:
                vals = sub[c].to_numpy(float)
                row[f"{pre}_{c}_mean"] = float(np.nanmean(vals))
                row[f"{pre}_{c}_std"] = float(np.nanstd(vals))
                row[f"{pre}_{c}_median"] = float(np.nanmedian(vals))
                means[c] = float(np.nanmean(vals))
            per_ch_means[ch] = means
            row[f"{pre}_contact_count_valid"] = len(sub)
            cum_rms[ch] += means.get("rms", 0.0)
            cum_energy[ch] += means.get("energy", 0.0)
            row[f"{pre}_cumulative_rms"] = cum_rms[ch]
            row[f"{pre}_cumulative_energy"] = cum_energy[ch]
        # fusion
        if "axial" in per_ch_means and "rotational" in per_ch_means:
            a, r = per_ch_means["axial"], per_ch_means["rotational"]
            for key in ("rms", "energy", "mav", "kurtosis", "spectral_centroid"):
                if key in a and key in r:
                    row[f"AR_{key}_ratio"] = a[key] / (r[key] + 1e-12)
                    row[f"AR_{key}_diff"] = a[key] - r[key]
            row["combined_rms"] = np.sqrt(a.get("rms", 0) ** 2 + r.get("rms", 0) ** 2)
            row["combined_energy"] = a.get("energy", 0) + r.get("energy", 0)
        # degradation-aware
        cum_contacts += sum(valid_counts.values())
        row["cumulative_contact_count"] = cum_contacts
        row["cumulative_experiment_order"] = PHYS_ORDER[exp]
        # reliability
        vc = min(valid_counts.get("axial", 0), valid_counts.get("rotational", 0))
        row["contact_count_valid"] = vc
        row["missing_contact_count"] = max(0, 6 - vc)
        row["feature_reliability"] = "partial" if exp == 77 else "full"
        row["energy_total_reliability"] = "low" if exp == 77 else "acceptable"
        row["rms_mean_reliability"] = "acceptable"
        row["VB_um"] = vb_map.get(exp, np.nan)   # OFFICIAL VB
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("physical_experiment_order").reset_index(drop=True)
    out.to_csv(FEATURES / "p8_2_features_experiment.csv", index=False)
    return out


def make_overlays(raw_df, seg_dir):
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    examples = [("A66_p1.txt", "axial normal"), ("R66_p1.txt", "rotational normal"),
                ("A77_p4.txt", "exp77 (4-contact case)"), ("A70_p3.txt", "axial mid-life")]
    for fn, title in examples:
        p = seg_dir / fn
        if not p.exists():
            continue
        t, v = load_signal(p)
        det = detect_contact(v, FS)
        from phm.segmentation_p8 import _smooth_envelope
        env = _smooth_envelope(v, max(1, int(0.005 * FS)))
        ds = max(1, len(v) // 4000)
        fig, ax = plt.subplots(figsize=(10, 3.4), dpi=130)
        tt = np.arange(len(v)) / FS
        ax.plot(tt[::ds], v[::ds], color="#9aa", lw=0.5, label="signal")
        ax.plot(tt[::ds], env[::ds], color="#1F4E79", lw=1.0, label="envelope")
        ax.axhline(det["threshold"], color="#B3541E", ls=":", lw=1.2, label="adaptive threshold")
        ax.axvspan(det["start"] / FS, det["end"] / FS, color="#4A6628", alpha=0.18,
                   label="full-contact window")
        ax.set_title(f"{fn} — {title}  (flag={det['flag']}, "
                     f"dur={(det['end']-det['start'])/FS:.2f}s)")
        ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude")
        ax.legend(fontsize=7, ncol=4, loc="upper center")
        fig.tight_layout()
        fig.savefig(OVERLAYS / f"overlay_{fn.replace('.txt','')}.png",
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  overlays -> {OVERLAYS}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    vb = load_official_vb(data_root=args.repo_root / "data" / "targets")
    vb_map = dict(zip(vb.experiment_id.astype(int), vb.VB_um.astype(float)))

    print("[1] raw_signal_manifest ...")
    raw_df, seg_dir = build_raw_manifest(args.repo_root, vb_map)
    print(f"    {len(raw_df)} rows ({int(raw_df.has_signal.sum())} signals, "
          f"{int((~raw_df.has_signal).sum())} target-only/missing)")
    print("[2] full-contact segmentation + features ...")
    seg_df, feat_df = segment_and_extract(raw_df, seg_dir)
    print(f"    {len(seg_df)} segments; valid={int(seg_df.valid_segment.sum())}")
    print("[3] aggregate to experiment ...")
    exp_df = aggregate_experiment(feat_df, vb_map)
    print(f"    {len(exp_df)} experiments x {exp_df.shape[1]} cols")
    print("[4] overlays ...")
    make_overlays(raw_df, seg_dir)
    print("\nDONE. Outputs in data/manifest/, data/features/, outputs/figures/p8_2_segmentation_overlays/")
    # quick QA echo
    nfeat = feat_df.shape[1] - 5
    print(f"segment features: {nfeat} | experiment cols: {exp_df.shape[1]} | "
          f"NaN cells exp-table: {int(exp_df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
