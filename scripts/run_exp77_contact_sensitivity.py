#!/usr/bin/env python3
"""
run_exp77_contact_sensitivity.py — P7.1 diagnostic: 4-vs-6 contact sensitivity.

Quantifies how unreliable experiment-level features become when only contacts
p1-p4 exist (the exp-77 pattern), via leave-contacts-out simulation on the 9
complete experiments + a median-imputation error proxy for per-contact columns.

READ-ONLY on real data. No imputation written, no signals reconstructed.

Uso:
    python run.py exp77-sensitivity
    python run.py exp77-sensitivity --data-root D:/otra/data
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT_CSV = Path("outputs/audit/p7_1_contact_sensitivity_summary.csv")

BASE_FEATURES = ["mean", "std", "rms", "max", "min", "peak_to_peak", "skewness",
                 "kurtosis", "crest_factor", "energy", "absolute_mean",
                 "dominant_freq_hz", "spectral_energy", "spectral_centroid_hz"]
DIR2CODE = {"axial": "A", "rotational": "R"}


def aggregates_from_contacts(df_exp: pd.DataFrame, contacts) -> dict:
    """Recompute the 9 pipeline aggregates (dataset_builder definitions) on a contact subset."""
    sub = df_exp[df_exp["contact_id"].isin(contacts)]
    out = {}
    per_dir = {}
    for direction, code in DIR2CODE.items():
        d = sub[sub["direction"] == direction].sort_values("contact_id")
        rms = d["rms"].to_numpy(dtype=float)
        erg = d["energy"].to_numpy(dtype=float)
        per_dir[code] = d.set_index("contact_id")[["rms", "energy"]]
        out[f"{code}_rms_mean_6_contacts"] = float(np.nanmean(rms)) if len(rms) else np.nan
        out[f"{code}_rms_std_6_contacts"] = float(np.nanstd(rms)) if len(rms) else np.nan
        out[f"{code}_energy_total_6_contacts"] = float(np.nansum(erg)) if len(erg) else np.nan
    both = np.concatenate([per_dir["A"]["energy"].to_numpy(float),
                           per_dir["R"]["energy"].to_numpy(float)])
    out["total_energy_6_contacts"] = float(np.nansum(both)) if len(both) else np.nan
    common = per_dir["A"].index.intersection(per_dir["R"].index)
    a = per_dir["A"].loc[common]
    r = per_dir["R"].loc[common]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_rms = np.where(r["rms"].to_numpy(float) > 0,
                             a["rms"].to_numpy(float) / r["rms"].to_numpy(float), np.nan)
        ratio_erg = np.where(r["energy"].to_numpy(float) > 0,
                             a["energy"].to_numpy(float) / r["energy"].to_numpy(float), np.nan)
    out["A_to_R_rms_ratio"] = float(np.nanmean(ratio_rms)) if len(common) else np.nan
    out["A_to_R_energy_ratio"] = float(np.nanmean(ratio_erg)) if len(common) else np.nan
    return out


def axis_of(feature_name: str) -> str:
    if feature_name.startswith("A_"):
        return "A"
    if feature_name.startswith("R_"):
        return "R"
    return "both"


def family_of(feature_name: str) -> str:
    for fam in ("rms_mean", "rms_std", "energy_total", "rms_ratio", "energy_ratio"):
        if fam.replace("_", "") in feature_name.replace("_", ""):
            return fam
    if "total_energy" in feature_name:
        return "energy_total"
    return "other"


def flag(err: float) -> str:
    if np.isnan(err):
        return "no_data"
    if err < 0.05:
        return "robust"
    if err < 0.15:
        return "moderate"
    return "unstable"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=None)
    args = ap.parse_args()

    if args.data_root is not None:
        cf_path = args.data_root / "interim" / "contact_features.csv"
    else:
        from phm.config import INTERIM_DIR
        cf_path = INTERIM_DIR / "contact_features.csv"

    cf = pd.read_csv(cf_path)
    complete = [e for e, g in cf.groupby("experiment_id")
                if g.dropna(subset=["rms"]).groupby("direction")["contact_id"].count().min() == 6]
    print(f"experimentos completos (6/6 ambos ejes): {sorted(complete)}")

    full_contacts = list(range(1, 7))
    pattern77 = [1, 2, 3, 4]
    pairs = list(itertools.combinations(full_contacts, 2))

    # --- A. leave-contacts-out on aggregates ---
    rec = []
    for e in complete:
        g = cf[cf["experiment_id"] == e]
        ref = aggregates_from_contacts(g, full_contacts)
        for pair in pairs:
            kept = [c for c in full_contacts if c not in pair]
            sub = aggregates_from_contacts(g, kept)
            for feat, ref_v in ref.items():
                if ref_v == 0 or np.isnan(ref_v):
                    continue
                rel = (sub[feat] - ref_v) / abs(ref_v)
                rec.append({"experiment_id": e, "dropped": f"p{pair[0]}p{pair[1]}",
                            "feature_name": feat, "rel_err": rel,
                            "is_pattern77": pair == (5, 6)})
    rec = pd.DataFrame(rec)

    rows = []
    p77 = rec[rec["is_pattern77"]]
    for feat, g in p77.groupby("feature_name"):
        abs_err = g["rel_err"].abs()
        signed = g["rel_err"].mean()
        allp = rec[rec["feature_name"] == feat]["rel_err"].abs()
        bias_txt = f"signed bias dropping p5p6: {signed:+.1%}"
        rows.append({
            "feature_name": feat, "sensor_axis": axis_of(feat),
            "feature_family": family_of(feat),
            "mean_relative_error_4vs6": round(float(abs_err.mean()), 4),
            "max_relative_error_4vs6": round(float(abs_err.max()), 4),
            "reliability_flag": flag(abs_err.mean()),
            "notes": f"aggregate; {bias_txt}; all-pairs p95 |err|: {allp.quantile(0.95):.1%}",
        })

    # --- B. median-imputation proxy for per-contact p5/p6 columns ---
    for direction, code in DIR2CODE.items():
        d = cf[(cf["direction"] == direction) & (cf["contact_id"].isin([5, 6]))]
        for feat in BASE_FEATURES:
            errs = []
            for e in complete:
                for cidx in (5, 6):
                    true = d[(d["experiment_id"] == e) & (d["contact_id"] == cidx)][feat]
                    others = d[(d["experiment_id"] != e) & (d["contact_id"] == cidx)][feat]
                    if len(true) == 1 and pd.notna(true.iloc[0]) and true.iloc[0] != 0 and others.notna().sum() > 2:
                        errs.append(abs(others.median() - true.iloc[0]) / abs(true.iloc[0]))
            if not errs:
                continue
            errs = pd.Series(errs)
            rows.append({
                "feature_name": f"{code}_{feat}_p5/p6 (per-contact)",
                "sensor_axis": code, "feature_family": feat,
                "mean_relative_error_4vs6": round(float(errs.mean()), 4),
                "max_relative_error_4vs6": round(float(errs.max()), 4),
                "reliability_flag": flag(errs.mean()),
                "notes": "median-imputation error proxy (what the pipeline imputes at exp 77)",
            })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # --- C. contact-trend diagnostic ---
    print("\ncontact-index trend (Spearman rho of feature vs contact_id, complete exps):")
    from scipy import stats
    for direction in ("axial", "rotational"):
        for feat in ("rms", "energy"):
            d = cf[(cf["direction"] == direction) & cf["experiment_id"].isin(complete)]
            rho = stats.spearmanr(d["contact_id"], d[feat], nan_policy="omit").statistic
            print(f"  {direction:<10} {feat:<7} rho={rho:+.3f}")

    agg = out[out["notes"].str.startswith("aggregate")]
    print(f"\n{OUT_CSV}: {len(out)} rows ({len(agg)} aggregates + {len(out)-len(agg)} per-contact proxies)")
    print("\nAGGREGATES (exp-77 pattern p5p6 dropped):")
    print(agg[["feature_name", "mean_relative_error_4vs6", "max_relative_error_4vs6",
               "reliability_flag", "notes"]].to_string(index=False))
    pc = out[~out["notes"].str.startswith("aggregate")]
    print("\nPER-CONTACT imputation proxy, worst 8:")
    worst = pc.sort_values("mean_relative_error_4vs6", ascending=False).head(8)
    print(worst[["feature_name", "mean_relative_error_4vs6", "reliability_flag"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
