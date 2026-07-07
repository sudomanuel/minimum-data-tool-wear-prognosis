"""run_r4_severity_mondrian.py — R4: conditional conformal validity under DOE heteroscedasticity.

Question: does the pooled (horizon-only) Mondrian band hide under-coverage on SEVERE cutting conditions?
Design: severity s = vc*fz per tool, median split {mild, severe} (9/9). Three taxonomies compared on the
record configs (m=3 Siegel+local-p; m=4 WLS(tau)+m-matched+local-p), all leakage-safe per-tool:
  H   : horizon-only bins near<=1 / mid 2-3 / far>=4   (current deployed band)
  SxH : severity x horizon 2x2 (near<=1 vs far>=2)     (bin budget: ~118/4 ~ 29 per bin, valid)
  S   : severity-only (2 bins)
Report: per-severity PICP under EACH taxonomy (the diagnostic), mean width, and the acceptance rule:
adopt SxH only if per-severity PICP >= 88% in BOTH clusters and mean width <= current +10%.
Outputs: results/r4_severity_mondrian.csv
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_record_attempts import fewshot_pred, cq, load
AL = 0.10
FEAT = os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv")


def tool_severity():
    f = pd.read_csv(FEAT)
    g = f.groupby("tool_id")[["vc", "fz"]].first()
    s = (g.vc * g.fz)
    med = s.median()
    return {t: ("severe" if v > med else "mild") for t, v in s.items()}, s.to_dict()


def band(res, sev, taxonomy):
    """Per-tool coverage/width + per-severity coverage. taxonomy in {'H','SxH','S'}."""
    tools = list(res)
    cov_tool, wid_tool = [], []
    cov_by_sev = {"mild": [], "severe": []}
    wid_by_sev = {"mild": [], "severe": []}
    for tt in tools:
        cal_r = np.concatenate([res[t][2] for t in tools if t != tt])
        cal_h = np.concatenate([res[t][1] for t in tools if t != tt])
        cal_s = np.concatenate([[sev[t]] * len(res[t][2]) for t in tools if t != tt])
        _, hh, rr = res[tt]
        tc, tw = [], []
        for h, r in zip(hh, rr):
            if taxonomy == "H":
                sel = (cal_h <= 1) if h <= 1 else ((cal_h >= 2) & (cal_h <= 3) if h <= 3 else cal_h >= 4)
            elif taxonomy == "S":
                sel = cal_s == sev[tt]
            else:  # SxH
                hs = cal_h <= 1 if h <= 1 else cal_h >= 2
                sel = hs & (cal_s == sev[tt])
            q = cq(cal_r[sel], 1 - AL) if sel.sum() >= 9 else cq(cal_r, 1 - AL)
            tc.append(r <= q); tw.append(2 * q)
        cov_tool.append(np.mean(tc)); wid_tool.append(np.mean(tw))
        cov_by_sev[sev[tt]].append(np.mean(tc)); wid_by_sev[sev[tt]].append(np.mean(tw))
    out = dict(PICP=100 * np.mean(cov_tool), width=np.mean(wid_tool))
    for s in ("mild", "severe"):
        out[f"PICP_{s}"] = 100 * np.mean(cov_by_sev[s]); out[f"width_{s}"] = np.mean(wid_by_sev[s])
    return out


def main():
    d = load(); sev, sval = tool_severity()
    n_sev = sum(1 for v in sev.values() if v == "severe")
    print(f"R4 severity-conditioned Mondrian. severity = vc*fz, median split -> "
          f"{18 - n_sev} mild / {n_sev} severe.\n")
    rows = []
    for m, est, ps, pl in [(3, "siegel", "full", 200.0), (4, "wls_tau", "m_matched", 200.0)]:
        res = fewshot_pred(d, m, est, ps, pl)
        print(f"--- m={m} record config ({est}) ---")
        print(f"{'taxonomy':6} {'PICP':>6} {'width':>7} | {'mild PICP':>10} {'sev PICP':>9} "
              f"{'mild w':>7} {'sev w':>7}")
        for tax in ("H", "SxH", "S"):
            r = band(res, sev, tax)
            rows.append(dict(m=m, taxonomy=tax, **{k: round(v, 1) for k, v in r.items()}))
            print(f"{tax:6} {r['PICP']:5.1f}% {r['width']:6.1f}u | {r['PICP_mild']:9.1f}% "
                  f"{r['PICP_severe']:8.1f}% {r['width_mild']:6.1f}u {r['width_severe']:6.1f}u")
        cur = [x for x in rows if x["m"] == m and x["taxonomy"] == "H"][0]
        sxh = [x for x in rows if x["m"] == m and x["taxonomy"] == "SxH"][0]
        diag = ("UNDER-COVERAGE on severe cells under H" if cur["PICP_severe"] < 85
                else "no material severity under-coverage under H")
        ok = (sxh["PICP_mild"] >= 88 and sxh["PICP_severe"] >= 88
              and sxh["width"] <= cur["width"] * 1.10)
        print(f"  diagnostic: {diag}")
        print(f"  acceptance(SxH): {'ADOPT' if ok else 'REJECT'} "
              f"(mild {sxh['PICP_mild']}%, severe {sxh['PICP_severe']}%, width {sxh['width']} vs "
              f"{cur['width']}u)\n")
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "r4_severity_mondrian.csv"), index=False)
    print("wrote results/r4_severity_mondrian.csv")


if __name__ == "__main__":
    main()
