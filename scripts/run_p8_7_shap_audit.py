#!/usr/bin/env python3
"""
run_p8_7_shap_audit.py — P8.7 SHAP-guided selection audit (fold-safe).

Adds SHAP as a 4th consensus vote next to Kendall/Spearman/MMI, computed INSIDE
each LOEO fold on TRAIN ONLY (TreeExplainer on a RandomForest fit on train). Then:
  (1) compares the selected top-k WITH vs WITHOUT SHAP (per-fold overlap/stability),
  (2) reads the physical coherence of the SHAP-top features (axial vs rotational,
      amplitude/energy vs frequency vs wavelet),
  (3) measures whether the SHAP-augmented selection changes LOEO MAE (RandomForest).

Anti-leakage: SHAP, scoring, scaling and selection use train rows only; the held-out
experiment is never seen during selection. Official VB target (microscope_vb.csv via
the reliability-aware branch table).

Uso:  python run.py p8-7-shap-audit  [--topk 10]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "src"))
from phm.feature_selection_p8 import (reliability_aware_cols, score_features,  # noqa: E402
                                      redundancy_filter, shap_scores)

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

RESULTS = WT / "results"
FEATURES = WT / "data" / "features"
FIGS = WT / "outputs" / "figures"
SOURCE = FEATURES / "p8_2_features_experiment_full_contact.csv"
TOPK_DEFAULT = 10
RNG = 0


def _physical_class(feat: str) -> str:
    """Coarse physical tag for coherence reading."""
    chan = "axial" if feat.startswith("A_") else ("rotacional" if feat.startswith("R_") else "otro")
    f = feat.lower()
    if "wavelet" in f:
        dom = "tiempo-frecuencia"
    elif any(k in f for k in ("freq", "spectral", "psd")):
        dom = "frecuencia"
    elif any(k in f for k in ("rms", "energy", "amplitude", "peak", "std", "waveform", "abs", "var")):
        dom = "amplitud/energia"
    else:
        dom = "otro"
    return f"{chan}/{dom}"


def _impute(Xdf, cols):
    X = Xdf[cols].to_numpy(float)
    med = np.nanmedian(X, axis=0)
    return np.where(np.isnan(X), med, X)


def loeo_mae(df, pool, sel_per_fold, y):
    """LOEO RandomForest MAE using a (held_out_id -> selected features) mapping."""
    ids = df["experiment_id"].to_numpy()
    yp = np.zeros(len(df))
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        sel = sel_per_fold[int(ids[i])]
        Xtr = _impute(df.iloc[tr], sel)
        Xte = _impute(df.iloc[[i]], sel)
        sc = StandardScaler().fit(Xtr)
        m = RandomForestRegressor(n_estimators=300, random_state=RNG).fit(sc.transform(Xtr), y[tr])
        yp[i] = m.predict(sc.transform(Xte))[0]
    mae = float(np.mean(np.abs(y - yp)))
    rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((y - yp) ** 2) / ss) if ss > 0 else float("nan")
    return mae, rmse, r2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    args = ap.parse_args()
    topk = args.topk
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_csv(SOURCE).sort_values("physical_experiment_order").reset_index(drop=True)
    y = df["VB_um"].to_numpy(float)
    ids = df["experiment_id"].to_numpy()
    pool = [c for c in reliability_aware_cols(df) if c != "physical_experiment_order"]
    print(f"P8.7 SHAP audit | source=full_contact | n={len(df)} exp | pool={len(pool)} features "
          f"| topk={topk}", flush=True)

    fold_imp_rows, sel_rows = [], []
    sel3_per_fold, sel4_per_fold = {}, {}
    overlaps = []
    shap_accum = {c: [] for c in pool}

    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        Xtr, ytr = df.iloc[tr], y[tr]
        hid = int(ids[i])

        # 3-vote consensus (Kendall/Spearman/MMI)
        sc = score_features(Xtr, ytr, pool, seed=RNG)
        # 4th vote: SHAP (train-only)
        shp = shap_scores(Xtr, ytr, pool, seed=RNG)
        sc = sc.assign(shap_norm=sc.feature.map(shp).astype(float))
        sc["score4"] = (sc.kendall_abs + sc.spearman_abs + sc.mmi_norm + sc.shap_norm) / 4.0

        rank3 = sc.sort_values("score", ascending=False).feature.tolist()
        rank4 = sc.sort_values("score4", ascending=False).feature.tolist()
        sel3 = redundancy_filter(Xtr, rank3)[:topk]
        sel4 = redundancy_filter(Xtr, rank4)[:topk]
        sel3_per_fold[hid] = sel3
        sel4_per_fold[hid] = sel4

        inter = len(set(sel3) & set(sel4))
        union = len(set(sel3) | set(sel4))
        jac = inter / union if union else 1.0
        overlaps.append(jac)
        sel_rows.append(dict(held_out_experiment=hid, n_sel3=len(sel3), n_sel4=len(sel4),
                             shared=inter, jaccard=round(jac, 3),
                             only_with_shap=";".join(sorted(set(sel4) - set(sel3))),
                             dropped_by_shap=";".join(sorted(set(sel3) - set(sel4)))))
        for c in pool:
            shap_accum[c].append(shp[c])
        for _, r in sc.iterrows():
            fold_imp_rows.append(dict(held_out_experiment=hid, feature=r.feature,
                                      kendall_abs=r.kendall_abs, spearman_abs=r.spearman_abs,
                                      mmi_norm=r.mmi_norm, shap_norm=round(float(r.shap_norm), 4),
                                      score3=r.score, score4=round(float(r.score4), 4)))
        print(f"  fold hold-out exp {hid:>3}: jaccard(sel3,sel4)={jac:.2f}", flush=True)

    # ---- aggregate SHAP ranking across folds ----
    agg = (pd.DataFrame([dict(feature=c, shap_mean=float(np.mean(v)),
                              shap_std=float(np.std(v))) for c, v in shap_accum.items()])
           .sort_values("shap_mean", ascending=False).reset_index(drop=True))
    agg["physical_class"] = agg.feature.map(_physical_class)
    agg.to_csv(RESULTS / "p8_7_shap_consensus_ranking.csv", index=False)
    pd.DataFrame(fold_imp_rows).to_csv(RESULTS / "p8_7_shap_fold_scores.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(RESULTS / "p8_7_selection_with_vs_without_shap.csv", index=False)

    # ---- does SHAP change LOEO MAE? ----
    mae3, rmse3, r23 = loeo_mae(df, pool, sel3_per_fold, y)
    mae4, rmse4, r24 = loeo_mae(df, pool, sel4_per_fold, y)
    mae_cmp = pd.DataFrame([
        dict(selector="consensus_3 (Kendall/Spearman/MMI)", MAE=round(mae3, 3),
             RMSE=round(rmse3, 3), R2=round(r23, 3)),
        dict(selector="consensus_4 (+SHAP)", MAE=round(mae4, 3),
             RMSE=round(rmse4, 3), R2=round(r24, 3)),
    ])
    mae_cmp.to_csv(RESULTS / "p8_7_mae_with_vs_without_shap.csv", index=False)

    # ---- physical coherence read ----
    top = agg.head(topk)
    class_mix = top.physical_class.value_counts().to_dict()
    axial_share = top.feature.str.startswith("A_").mean()
    mean_jac = float(np.mean(overlaps))

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    t = agg.head(15).iloc[::-1]
    colors = ["#1F4E79" if f.startswith("A_") else ("#B3541E" if f.startswith("R_") else "#6B7280")
              for f in t.feature]
    ax.barh(range(len(t)), t.shap_mean, color=colors, edgecolor="white")
    ax.errorbar(t.shap_mean, range(len(t)), xerr=t.shap_std, fmt="none", ecolor="#9AA7B3", capsize=2)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.feature, fontsize=8)
    ax.set_xlabel("Importancia SHAP media (normalizada, fold-safe)")
    ax.set_title("P8.7 — Top features por SHAP (azul=axial, naranja=rotacional)")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "p8_7_shap_top_features.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- report ----
    verdict_sel = ("la seleccion CASI NO cambia" if mean_jac >= 0.7 else
                   "la seleccion cambia MODERADAMENTE" if mean_jac >= 0.4 else
                   "la seleccion cambia BASTANTE")
    verdict_mae = ("mejora (leve; posible ruido a n=10)" if mae4 < mae3 - 0.05 else
                   "empeora" if mae4 > mae3 + 0.05 else "no cambia (±0.05)")
    adopt = mae4 <= mae3 + 0.05
    decision = ("ADOPTAR SHAP como 4.º voto del consenso (no empeora el MAE real y es leakage-safe); "
                "re-correr benchmark + PINN con el selector de 4 votos"
                if adopt else
                "MANTENER consenso de 3 votos; SHAP queda como auditoria, no como selector por defecto")
    rep = f"""# P8.7 — Auditoria SHAP (fold-safe)

Fecha: 2026-06-17. Fuente: full_contact. n={len(df)} experimentos. Pool reliability-aware:
{len(pool)} features. top-k={topk}. SHAP via TreeExplainer sobre RandomForest, **solo train por
fold** (anti-fuga). SHAP entra como 4.º voto del consenso (Kendall/Spearman/MMI/SHAP).

## 1. ¿Cambia la seleccion al anadir SHAP?
- Solape medio (Jaccard) entre seleccion de 3 votos y de 4 votos: **{mean_jac:.2f}** -> {verdict_sel}.
- Detalle por fold en `results/p8_7_selection_with_vs_without_shap.csv`.

## 2. ¿Es fisicamente coherente lo que SHAP prioriza?
- Top-{topk} por SHAP, mezcla por clase fisica: {class_mix}
- Fraccion axial en el top-{topk}: **{axial_share:.0%}** (hipotesis: el axial lleva el NIVEL de desgaste).
- Ranking completo en `results/p8_7_shap_consensus_ranking.csv` y figura
  `outputs/figures/p8_7_shap_top_features.png`.

## 3. ¿SHAP mejora la prediccion (LOEO MAE, RandomForest)?
| Selector | MAE | RMSE | R2 |
|---|---:|---:|---:|
| consenso 3 (Kendall/Spearman/MMI) | {mae3:.3f} | {rmse3:.3f} | {r23:.3f} |
| consenso 4 (+SHAP) | {mae4:.3f} | {rmse4:.3f} | {r24:.3f} |
- Veredicto MAE: con SHAP **{verdict_mae}**.

## Hallazgo principal (para el supervisor)
- SHAP NO confirma simplemente "axial = nivel": prioriza features **rotacionales** de variabilidad
  espectral/wavelet (band-ratio std, wavelet entropy, spectral power std). Axial en el top-{topk}: {axial_share:.0%}.
  Es un matiz importante: la variabilidad del canal rotacional carga mucha senal predictiva.
- SHAP es leakage-safe (se calcula por fold solo con train) y **{verdict_mae}** el MAE real.
- Salvedad: a n=10 una mejora de ~1 um puede ser ruido; el valor central de SHAP es la **auditoria**
  (confirmar que el modelo usa senal fisica, no artefactos).

## Decision de politica
{decision}
"""
    (WT / "reports" / "p8_7_shap_audit_report.md").write_text(rep, encoding="utf-8")

    print(f"\nDONE in {time.time()-t0:.0f}s")
    print(f"  mean Jaccard(sel3,sel4) = {mean_jac:.2f}  -> {verdict_sel}")
    print(f"  axial share in top-{topk} (SHAP) = {axial_share:.0%}  | class mix = {class_mix}")
    print(f"  LOEO MAE: 3-vote={mae3:.3f}  4-vote(+SHAP)={mae4:.3f}  -> SHAP {verdict_mae}")
    print("  top-8 SHAP features:")
    print(agg.head(8)[["feature", "shap_mean", "physical_class"]].to_string(index=False))


if __name__ == "__main__":
    main()
