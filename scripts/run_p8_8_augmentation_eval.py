#!/usr/bin/env python3
"""
run_p8_8_augmentation_eval.py — P8.8 fold-safe augmentation evaluation.

Evaluates the R0..R3 augmentation ladder under LOEO on the official VB target,
RandomForest on the reliability-aware pool with per-fold 3-vote selection. For each
regime and several seeds it measures the REAL held-out MAE/RMSE/R2; augmentation is
ADOPTED only if the best regime beats R0 by more than the seed-to-seed noise AND
physics is preserved.

Anti-leakage: selection, scaling and augmentation all use train rows of the fold only;
the held-out experiment is never augmented and never seen during fitting.

Uso:  python run.py p8-8-augmentation  [--topk 10] [--n-aug 30] [--sigma 0.05] [--seeds 6]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "src"))
from phm.feature_selection_p8 import reliability_aware_cols, select_topk  # noqa: E402
from phm.augmentation_p8 import augment, physics_ok, REGIMES, REGIME_LABEL  # noqa: E402

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

RESULTS = WT / "results"
FEATURES = WT / "data" / "features"
SOURCE = FEATURES / "p8_2_features_experiment_full_contact.csv"


def _impute(Xdf, cols, med=None):
    X = Xdf[cols].to_numpy(float)
    if med is None:
        med = np.nanmedian(X, axis=0)
    return np.where(np.isnan(X), med, X), med


def loeo_regime(df, pool, y, order, regime, topk, n_aug, sigma, seed):
    """One LOEO pass for a regime+seed; returns (MAE, RMSE, R2, physics_ok_all)."""
    ids = df["experiment_id"].to_numpy()
    yp = np.zeros(len(df))
    phys_all = True
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        Xtr_df, ytr = df.iloc[tr], y[tr]
        otr = order[tr]
        # selection (train-only, 3-vote)
        if len(pool) > topk:
            sel, _ = select_topk(Xtr_df, ytr, pool, k=topk, seed=0)
        else:
            sel = list(pool)
        Xtr, med = _impute(Xtr_df, sel)
        Xte, _ = _impute(df.iloc[[i]], sel, med=med)
        sc = StandardScaler().fit(Xtr)                 # scaler on REAL train only
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        # augment in scaled space, train-only
        rng = np.random.default_rng(seed * 100 + i)
        Xa, ya = augment(Xtr_s, ytr, otr, regime, n_aug=n_aug, sigma=sigma, rng=rng)
        phys_all = phys_all and physics_ok(ytr, ya)
        m = RandomForestRegressor(n_estimators=300, random_state=seed).fit(Xa, ya)
        yp[i] = m.predict(Xte_s)[0]
    mae = float(np.mean(np.abs(y - yp)))
    rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((y - yp) ** 2) / ss) if ss > 0 else float("nan")
    return mae, rmse, r2, phys_all


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n-aug", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_csv(SOURCE).sort_values("physical_experiment_order").reset_index(drop=True)
    y = df["VB_um"].to_numpy(float)
    order = df["physical_experiment_order"].to_numpy(float)
    pool = [c for c in reliability_aware_cols(df) if c != "physical_experiment_order"]
    seeds = list(range(args.seeds))
    print(f"P8.8 augmentation | n={len(df)} | pool={len(pool)} | topk={args.topk} | "
          f"n_aug={args.n_aug} | sigma={args.sigma} | seeds={len(seeds)}", flush=True)

    rows = []
    for regime in REGIMES:
        used_seeds = [0] if regime == "R0" else seeds       # R0 deterministic
        maes, rmses, r2s, phys = [], [], [], True
        for s in used_seeds:
            mae, rmse, r2, ok = loeo_regime(df, pool, y, order, regime, args.topk,
                                            args.n_aug, args.sigma, s)
            maes.append(mae); rmses.append(rmse); r2s.append(r2); phys = phys and ok
        rows.append(dict(regime=regime, method=REGIME_LABEL[regime],
                         MAE_mean=round(float(np.mean(maes)), 3),
                         MAE_std=round(float(np.std(maes)), 3),
                         RMSE_mean=round(float(np.mean(rmses)), 3),
                         R2_mean=round(float(np.mean(r2s)), 3),
                         physics_ok=bool(phys), n_seeds=len(used_seeds)))
        print(f"  {regime} ({REGIME_LABEL[regime]:<22}) "
              f"MAE={np.mean(maes):.3f} +/- {np.std(maes):.3f}  physics_ok={phys}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "p8_8_augmentation_eval.csv", index=False)

    r0 = res[res.regime == "R0"].iloc[0]
    aug = res[res.regime != "R0"]
    best = aug.loc[aug.MAE_mean.idxmin()]
    improvement = float(r0.MAE_mean - best.MAE_mean)        # >0 means augmentation helps
    noise = float(best.MAE_std)
    beats_noise = improvement > noise and improvement > 0.05
    adopt = bool(beats_noise and best.physics_ok)
    decision = (f"ADOPTAR {best.regime} ({best.method}): mejora {improvement:.2f} um el held-out real "
                f"(> ruido {noise:.2f}) y preserva la fisica"
                if adopt else
                f"NO adoptar augmentation: la mejor ({best.regime}) cambia el MAE en {improvement:+.2f} um, "
                f"dentro del ruido ({noise:.2f} um) -> no supera el control sin aumentar")

    rep = f"""# P8.8 — Evaluacion de augmentation fold-safe

Fecha: 2026-06-17. Fuente: full_contact. n={len(df)} exp. Pool reliability-aware {len(pool)} features.
Modelo: RandomForest. Seleccion 3-votos por fold (topk={args.topk}). n_aug={args.n_aug},
sigma={args.sigma}, seeds={len(seeds)}. Anti-fuga: seleccion/scaler/augmentation solo con train;
el held-out nunca se aumenta ni se ve al ajustar.

## Resultados (LOEO MAE, media +/- desv. entre semillas)
| Regimen | Metodo | MAE | +/- | R2 | Fisica OK |
|---|---|---:|---:|---:|:--:|
""" + "\n".join(
        f"| {r.regime} | {r.method} | {r.MAE_mean:.3f} | {r.MAE_std:.3f} | {r.R2_mean:.3f} | "
        f"{'si' if r.physics_ok else 'NO'} |" for _, r in res.iterrows()) + f"""

## Veredicto (criterio: adoptar solo si mejora el held-out REAL mas que el ruido y preserva fisica)
- Control sin aumentar (R0): MAE = {r0.MAE_mean:.3f}
- Mejor regimen con aumentacion: {best.regime} ({best.method}) MAE = {best.MAE_mean:.3f}
- Mejora vs R0: {improvement:+.3f} um | ruido entre semillas: {noise:.3f} um
- **Decision: {decision}.**

## Lectura (honesta)
- A n=10 con una sola herramienta, la augmentation simple (jitter/interpolacion) {"ayuda de forma robusta" if adopt else "NO ayuda de forma robusta: la mejora (si la hay) no supera el ruido entre semillas"}.
- La interpolacion es physics-safe por construccion (VB entre vecinos reales); el jitter mantiene el VB.
- Queda como **regimen evaluado**: se re-evaluara cuando haya mas herramientas (mas senal real
  suele volver innecesaria o util la augmentation segun el caso).
"""
    (WT / "reports" / "p8_8_augmentation_report.md").write_text(rep, encoding="utf-8")

    print(f"\nDONE in {time.time()-t0:.0f}s")
    print(res.to_string(index=False))
    print(f"\nDECISION: {decision}")


if __name__ == "__main__":
    main()
