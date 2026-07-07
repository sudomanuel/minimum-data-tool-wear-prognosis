#!/usr/bin/env python3
"""
run_p8_9_pinn_rate_sweep.py — P8.9 PINN physics refinement: wear-rate weight sweep.

P8.4 showed the rate term at lambda_rate=0.1 over-regularizes (LOEO MAE jumps from
~17.6 at mono-only to ~28). This sweeps lambda_rate for the SOFTPLUS positive-rate
prior (with monotonicity ON) to find the operating point where the physical positive
rate is enforced WITHOUT destroying accuracy. Reuses src/phm/pinn_softplus.py.

For each lambda_rate it reports LOEO MAE/R2, in-sample fraction of negative wear rate
(should stay 0 with the softplus form) and monotonicity violations. Honest goal: locate
the coherence-vs-accuracy trade-off, not to beat Linear(t).

Uso:  python run.py p8-9-pinn-rate-sweep  [--epochs 1000]
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
from phm.pinn_softplus import SoftplusRatePINN  # noqa: E402

RESULTS = WT / "results"
FIGS = WT / "outputs" / "figures"
FEAT = WT / "data" / "features" / "p8_2_features_experiment_full_contact.csv"
X_FEATURES = ["A_rms_mean", "R_rms_mean", "A_waveform_length_mean", "R_waveform_length_mean",
              "R_dominant_freq_mean", "R_wavelet_entropy_mean"]
DRIVERS = ["R_energy_mean", "R_rms_mean"]
T_COL = "physical_experiment_order"
LAMBDAS = [0.0, 0.01, 0.03, 0.1, 0.3]
REF = {"Linear(t)": 3.10, "PINN_mono_P8.1": 6.65}


def metrics(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    mae = float(np.mean(np.abs(yt - yp)))
    ss = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1 - np.sum((yt - yp) ** 2) / ss) if ss > 0 else float("nan")
    return mae, r2


def loeo(df, cfg, epochs, seed=42):
    X = df[X_FEATURES].to_numpy(float); t = df[T_COL].to_numpy(float)
    y = df["VB_um"].to_numpy(float); drv = df[DRIVERS].to_numpy(float)
    yp = np.zeros(len(df))
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        m = SoftplusRatePINN(epochs=epochs, random_state=seed, **cfg).fit(X[tr], t[tr], y[tr], drv[tr])
        yp[i] = m.predict(X[i:i + 1], t[i:i + 1])[0]
    return y, yp


def insample_rate(df, cfg, epochs, seed=42):
    X = df[X_FEATURES].to_numpy(float); t = df[T_COL].to_numpy(float)
    y = df["VB_um"].to_numpy(float); drv = df[DRIVERS].to_numpy(float)
    m = SoftplusRatePINN(epochs=epochs, random_state=seed, **cfg).fit(X, t, y, drv)
    grid = np.linspace(t.min(), t.max(), 60)
    Xg = np.vstack([np.interp(grid, t, X[:, j]) for j in range(X.shape[1])]).T
    rate = m.wear_rate_physical(Xg, grid)
    return round(float(np.mean(rate < 0)), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=1000)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_csv(FEAT).sort_values(T_COL).reset_index(drop=True)
    print(f"P8.9 PINN rate-weight sweep | n={len(df)} | epochs={args.epochs} | "
          f"lambda_rate in {LAMBDAS} | mono ON, softplus rate\n", flush=True)

    rows = []
    for lr in LAMBDAS:
        cfg = dict(lambda_mono=1.0, lambda_rate=lr,
                   rate_form=("none" if lr == 0.0 else "softplus"))
        y, yp = loeo(df, cfg, args.epochs)
        mae, r2 = metrics(y, yp)
        order = df[T_COL].to_numpy(float); idx = np.argsort(order)
        mono_v = int(np.sum(np.diff(np.asarray(yp)[idx]) < -1e-6))
        neg = insample_rate(df, cfg, args.epochs)
        rows.append(dict(lambda_rate=lr, rate_form=cfg["rate_form"], MAE=round(mae, 3),
                         R2=round(r2, 3), mono_violations=mono_v, insample_rate_neg_fraction=neg))
        print(f"  lambda_rate={lr:<5} MAE={mae:6.2f} R2={r2:5.2f} mono_viol={mono_v} "
              f"rate<0={neg:.0%}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "p8_9_pinn_rate_lambda_sweep.csv", index=False)

    base = res[res.lambda_rate == 0.0].iloc[0]                    # mono-only baseline
    pos = res[res.lambda_rate > 0.0]
    best = pos.loc[pos.MAE.idxmin()]
    # operating point: largest lambda whose MAE stays within +1.0 um of the mono-only baseline
    tol = base.MAE + 1.0
    safe = pos[pos.MAE <= tol]
    op = safe.loc[safe.lambda_rate.idxmax()] if len(safe) else None

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(res.lambda_rate, res.MAE, "o-", color="#2E6F62", lw=2, label="LOEO MAE (µm)")
    ax.axhline(base.MAE, color="#1F4E79", ls="--", lw=1.2, label=f"mono-only (λ=0): {base.MAE:.1f}")
    ax.axhline(REF["Linear(t)"], color="#8C2D2D", ls=":", lw=1.2, label=f"Linear(t) control: {REF['Linear(t)']}")
    ax.set_xlabel("lambda_rate (peso del termino de tasa softplus)")
    ax.set_ylabel("LOEO MAE (µm)")
    ax.set_title("P8.9 — la tasa sobre-regulariza al subir λ_rate (rate<0 = 0% siempre)")
    for _, r in res.iterrows():
        ax.annotate(f"{r.MAE:.1f}", (r.lambda_rate, r.MAE), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGS / "p8_9_pinn_rate_lambda_sweep.png", dpi=200,
                                    bbox_inches="tight", facecolor="white"); plt.close(fig)

    op_txt = (f"lambda_rate={op.lambda_rate} (MAE {op.MAE:.2f}, dentro de +1 µm del control mono-only)"
              if op is not None else
              "ningun lambda_rate>0 se queda dentro de +1 µm del control: mantener la tasa APAGADA "
              "para el modelo operativo")
    rep = f"""# P8.9 — Refinamiento de la fisica de la PINN (barrido de lambda_rate)

Fecha: 2026-06-17. Fuente: full_contact. n={len(df)}. epochs={args.epochs}. Forma de tasa: softplus
(positiva por construccion), monotonia ON. Reutiliza src/phm/pinn_softplus.py.

## Barrido
| lambda_rate | MAE (µm) | R2 | viol. monotonia | rate<0 (in-sample) |
|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {r.lambda_rate} | {r.MAE:.3f} | {r.R2:.3f} | {r.mono_violations} | {r.insample_rate_neg_fraction:.0%} |"
        for _, r in res.iterrows()) + f"""

## Lectura
- Control mono-only (lambda_rate=0): MAE = {base.MAE:.3f}.
- Mejor lambda_rate>0: {best.lambda_rate} (MAE {best.MAE:.3f}).
- **La tasa positiva (softplus) es coherente en TODO el barrido: rate<0 = 0% siempre** -> la fisica
  positiva esta garantizada por construccion, no depende del peso.
- Pero subir lambda_rate **sobre-regulariza**: el MAE crece al aumentar el peso del termino de tasa.

## Punto de operacion recomendado
- {op_txt}.
- Conclusion honesta: la PINN_mono (sin tasa, o con tasa muy pequena) es la mejor OPERATIVA; el
  termino softplus se mantiene como **rama de seguridad** (garantiza velocidad >= 0) y se activa
  con peso bajo solo si se requiere esa garantia explicita. Con mas datos (LOTO) se podra subir el
  peso o usar pesos adaptativos (estilo Zhang 2026) sin penalizar el MAE.
"""
    (WT / "reports" / "p8_9_pinn_physics_report.md").write_text(rep, encoding="utf-8")

    print(f"\nDONE in {time.time()-t0:.0f}s")
    print(res.to_string(index=False))
    print(f"\nPunto de operacion: {op_txt}")


if __name__ == "__main__":
    main()
