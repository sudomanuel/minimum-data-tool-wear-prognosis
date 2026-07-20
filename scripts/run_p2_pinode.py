# -*- coding: utf-8 -*-
"""run_p2_pinode.py — PAPER 2 · Physics-Informed Neural ODE for continuous tool-wear dynamics.

Implements Sections 3.1–3.5 of the Paper-2 formulation on the Paper-1 campaign (18 tools):

  eq.(1)-(2)  dVB/dt = f_theta(VB, t; c_k),  trajectory by explicit RK4 integration
  eq.(3)      f_theta = a_k * g_phi(VB, t),  softplus output => rate >= 0 => monotone by design
  eq.(4)      z_k(t)  = log( e_k(t) / e_k(break-in) )      [within-tool, dimensionless]
  eq.(5)      gate     [1 + beta*tanh(w_psi(z))],  beta <= 0.3;  beta=0 => pure physics field
  eq.(6)-(9)  L = L_data(Huber) + lam_mono*L_mono + lam_phys*L_phys + lam_gate*L_gate
  eq.(10)     Mondrian conformal bands binned by elapsed time since the last anchor

Protocol inherited from Pusma et al. (2025): leakage-safe LOOCV by tool, budgets m in {2,3,4},
anchors visible / future sealed. Resilience regimes R0 (native), R1 (time jitter), R2 (gappy).
Baselines: deployed Paper-1 power law, physics-free Neural ODE, GRU with time-deltas.

ADOPTION RULE (pre-stated, inherited): adopt only if it beats the Paper-1 record at the same
budget WITH valid interval coverage.  Outputs: results/p2_pinode_*.csv
"""
import os, sys, json, math, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load as load_wear, theil_sen           # Paper-1 data + robust fit
from run_optimal_config_search import fit_p                    # Paper-1 fleet exponent

CENSOR = 300.0
SIGMA_MEAS = 5.0          # microscope noise scale (Paper 1 QC)
BETA_MAX = 0.30           # eq.(5) coupling budget
torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------- data
def load_all():
    """Wear curves + within-tool normalised radial excitation z (eq. 4)."""
    d = load_wear().sort_values(["tool_id", "order"]).reset_index(drop=True)
    f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
    f = f.sort_values(["tool_id", "within_tool_order"])
    e = f[["tool_id", "within_tool_order", "R_energy__mean"]].rename(
        columns={"within_tool_order": "order", "R_energy__mean": "energy"})
    d = d.merge(e, on=["tool_id", "order"], how="left")
    # eq.(4): log-ratio to the tool's OWN break-in energy -> dimensionless, within-tool only
    d["z"] = 0.0
    for t, g in d.groupby("tool_id"):
        e0 = g.sort_values("order").energy.iloc[0]
        d.loc[g.index, "z"] = np.log(np.clip(g.energy.to_numpy(float), 1e-9, None) /
                                     max(float(e0), 1e-9))
    d["z"] = d.z.fillna(0.0).clip(-3, 3)
    return d


def tools_of(d):
    return sorted(d.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))


def tool_arrays(d, t):
    g = d[(d.tool_id == t) & (d.vb <= CENSOR)].sort_values("order")
    return (g.order.to_numpy(float), g.vb.to_numpy(float), g.z.to_numpy(float))


# ------------------------------------------------------------------ resilience regimes
def apply_regime(o, v, z, regime, rng):
    """R0 native · R1 time jitter · R2 gappy (delete interior readings)."""
    if regime == "R0":
        return o, v, z
    if regime == "R1":
        jit = rng.uniform(-0.35, 0.35, size=len(o))
        jit[0] = 0.0                                   # anchor origin fixed
        oo = np.maximum.accumulate(o + jit)            # keep strictly increasing
        return oo, v, z
    if regime == "R2":
        if len(o) <= 4:
            return o, v, z
        interior = np.arange(1, len(o) - 1)
        drop = rng.choice(interior, size=min(2, len(interior) - 1), replace=False)
        keep = np.setdiff1d(np.arange(len(o)), drop)
        return o[keep], v[keep], z[keep]
    raise ValueError(regime)


# ------------------------------------------------------------------------ the ODE model
class WearField(nn.Module):
    """eq.(3)+(5): f = a_k * g_phi(VB,t) * [1 + beta*tanh(w_psi(z))], g_phi >= 0 (softplus)."""

    def __init__(self, hidden=16, gate_hidden=8, beta=0.0):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1), nn.Softplus())
        self.w = nn.Sequential(nn.Linear(1, gate_hidden), nn.Tanh(), nn.Linear(gate_hidden, 1))
        self.beta = beta

    def shape(self, vb, t):
        x = torch.stack([vb / 100.0, t / 10.0], dim=-1)
        return self.g(x).squeeze(-1)

    def forward(self, vb, t, z=None):
        r = self.shape(vb, t)
        if self.beta > 0 and z is not None:
            r = r * (1.0 + self.beta * torch.tanh(self.w(z.reshape(-1, 1)).squeeze(-1)))
        return r


def rk4_batch(field, vb0, a_k, times, zmat, mask, steps_per_seg=2):
    """eq.(2) batched: integrate ALL trajectories of a fold in parallel.

    vb0 (B,) · a_k (B,) · times (B,N) · zmat (B,N) zero-order hold per segment · mask (B,N).
    Returns (B,N) predicted wear. Segment i uses z[:, i] (the excitation in force over it).
    """
    B, N = times.shape
    out = [vb0]
    vb = vb0
    for i in range(N - 1):
        t0 = times[:, i]
        h = (times[:, i + 1] - t0) / steps_per_seg
        z = zmat[:, i]
        for s_ in range(steps_per_seg):
            tc = t0 + s_ * h
            k1 = a_k * field(vb, tc, z)
            k2 = a_k * field(vb + 0.5 * h * k1, tc + 0.5 * h, z)
            k3 = a_k * field(vb + 0.5 * h * k2, tc + 0.5 * h, z)
            k4 = a_k * field(vb + h * k3, tc + h, z)
            vb = vb + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(vb)
    return torch.stack(out, dim=1)


def pack_fold(train_tools, d, p_star):
    """Pad every training trajectory to a common length; precompute Paper-1 priors."""
    recs = []
    for t in train_tools:
        o, v, z = tool_arrays(d, t)
        if len(o) < 3:
            continue
        tau = o ** p_star
        a0, b0 = theil_sen(tau, v)
        a_k = max(float(a0) * p_star * max(o[0], 1.0) ** (p_star - 1), 1e-3)
        gpow = np.clip(a0 * p_star * np.maximum(o, 1e-6) ** (p_star - 1), 0, None)
        recs.append((o, v, z, a_k, gpow))
    if not recs:
        return None
    B = len(recs); N = max(len(r[0]) for r in recs)
    T = np.zeros((B, N)); V = np.zeros((B, N)); Z = np.zeros((B, N))
    G = np.zeros((B, N)); M = np.zeros((B, N)); A = np.zeros(B)
    for i, (o, v, z, a_k, gp) in enumerate(recs):
        n = len(o)
        T[i, :n] = o; V[i, :n] = v; Z[i, :n] = z; G[i, :n] = gp; M[i, :n] = 1.0
        if n < N:                      # pad with a frozen tail (masked out of the loss)
            T[i, n:] = o[-1] + np.arange(1, N - n + 1)
            V[i, n:] = v[-1]; Z[i, n:] = z[-1]; G[i, n:] = gp[-1]
        A[i] = a_k
    return dict(T=torch.as_tensor(T), V=torch.as_tensor(V), Z=torch.as_tensor(Z),
                G=torch.as_tensor(G), M=torch.as_tensor(M), A=torch.as_tensor(A))


def train_field(train_tools, d, p_star, beta, lam_phys, lam_mono, lam_gate,
                epochs=140, lr=1.2e-2, seed=0):
    """Fit the shared field on the training fleet (eqs. 6-9), batched over tools."""
    torch.manual_seed(seed)
    pk = pack_fold(train_tools, d, p_star)
    if pk is None:
        return None
    field = WearField(beta=beta)
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    T, V, Z, G, M, A = pk["T"], pk["V"], pk["Z"], pk["G"], pk["M"], pk["A"]
    w = M[:, 1:]

    for _ in range(epochs):
        opt.zero_grad()
        pred = rk4_batch(field, V[:, 0], A, T, Z, M)
        # eq.(7) Huber at the microscope noise scale, masked
        r = pred[:, 1:] - V[:, 1:]
        hub = torch.where(r.abs() <= SIGMA_MEAS, 0.5 * r ** 2,
                          SIGMA_MEAS * (r.abs() - 0.5 * SIGMA_MEAS))
        loss = (hub * w).sum() / w.sum()
        # eq.(9) physics prior: neural rate ~ differential form of the Paper-1 law
        gs = field.shape(V.reshape(-1), T.reshape(-1)).reshape(V.shape)
        loss = loss + lam_phys * (((A.reshape(-1, 1) * gs - G) ** 2) * M).sum() / M.sum()
        # eq.(8) soft deceleration
        dg = (gs[:, 1:] - gs[:, :-1])
        loss = loss + lam_mono * ((torch.clamp(dg, min=0.0) ** 2) * w).sum() / w.sum()
        if beta > 0:                        # L_gate: coupling must be earned
            zz = torch.linspace(-2, 2, 32)
            loss = loss + lam_gate * torch.mean(field.w(zz.reshape(-1, 1)) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 5.0)
        opt.step()
    return field


def _single(o, v, z):
    return (torch.as_tensor(o).reshape(1, -1), torch.as_tensor(v).reshape(1, -1),
            torch.as_tensor(z).reshape(1, -1), torch.ones(1, len(o)))


def personalise(field, o, v, z, m, p_star):
    """Per-tool rate scale a_k from the m anchors only."""
    tau = o[:m] ** p_star
    a0, _ = theil_sen(tau, v[:m])
    seed = max(float(a0) * p_star * max(o[0], 1.0) ** (p_star - 1), 1e-3)
    T, V, Z, M = _single(o[:m], v[:m], z[:m])
    best, best_a = np.inf, seed
    with torch.no_grad():
        for mult in np.linspace(0.2, 3.2, 25):
            a = torch.as_tensor([seed * mult])
            pred = rk4_batch(field, V[:, 0], a, T, Z, M)
            err = float(torch.mean((pred - V) ** 2))
            if err < best:
                best, best_a = err, seed * mult
    return best_a


def forecast(field, o, v, z, m, a_k):
    T, V, Z, M = _single(o, v, z)
    with torch.no_grad():
        pred = rk4_batch(field, V[:, 0], torch.as_tensor([a_k]), T, Z, M)
    return pred.reshape(-1).numpy()


# ------------------------------------------------------------------------ baselines
def baseline_p1(o, v, m, p_star):
    """Deployed Paper-1 few-shot power law (robust fit in tau)."""
    tau = o[:m] ** p_star
    a, b = theil_sen(tau, v[:m])
    return b + a * (o ** p_star)


def baseline_gru(train_tools, d, m, seed=0):
    """Discrete-time GRU with time-delta input — the irregular-sampling straw man."""
    torch.manual_seed(seed)
    net = nn.GRU(input_size=2, hidden_size=16, batch_first=True)
    head = nn.Linear(16, 1)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=1e-2)
    seqs = []
    for t in train_tools:
        o, v, _ = tool_arrays(d, t)
        if len(o) < 3:
            continue
        dt = np.diff(o, prepend=o[0])
        seqs.append((torch.as_tensor(np.stack([v / 100.0, dt], 1)).unsqueeze(0),
                     torch.as_tensor(v[1:] / 100.0)))
    for _ in range(320):
        opt.zero_grad(); loss = 0.0
        for x, y in seqs:
            h, _ = net(x[:, :-1, :])
            loss = loss + torch.mean((head(h).squeeze() - y) ** 2)
        loss.backward(); opt.step()
    return net, head


def gru_forecast(net, head, o, v, m):
    """Roll the GRU forward from the m anchors (autoregressive)."""
    with torch.no_grad():
        dt = np.diff(o, prepend=o[0])
        x = torch.as_tensor(np.stack([v[:m] / 100.0, dt[:m]], 1)).unsqueeze(0)
        h, hn = net(x)
        cur = head(h[:, -1]).squeeze().item()
        out = list(v[:m]) + [cur * 100.0]
        for i in range(m + 1, len(o)):
            xi = torch.as_tensor([[cur, dt[i]]]).unsqueeze(0)
            hh, hn = net(xi, hn)
            cur = head(hh[:, -1]).squeeze().item()
            out.append(cur * 100.0)
    return np.asarray(out[:len(o)], float)


# ------------------------------------------------------------------ conformal (eq. 10)
def mondrian_bands(residuals, deltas, alpha=0.10, edges=(1.5, 3.5)):
    """Per-Delta-bin quantiles; Delta = elapsed time since the last anchor."""
    residuals = np.asarray(residuals); deltas = np.asarray(deltas)
    q = {}
    for b, (lo, hi) in enumerate([(-np.inf, edges[0]), (edges[0], edges[1]), (edges[1], np.inf)]):
        sel = residuals[(deltas > lo) & (deltas <= hi)]
        if len(sel) == 0:
            sel = residuals
        n = len(sel)
        k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
        q[b] = float(np.sort(sel)[k - 1])
    return q


def bin_of(delta, edges=(1.5, 3.5)):
    return 0 if delta <= edges[0] else (1 if delta <= edges[1] else 2)


# ----------------------------------------------------------------------------- runner
_FIELD_CACHE = {}


def get_field(tool_out, d, tools, p_star, key, beta, lam_phys, lam_mono, lam_gate, seed=0):
    """A field depends only on the FOLD and the hyper-parameters — never on m or the regime."""
    ck = (tool_out, key, seed)
    if ck not in _FIELD_CACHE:
        _FIELD_CACHE[ck] = train_field([x for x in tools if x != tool_out], d, p_star,
                                       beta, lam_phys, lam_mono, lam_gate, seed=seed)
    return _FIELD_CACHE[ck]


def run(regimes=("R0", "R1", "R2"), budgets=(2, 3, 4),
        lam_phys=1.0, lam_mono=0.1, lam_gate=0.05, seed=0):
    d = load_all()
    tools = tools_of(d)
    rows = []
    p_star_by_fold = {t: fit_p(d[d.tool_id != t]) for t in tools}
    variants = [("pinode", BETA_MAX, lam_phys),        # physics prior + vibration gate
                ("pinode_nogate", 0.0, lam_phys),      # physics prior, no gate (ablation)
                ("ode_nophys", 0.0, 0.0)]              # neural ODE without physics (ablation)

    # pre-train every field once per fold (cache), then reuse across regimes and budgets
    print(f"pre-training {len(tools) * len(variants)} fields (cached per fold)...")
    for ti, t in enumerate(tools, 1):
        for key, beta, lp in variants:
            get_field(t, d, tools, p_star_by_fold[t], key, beta, lp, lam_mono, lam_gate, seed)
        print(f"  fold {ti}/{len(tools)} ({t}) ready", flush=True)

    for regime in regimes:
        for m in budgets:
            rng = np.random.default_rng(1000 + 7 * m + (hash(regime) % 97))
            acc = {k: [] for k in ["pinode", "pinode_nogate", "ode_nophys", "p1", "gru"]}
            conf = {"resid": [], "delta": []}
            for t in tools:
                o0, v0, z0 = tool_arrays(d, t)
                if len(o0) <= m:
                    continue
                o, v, z = apply_regime(o0, v0, z0, regime, rng)
                if len(o) <= m:
                    continue
                p_star = p_star_by_fold[t]
                acc["p1"].append(np.mean(np.abs(baseline_p1(o, v, m, p_star)[m:] - v[m:])))
                for key, beta, lp in variants:
                    fld = get_field(t, d, tools, p_star, key, beta, lp, lam_mono, lam_gate, seed)
                    if fld is None:
                        continue
                    a_k = personalise(fld, o, v, z, m, p_star)
                    err = np.abs(forecast(fld, o, v, z, m, a_k)[m:] - v[m:])
                    acc[key].append(float(np.mean(err)))
                    if key == "pinode":
                        conf["resid"].extend(err.tolist())
                        conf["delta"].extend((o[m:] - o[m - 1]).tolist())
                net, head = baseline_gru([x for x in tools if x != t], d, m, seed=seed)
                acc["gru"].append(float(np.mean(np.abs(gru_forecast(net, head, o, v, m)[m:] - v[m:]))))

            q = mondrian_bands(conf["resid"], conf["delta"]) if conf["resid"] else {}
            cov = wid = float("nan")
            if q:
                cov = 100.0 * float(np.mean([r <= q[bin_of(dl)]
                                             for r, dl in zip(conf["resid"], conf["delta"])]))
                wid = 2.0 * float(np.mean([q[bin_of(dl)] for dl in conf["delta"]]))
            for k, vals in acc.items():
                if not vals:
                    continue
                rows.append(dict(regime=regime, m=m, model=k, n_tools=len(vals),
                                 MAE=round(float(np.mean(vals)), 2),
                                 MAE_median=round(float(np.median(vals)), 2),
                                 coverage=round(cov, 1) if k == "pinode" else "",
                                 width=round(wid, 1) if k == "pinode" else ""))
                print(f"  {regime} m={m} {k:15s} MAE {np.mean(vals):7.2f} µm "
                      f"(median {np.median(vals):6.2f}) n={len(vals)}"
                      + (f" | cov {cov:.1f}% width {wid:.1f} µm" if k == "pinode" and q else ""),
                      flush=True)

    out = pd.DataFrame(rows)
    dst = os.path.join(ROOT, "results", "p2_pinode_results.csv")
    out.to_csv(dst, index=False)
    print("wrote", dst)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="R0 only, m=3,4")
    a = ap.parse_args()
    print("PAPER 2 · PI-Neural ODE — leakage-safe LOOCV (18 tools)\n" + "=" * 62)
    run(regimes=("R0",) if a.quick else ("R0", "R1", "R2"),
        budgets=(3, 4) if a.quick else (2, 3, 4))
