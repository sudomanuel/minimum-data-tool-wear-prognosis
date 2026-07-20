# -*- coding: utf-8 -*-
"""run_p3_pimaml.py — PAPER 3 · Physics-Informed MAML for cross-condition few-shot tool-wear prognosis.

Everything the manuscript of Paper 3 reports comes out of this script.

WHY THIS EXISTS (and how it differs from the verdict of Paper 1)
---------------------------------------------------------------
Paper 1 adjudicated meta-learning a "non-starter" on this campaign under the task definition
"one tool = one task" (18 tasks). This script tests whether that verdict is a property of the
MECHANISM or of the TASK CONSTRUCTION, by:
  (A) defining a task as a triple (tool i, support window w, query horizon h) -> hundreds of
      real tasks out of the same 18 trajectories;
  (B) adding physics-generated tasks sampled from the fleet-calibrated law VB=b+a*t^p of
      Paper 1 (a distribution over tasks, NOT augmentation of a given tool's data);
  (C) constraining the inner adaptation loop with the differential form of the physical
      admissibility constraints (monotone, decelerating).

PROTOCOLS
  LOOCV : 18 folds, one complete tool held out; every task built from it is removed from
          meta-training (task-level leakage control).
  LOLO  : leave-one-factor-LEVEL-out (vc / feed / cooling). In this campaign
          leave-one-CONDITION-out degenerates to LOOCV (one tool per condition), so LOLO is
          the honest cross-condition test.

PRE-STATED ADOPTION RULE (inherited from Paper 1): adopt only if it beats the deployed record
at the same measurement budget (11.0 um at m=3, 5.6 um at m=4). Reported either way.

Outputs: results/p3_main.csv, p3_adaptation.csv, p3_lolo.csv, p3_ablation.csv
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CENSOR = 300.0
SEED = 0
DEV = "cpu"
RES = os.path.join(ROOT, "results")

# deployed records of Paper 1 (the bar to beat)
REC = {3: 11.02, 4: 5.63}
FLEET_BASE = 18.7


# ------------------------------------------------------------------ data
def load():
    f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
    d = (f[["tool_id", "within_tool_order", "vb_um", "vc", "fz"]].drop_duplicates()
         .rename(columns={"within_tool_order": "order", "vb_um": "vb"})
         .sort_values(["tool_id", "order"]).reset_index(drop=True))
    # cooling is not a column: recover it as the second tool of each (vc,fz) cell
    cond = d.groupby(["vc", "fz"]).tool_id.unique().to_dict()
    cool = {}
    for (vc, fz), tools in cond.items():
        for k, t in enumerate(sorted(tools, key=lambda x: int(str(x).lstrip("T") or 0))):
            cool[t] = k            # 0 / 1 = the two cooling states of that cell
    d["cool"] = d.tool_id.map(cool)
    return d


def tools_of(d):
    return sorted(d.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))


def traj(d, t):
    g = d[(d.tool_id == t) & (d.vb <= CENSOR)].sort_values("order")
    return g.order.to_numpy(float), g.vb.to_numpy(float)


# ------------------------------------------------- physics law of Paper 1
def fit_fleet_p(d, tools):
    """Pooled-SSE fleet exponent on the given tools, on the deployed grid p in [0.20,1.00]."""
    best_p, best = 0.20, np.inf
    for p in np.arange(0.20, 1.001, 0.05):
        tot = 0.0
        for t in tools:
            o, v = traj(d, t)
            if len(o) < 2:
                continue
            A = np.column_stack([np.ones(len(o)), o ** p])
            c, *_ = np.linalg.lstsq(A, v, rcond=None)
            tot += float(np.sum((A @ c - v) ** 2))
        if tot < best:
            best, best_p = tot, p
    return float(best_p)


def fleet_posterior(d, tools, p):
    """Empirical fleet posterior over (b, a) and the residual noise, at exponent p."""
    bs, as_, res = [], [], []
    for t in tools:
        o, v = traj(d, t)
        if len(o) < 2:
            continue
        A = np.column_stack([np.ones(len(o)), o ** p])
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        bs.append(c[0]); as_.append(c[1]); res.append(v - A @ c)
    r = np.concatenate(res) if res else np.array([0.0])
    return (np.array(bs), np.array(as_), float(np.std(r)))


# ------------------------------------------------------------ task builder
def real_tasks(d, tools, m_list=(3, 4), h_max=None):
    """A task = (tool, first m points as support, following h points as query)."""
    T = []
    for t in tools:
        o, v = traj(d, t)
        n = len(o)
        for m in m_list:
            if n <= m:
                continue
            hi = n - m if h_max is None else min(h_max, n - m)
            for h in range(1, hi + 1):
                T.append(dict(kind="real", tool=t, m=m,
                              so=o[:m], sv=v[:m], qo=o[m:m + h], qv=v[m:m + h]))
    return T


def physics_tasks(rng, bs, as_, sd, p, n_tasks, m_list=(3, 4), n_max=12):
    """Tasks sampled from the fleet law family: a DISTRIBUTION over tasks, not augmentation."""
    T = []
    for _ in range(n_tasks):
        b = rng.normal(bs.mean(), max(bs.std(), 1e-6))
        a = rng.normal(as_.mean(), max(as_.std(), 1e-6))
        if a <= 0:
            a = abs(a) + 1e-3
        n = rng.integers(5, n_max + 1)
        o = np.arange(1, n + 1, dtype=float)
        v = b + a * o ** p + rng.normal(0, sd, size=n)
        m = int(rng.choice(m_list))
        if n <= m:
            continue
        h = int(rng.integers(1, n - m + 1))
        T.append(dict(kind="phys", tool=None, m=m,
                      so=o[:m], sv=v[:m], qo=o[m:m + h], qv=v[m:m + h]))
    return T


# ------------------------------------------------------------------ model
class WearNet(nn.Module):
    """Small trajectory net: (tau, support summary) -> VB. Capacity kept deliberately low."""

    def __init__(self, hid=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(4, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(),
                               nn.Linear(hid, 1))

    def forward(self, tau, ctx):
        # tau: (N,1) transformed time ; ctx: (3,) support summary broadcast over the batch
        x = torch.cat([tau, ctx.expand(tau.shape[0], 3)], dim=1)
        return self.f(x).squeeze(-1)


def ctx_of(sv, so, p, scale):
    """Support summary: level, slope in tau, and window length (all scaled)."""
    tau = so ** p
    b0 = sv.mean()
    sl = ((sv[-1] - sv[0]) / max(tau[-1] - tau[0], 1e-6))
    return torch.tensor([b0 / scale, sl / scale, len(so) / 5.0], dtype=torch.float32)


def functional_forward(model, params, tau, ctx):
    """Forward with an explicit parameter list (needed for the differentiable inner loop)."""
    x = torch.cat([tau, ctx.expand(tau.shape[0], 3)], dim=1)
    x = torch.tanh(torch.nn.functional.linear(x, params[0], params[1]))
    x = torch.tanh(torch.nn.functional.linear(x, params[2], params[3]))
    return torch.nn.functional.linear(x, params[4], params[5]).squeeze(-1)


def physics_penalty(params, model, ctx, tau_min, tau_max, scale, n=8):
    """Differential form of the P1 admissibility constraints, on a dense tau grid."""
    tau = torch.linspace(float(tau_min), float(tau_max), n).unsqueeze(1).requires_grad_(True)
    y = functional_forward(model, params, tau, ctx)
    g1 = torch.autograd.grad(y.sum(), tau, create_graph=True)[0]
    g2 = torch.autograd.grad(g1.sum(), tau, create_graph=True)[0]
    mono = torch.relu(-g1).pow(2).mean()        # dVB/dtau >= 0
    conc = torch.relu(g2).pow(2).mean()         # d2VB/dtau2 <= 0
    return mono, conc


def adapt(model, params, task, p, scale, K, alpha, eta1, eta2, create_graph):
    """Inner loop: K constrained gradient steps on the support set."""
    so = torch.tensor(task["so"], dtype=torch.float32)
    sv = torch.tensor(task["sv"], dtype=torch.float32) / scale
    ctx = ctx_of(task["sv"], task["so"], p, scale)
    tau_s = (so ** p).unsqueeze(1)
    tmin, tmax = float((so ** p).min()), float((task["qo"] ** p).max())
    cur = list(params)
    for _ in range(K):
        pred = functional_forward(model, cur, tau_s, ctx)
        loss = ((pred - sv) ** 2).mean()
        if eta1 > 0 or eta2 > 0:
            mono, conc = physics_penalty(cur, model, ctx, tmin, tmax, scale)
            loss = loss + eta1 * mono + eta2 * conc
        grads = torch.autograd.grad(loss, cur, create_graph=create_graph, allow_unused=True)
        cur = [c - alpha * (g if g is not None else torch.zeros_like(c))
               for c, g in zip(cur, grads)]
    return cur, ctx


def query_loss(model, params, task, ctx, p, scale):
    qo = torch.tensor(task["qo"], dtype=torch.float32)
    qv = torch.tensor(task["qv"], dtype=torch.float32) / scale
    pred = functional_forward(model, params, (qo ** p).unsqueeze(1), ctx)
    return ((pred - qv) ** 2).mean(), pred


# ------------------------------------------------------------- meta-train
def meta_train(tasks, p, scale, K=3, alpha=0.05, beta=1e-3, eta1=1.0, eta2=1.0,
               iters=250, batch=12, second_order=False, seed=SEED, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = WearNet().to(DEV)
    params = [q.clone().detach().requires_grad_(True) for q in model.parameters()]
    opt = torch.optim.Adam(params, lr=beta)
    for it in range(iters):
        opt.zero_grad()
        idx = rng.choice(len(tasks), size=min(batch, len(tasks)), replace=False)
        total = 0.0
        for j in idx:
            t = tasks[int(j)]
            fast, ctx = adapt(model, params, t, p, scale, K, alpha, eta1, eta2,
                              create_graph=second_order)
            ql, _ = query_loss(model, fast, t, ctx, p, scale)
            total = total + ql
        (total / len(idx)).backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if verbose and it % 100 == 0:
            print(f"    iter {it:4d} meta-loss {float(total.detach())/len(idx):.4f}")
    return model, [q.detach() for q in params]


def evaluate(model, params, d, test_tools, p, scale, m, K, alpha, eta1, eta2):
    """Adapt on the first m points of each held-out tool, score on the sealed remainder."""
    per, preds, trues = [], [], []
    for t in test_tools:
        o, v = traj(d, t)
        if len(o) <= m:
            continue
        task = dict(so=o[:m], sv=v[:m], qo=o[m:], qv=v[m:])
        pr = [q.clone().detach().requires_grad_(True) for q in params]
        fast, ctx = adapt(model, pr, task, p, scale, K, alpha, eta1, eta2, create_graph=False)
        with torch.no_grad():
            qo = torch.tensor(task["qo"], dtype=torch.float32)
            yh = functional_forward(model, fast, (qo ** p).unsqueeze(1), ctx).numpy() * scale
        per.append(float(np.mean(np.abs(yh - task["qv"]))))
        preds.extend(yh); trues.extend(task["qv"])
    preds, trues = np.array(preds), np.array(trues)
    r2 = 1 - np.sum((trues - preds) ** 2) / np.sum((trues - trues.mean()) ** 2) if len(trues) > 2 else np.nan
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2))) if len(trues) else np.nan
    return float(np.mean(per)) if per else np.nan, float(r2), rmse, per


# ------------------------------------------------------------------- runs
def run_protocol(d, folds, name, m_list=(3, 4), K=3, lam=0.5, eta=(1.0, 1.0), n_phys=400,
                 iters=250, second_order=False, tag="", K_eval=None):
    """folds: list of (fold_name, test_tools). Meta-train per fold on the remaining tools."""
    rows = []
    for m in m_list:
        fold_mae, fold_r2, per_tool = [], [], []
        per_by_K = {}
        for fname, test_tools in folds:
            train_tools = [t for t in tools_of(d) if t not in test_tools]
            p = fit_fleet_p(d, train_tools)
            bs, as_, sd = fleet_posterior(d, train_tools, p)
            scale = float(np.mean([traj(d, t)[1].mean() for t in train_tools]))
            rng = np.random.default_rng(SEED)
            T = []
            if lam > 0:
                T += real_tasks(d, train_tools, m_list=(m,))
            if lam < 1:
                T += physics_tasks(rng, bs, as_, sd, p, int(n_phys * (1 - lam)), m_list=(m,))
            if not T:
                continue
            model, params = meta_train(T, p, scale, K=K, eta1=eta[0], eta2=eta[1],
                                       iters=iters, second_order=second_order)
            Ks = [K] if K_eval is None else K_eval
            for kk in Ks:
                mae, r2, rmse, per = evaluate(model, params, d, test_tools, p, scale, m,
                                              kk, 0.05, eta[0], eta[1])
                if np.isnan(mae):
                    continue
                if K_eval is None:
                    fold_mae.append(mae); fold_r2.append(r2); per_tool.extend(per)
                else:
                    per_by_K.setdefault(kk, []).append((mae, r2, per))
        if K_eval is not None:
            for kk, lst in sorted(per_by_K.items()):
                allper = [x for _, _, pp in lst for x in pp]
                rows.append(dict(protocol=name, tag=f"K={kk}", m=m, K=kk, lam=lam,
                                 eta1=eta[0], eta2=eta[1],
                                 MAE=round(float(np.mean(allper)), 2),
                                 MAE_median=round(float(np.median(allper)), 2),
                                 R2=round(float(np.nanmean([r for _, r, _ in lst])), 3),
                                 n_tools=len(allper), record=REC.get(m, np.nan),
                                 beats_record=bool(np.mean(allper) < REC.get(m, np.inf))))
                print(f"  [{name} · K={kk}] m={m} -> MAE {np.mean(allper):6.2f} um (record {REC.get(m)})", flush=True)
            continue
        if not per_tool:
            continue
        rows.append(dict(protocol=name, tag=tag, m=m, K=K, lam=lam, eta1=eta[0], eta2=eta[1],
                         MAE=round(float(np.mean(per_tool)), 2),
                         MAE_median=round(float(np.median(per_tool)), 2),
                         R2=round(float(np.nanmean(fold_r2)), 3),
                         n_tools=len(per_tool), record=REC.get(m, np.nan),
                         beats_record=bool(np.mean(per_tool) < REC.get(m, np.inf))))
        print(f"  [{name}{(' · '+tag) if tag else ''}] m={m} K={K} lam={lam} eta={eta} "
              f"-> MAE {np.mean(per_tool):6.2f} um (record {REC.get(m)}) "
              f"{'BEATS' if np.mean(per_tool) < REC.get(m, np.inf) else 'below record'}", flush=True)
    return rows


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    d = load()
    tools = tools_of(d)
    print(f"dataset: {len(tools)} tools, {len(d)} inspections")
    p_all = fit_fleet_p(d, tools)
    print(f"fleet exponent on all tools: p* = {p_all:.2f}")
    n_real = len(real_tasks(d, tools))
    print(f"task construction: tools-as-tasks = {len(tools)}  ->  sub-window tasks = {n_real}")

    # ---------- PROTOCOL A: LOOCV by tool ----------
    print("\n=== A · LOOCV by tool (18 folds) ===")
    folds_loocv = [(t, [t]) for t in tools]
    main_rows = run_protocol(d, folds_loocv, "LOOCV", tag="PI-MAML")

    # ---------- PROTOCOL B: LOLO (leave-one-factor-level-out) ----------
    print("\n=== B · LOLO · leave-one-factor-level-out (cross-condition) ===")
    lolo_rows = []
    for factor, col in [("vc", "vc"), ("feed", "fz"), ("cooling", "cool")]:
        folds = []
        for lev in sorted(d[col].unique()):
            tt = sorted(d[d[col] == lev].tool_id.unique(),
                        key=lambda x: int(str(x).lstrip("T") or 0))
            folds.append((f"{factor}={lev}", tt))
        r = run_protocol(d, folds, "LOLO", tag=factor)
        lolo_rows += r

    # ---------- PROTOCOL C: adaptation curve (K = 0..3) ----------
    print("\n=== C · adaptation curve (error vs. inner steps) ===")
    adapt_rows = run_protocol(d, folds_loocv, "LOOCV", K=3, K_eval=[0, 1, 2, 3],
                              tag="adaptation")
    for row in adapt_rows:
        row["study"] = "adaptation" 

    # ---------- PROTOCOL D: ablations ----------
    print("\n=== D · ablations (task mixture and physics constraints) ===")
    abl_rows = []
    for tag, lam, eta in [("lambda=1 (real tasks only)", 1.0, (1.0, 1.0)),
                          ("lambda=0 (physics tasks only)", 0.0, (1.0, 1.0)),
                          ("eta=0 (standard MAML, no physics)", 0.5, (0.0, 0.0))]:
        r = run_protocol(d, folds_loocv, "LOOCV", m_list=(4,), lam=lam, eta=eta, tag=tag)
        for row in r:
            row["study"] = "ablation"
        abl_rows += r

    pd.DataFrame(main_rows).to_csv(os.path.join(RES, "p3_main.csv"), index=False)
    pd.DataFrame(lolo_rows).to_csv(os.path.join(RES, "p3_lolo.csv"), index=False)
    pd.DataFrame(adapt_rows).to_csv(os.path.join(RES, "p3_adaptation.csv"), index=False)
    pd.DataFrame(abl_rows).to_csv(os.path.join(RES, "p3_ablation.csv"), index=False)
    meta = dict(n_tools=len(tools), n_inspections=int(len(d)), p_fleet=p_all,
                tasks_tools_as_tasks=len(tools), tasks_subwindow=n_real,
                record=REC, minutes=round((time.time() - t0) / 60, 1))
    json.dump(meta, open(os.path.join(RES, "p3_meta.json"), "w"), indent=1)
    print(f"\ndone in {meta['minutes']} min -> results/p3_*.csv")


if __name__ == "__main__" and os.environ.get("P3_ARM") != "phi":
    main()


# =====================================================================================
# PI-MAML-phi : THE PHYSICS IS THE ARCHITECTURE (not a penalty on a free network)
# -------------------------------------------------------------------------------------
# The free-network variant above fails because a flexible learner cannot extrapolate from
# three points -- the same failure mode Paper 1 documented for capacity in this regime.
# Here the base learner IS the wear law of Paper 1: VB(t) = b + a*t^p with a>0 (softplus)
# and 0<p<1 (sigmoid), so every adapted forecast is admissible by construction. What is
# meta-learned is the INITIALISATION (b0, a0, p) and the PER-PARAMETER adaptation rates
# (Meta-SGD). The inner loop adapts (b, a) on the support set.
#
# CRITICAL CONTROL: `closed_form` fits the same law to the same m points directly
# (least squares at the fleet exponent) with NO meta-learning. If PI-MAML-phi merely
# converges to it, meta-learning contributes nothing and we must say so.
# =====================================================================================
def _law(b, la, lp, t):
    return b + torch.nn.functional.softplus(la) * t.pow(torch.sigmoid(lp))


def _inner_phi(th, so, sv, K, create_graph=True):
    b, la, lp = th[0], th[1], th[2]
    ab, aa = torch.exp(th[3]), torch.exp(th[4])
    for _ in range(K):
        loss = ((_law(b, la, lp, so) - sv) ** 2).mean()
        gb, ga = torch.autograd.grad(loss, [b, la], create_graph=create_graph)
        b = b - ab * gb
        la = la - aa * ga
    return b, la, lp


def meta_train_phi(tasks, K, iters=250, seed=SEED, lr=0.05):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    th = [torch.tensor(x, requires_grad=True, dtype=torch.float32) for x in
          (100.0, np.log(20.0), 0.0, np.log(0.05), np.log(0.05))]
    opt = torch.optim.Adam(th, lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        tot = 0.0
        idx = rng.choice(len(tasks), size=min(16, len(tasks)), replace=False)
        for j in idx:
            t = tasks[int(j)]
            so = torch.tensor(t["so"], dtype=torch.float32)
            sv = torch.tensor(t["sv"], dtype=torch.float32)
            qo = torch.tensor(t["qo"], dtype=torch.float32)
            qv = torch.tensor(t["qv"], dtype=torch.float32)
            b, la, lp = _inner_phi(th, so, sv, K)
            tot = tot + ((_law(b, la, lp, qo) - qv) ** 2).mean()
        (tot / len(idx)).backward()
        torch.nn.utils.clip_grad_norm_(th, 10.0)
        opt.step()
    return [x.detach() for x in th]


def closed_form(so, sv, p):
    """CONTROL: least-squares fit of the same law to the same support, no meta-learning."""
    A = np.column_stack([np.ones(len(so)), so ** p])
    c, *_ = np.linalg.lstsq(A, sv, rcond=None)
    return lambda t: c[0] + c[1] * t ** p


def eval_phi(d, th, test_tools, m, K, p_fleet, use_closed_form=False):
    per = []
    for t in test_tools:
        o, v = traj(d, t)
        if len(o) <= m:
            continue
        if use_closed_form:
            f = closed_form(o[:m], v[:m], p_fleet)
            yh = f(o[m:])
        else:
            thc = [x.clone().requires_grad_(True) for x in th]
            b, la, lp = _inner_phi(thc, torch.tensor(o[:m], dtype=torch.float32),
                                   torch.tensor(v[:m], dtype=torch.float32), K,
                                   create_graph=False)
            with torch.no_grad():
                yh = _law(b, la, lp, torch.tensor(o[m:], dtype=torch.float32)).numpy()
        per.append(float(np.mean(np.abs(yh - v[m:]))))
    return per


def run_phi(d, folds, name, m_list=(3, 4), K_list=(3, 10, 30), lam=0.5, n_phys=400,
            iters=250, tag="PI-MAML-phi"):
    rows = []
    for m in m_list:
        acc = {k: [] for k in K_list}
        acc_cf = []
        for fname, test_tools in folds:
            train_tools = [t for t in tools_of(d) if t not in test_tools]
            p = fit_fleet_p(d, train_tools)
            bs, as_, sd = fleet_posterior(d, train_tools, p)
            rng = np.random.default_rng(SEED)
            T = []
            if lam > 0:
                T += real_tasks(d, train_tools, m_list=(m,))
            if lam < 1:
                T += physics_tasks(rng, bs, as_, sd, p, int(n_phys * (1 - lam)), m_list=(m,))
            if not T:
                continue
            acc_cf += eval_phi(d, None, test_tools, m, 0, p, use_closed_form=True)
            for K in K_list:
                th = meta_train_phi(T, K, iters=iters)
                acc[K] += eval_phi(d, th, test_tools, m, K, p)
        for K in K_list:
            if not acc[K]:
                continue
            rows.append(dict(protocol=name, tag=tag, m=m, K=K,
                             MAE=round(float(np.mean(acc[K])), 2),
                             MAE_median=round(float(np.median(acc[K])), 2),
                             n_tools=len(acc[K]), record=REC.get(m),
                             beats_record=bool(np.mean(acc[K]) < REC.get(m, np.inf))))
            print(f"  [{name} · {tag}] m={m} K={K} -> MAE {np.mean(acc[K]):6.2f} um "
                  f"(record {REC.get(m)}) "
                  f"{'BEATS' if np.mean(acc[K]) < REC.get(m, np.inf) else 'below record'}",
                  flush=True)
        if acc_cf:
            rows.append(dict(protocol=name, tag="closed-form control (no meta-learning)",
                             m=m, K=0, MAE=round(float(np.mean(acc_cf)), 2),
                             MAE_median=round(float(np.median(acc_cf)), 2),
                             n_tools=len(acc_cf), record=REC.get(m),
                             beats_record=bool(np.mean(acc_cf) < REC.get(m, np.inf))))
            print(f"  [{name} · CONTROL closed-form] m={m} -> MAE {np.mean(acc_cf):6.2f} um",
                  flush=True)
    return rows


def main_phi():
    """Physics-structured arm + the closed-form control, on both protocols."""
    d = load()
    tools = tools_of(d)
    print("=== PHI · physics-structured PI-MAML + closed-form control ===", flush=True)
    rows = run_phi(d, [(t, [t]) for t in tools], "LOOCV")
    print("=== PHI · LOLO cross-condition ===", flush=True)
    for factor, col in [("vc", "vc"), ("feed", "fz"), ("cooling", "cool")]:
        folds = [(f"{factor}={lev}",
                  sorted(d[d[col] == lev].tool_id.unique(),
                         key=lambda x: int(str(x).lstrip("T") or 0)))
                 for lev in sorted(d[col].unique())]
        r = run_phi(d, folds, "LOLO", K_list=(10,), tag=f"PI-MAML-phi · {factor}")
        rows += r
    pd.DataFrame(rows).to_csv(os.path.join(RES, "p3_phi.csv"), index=False)
    print("wrote results/p3_phi.csv", flush=True)


if __name__ == "__main__" and os.environ.get("P3_ARM") == "phi":
    main_phi()
