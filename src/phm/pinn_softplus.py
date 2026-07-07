"""
pinn_softplus.py — PINN with a POSITIVITY-PRESERVING wear-rate prior (P8.4).

New, self-contained module (does NOT modify the legacy `pinn.py`). Fixes the P3/P8.1 failure
where the rate term `g(E)=a+softplus(b)*E` could demand NEGATIVE wear rates. Here the rate prior
is positive by construction:

    dVB/dt ≈ softplus( beta0 + beta1*t + beta2*t^2 + beta3*E_mean + beta4*RMS_mean )

Works in standardized-VB space with t normalized to [0,1]; derivatives by autodiff.
Reliability-aware: the rate driver is a per-contact MEAN energy (E_mean), never raw energy_total
(which is biased at exp77's missing contacts).

rate_form:
  'none'     -> no rate term (data-only / mono-only)
  'old'      -> legacy r = a + softplus(b)*E_drv   (can go negative — the historical baseline)
  'softplus' -> r = softplus(linear(t, t^2, E_drv, RMS_drv))   (positive by construction)
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH = True
except Exception as _e:  # pragma: no cover
    TORCH = False
    _IMPORT_ERR = _e


def _mlp(n_in, hidden=(16, 16)):
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers)


class SoftplusRatePINN:
    def __init__(self, hidden=(16, 16), epochs=1500, lr=0.01, weight_decay=1e-4,
                 lambda_mono=0.0, lambda_rate=0.0, rate_form="none", random_state=42):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_mono = lambda_mono
        self.lambda_rate = lambda_rate
        self.rate_form = rate_form
        self.random_state = random_state

    # ---- scaling helpers ----
    def _fit_scalers(self, X, t, y, drv):
        self.x_mean_, self.x_std_ = X.mean(0), X.std(0) + 1e-8
        self.t_min_, self.t_rng_ = float(t.min()), float(t.max() - t.min()) + 1e-12
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std()) + 1e-8
        self.d_mean_ = drv.mean(0)
        self.d_std_ = drv.std(0) + 1e-8

    def _sx(self, X):
        return (np.asarray(X, float) - self.x_mean_) / self.x_std_

    def _st(self, t):
        return (np.asarray(t, float).reshape(-1) - self.t_min_) / self.t_rng_

    def _sd(self, drv):
        return (np.asarray(drv, float) - self.d_mean_) / self.d_std_

    def _forward_with_grad(self, net, Xs, ts):
        ts = ts.clone().requires_grad_(True)
        inp = torch.cat([Xs, ts.unsqueeze(1)], dim=1)
        f = net(inp).squeeze(1)
        dydt = torch.autograd.grad(f.sum(), ts, create_graph=True)[0]
        return f, dydt

    def fit(self, X, t, y, drv):
        """X: features; t: order coord; y: VB; drv: [E_drv, RMS_drv] rate drivers."""
        if not TORCH:
            raise ImportError(f"PyTorch required: {_IMPORT_ERR!r}")
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))
        X = np.asarray(X, float); t = np.asarray(t, float).reshape(-1)
        y = np.asarray(y, float).reshape(-1); drv = np.asarray(drv, float)
        if drv.ndim == 1:
            drv = drv.reshape(-1, 1)
        self._fit_scalers(X, t, y, drv)

        Xs = torch.tensor(self._sx(X), dtype=torch.float32)
        ts = torch.tensor(self._st(t), dtype=torch.float32)
        ys = torch.tensor((y - self.y_mean_) / self.y_std_, dtype=torch.float32)
        ds = torch.tensor(self._sd(drv), dtype=torch.float32)        # (N, n_drv)
        E_drv = ds[:, 0]
        RMS_drv = ds[:, 1] if ds.shape[1] > 1 else ds[:, 0]

        self.net_ = _mlp(Xs.shape[1] + 1, self.hidden)
        params = list(self.net_.parameters())
        use_rate = self.lambda_rate > 0 and self.rate_form != "none"
        if self.rate_form == "old":
            self.g_a_ = nn.Parameter(torch.zeros(1)); self.g_b_ = nn.Parameter(torch.zeros(1))
            if use_rate:
                params += [self.g_a_, self.g_b_]
        elif self.rate_form == "softplus":
            self.rate_head_ = nn.Linear(4, 1)               # [t, t^2, E_drv, RMS_drv] -> 1
            if use_rate:
                params += list(self.rate_head_.parameters())

        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        use_mono = self.lambda_mono > 0
        sp = nn.functional.softplus

        self.net_.train()
        for _ in range(int(self.epochs)):
            opt.zero_grad()
            if use_mono or use_rate:
                f, dydt = self._forward_with_grad(self.net_, Xs, ts)
            else:
                f = self.net_(torch.cat([Xs, ts.unsqueeze(1)], dim=1)).squeeze(1)
                dydt = None
            loss = torch.mean((f - ys) ** 2)
            if use_mono:
                loss = loss + self.lambda_mono * torch.mean(torch.relu(-dydt) ** 2)
            if use_rate:
                if self.rate_form == "old":
                    r = self.g_a_ + sp(self.g_b_) * E_drv          # CAN be negative
                else:
                    feats = torch.stack([ts, ts ** 2, E_drv, RMS_drv], dim=1)
                    r = sp(self.rate_head_(feats).squeeze(1))      # POSITIVE by construction
                loss = loss + self.lambda_rate * torch.mean((dydt - r) ** 2)
            loss.backward()
            opt.step()
        return self

    def predict(self, X, t):
        Xs = torch.tensor(self._sx(X), dtype=torch.float32)
        ts = torch.tensor(self._st(t), dtype=torch.float32)
        self.net_.eval()
        with torch.no_grad():
            f = self.net_(torch.cat([Xs, ts.unsqueeze(1)], dim=1)).squeeze(1).numpy()
        return f * self.y_std_ + self.y_mean_

    def wear_rate_physical(self, X, t):
        """dVB/d(order) in physical µm per experiment-step (diagnostic)."""
        Xs = torch.tensor(self._sx(X), dtype=torch.float32)
        ts = torch.tensor(self._st(t), dtype=torch.float32)
        self.net_.eval()
        f, dydt = self._forward_with_grad(self.net_, Xs, ts)
        # d(VB)/d(order) = y_std * d(y_std)/d(t_norm) / t_rng
        return (dydt.detach().numpy() * self.y_std_ / self.t_rng_)
