"""
nn_estimators.py — estimadores neuronales compatibles con scikit-learn.

`BayesianNNRegressor`: red neuronal bayesiana (Bayes-by-Backprop, Blundell
et al. 2015) que reproduce el *concepto* del modelo de Airao et al. (2026,
Wear): pesos probabilisticos con prior gaussiano que actua como
regularizacion, 1 capa oculta de 32 neuronas con activacion tanh, y
cuantificacion de incertidumbre (epistemica via muestreo de pesos +
aleatoria via varianza de observacion aprendida).

----------------------------------------------------------------------------
NOTA DE FIDELIDAD (para la seccion de metodologia del paper):

El paper usa MATLAB `trainbr` = Bayesian Regularization de MacKay +
Levenberg-Marquardt, que estima los hiperparametros de regularizacion
(psi, phi) via el marco de evidencia / Hessiano. Aqui usamos inferencia
variacional (minimiza el ELBO = NLL + KL[q||prior]). Ambos son BNN y
comparten la idea esencial — prior sobre pesos + cuantificacion de
incertidumbre — pero el metodo de inferencia difiere. Se documenta
explicitamente para no presentarlo como una reimplementacion exacta de
`trainbr`.
----------------------------------------------------------------------------

Diseno pensado para n pequeno (LOEO con 9 muestras de train):
  - escala `y` internamente (mean/std) para estabilizar el entrenamiento,
  - full-batch Adam (sin minibatches con n=9),
  - prediccion = media predictiva sobre `n_samples_predict` muestras MC,
  - `kl_weight` (cold-posterior tempering) controla cuanto pesa el prior;
    es la perilla de regularizacion principal con datos escasos.

Este modulo es el BACKBONE del futuro B-PINN (Etapa 3-4): para convertirlo
en PINN basta anadir los terminos de perdida fisica (monotonicidad
dVB/dt >= 0 y ley de tasa de desgaste guiada por energia de vibracion) al
ELBO en `fit`.
"""
from __future__ import annotations

import math
import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as _exc:  # pragma: no cover
    TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _exc


# =============================================================================
# Capa lineal bayesiana (reparameterization trick + KL cerrada)
# =============================================================================
if TORCH_AVAILABLE:

    class _BayesianLinear(nn.Module):
        """Capa densa con pesos ~ N(mu, softplus(rho)^2) y prior N(0, prior_sigma^2)."""

        def __init__(self, in_features: int, out_features: int,
                     prior_sigma: float = 1.0):
            super().__init__()
            self.prior_sigma = float(prior_sigma)
            # Posterior variacional: media + rho (sigma = softplus(rho)).
            self.w_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0.0, 0.1))
            self.w_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))
            self.b_mu = nn.Parameter(torch.zeros(out_features))
            self.b_rho = nn.Parameter(torch.full((out_features,), -5.0))
            self._kl = torch.tensor(0.0)

        def forward(self, x):
            w_sigma = F.softplus(self.w_rho)
            b_sigma = F.softplus(self.b_rho)
            # Muestra de pesos (reparameterization): permite backprop estocastico.
            w = self.w_mu + w_sigma * torch.randn_like(w_sigma)
            b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
            self._kl = self._kl_term(self.w_mu, w_sigma) + self._kl_term(self.b_mu, b_sigma)
            return F.linear(x, w, b)

        def _kl_term(self, mu, sigma):
            ps = self.prior_sigma
            return torch.sum(
                torch.log(ps / sigma) + (sigma ** 2 + mu ** 2) / (2.0 * ps ** 2) - 0.5
            )

    class _BNNNet(nn.Module):
        """1 capa oculta (tanh) bayesiana + cabeza lineal bayesiana.

        Estructura identica a la del paper (1x32, tanh). `log_noise` modela
        la incertidumbre aleatoria (homoscedastica) en el espacio escalado.
        """

        def __init__(self, in_features: int, hidden: int = 32,
                     prior_sigma: float = 1.0):
            super().__init__()
            self.l1 = _BayesianLinear(in_features, hidden, prior_sigma)
            self.l2 = _BayesianLinear(hidden, 1, prior_sigma)
            self.log_noise = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            h = torch.tanh(self.l1(x))
            return self.l2(h).squeeze(-1)

        def kl_sum(self):
            return self.l1._kl + self.l2._kl


# =============================================================================
# Estimador sklearn-compatible
# =============================================================================
class BayesianNNRegressor(BaseEstimator, RegressorMixin):
    """Red neuronal bayesiana (variacional) compatible con scikit-learn.

    Pensada para entrar en el harness por capas (`all_baseline_builders`)
    exactamente igual que cualquier otro modelo: recibe X ya imputado y
    escalado por el `Pipeline` y se evalua bajo LOEO. Reproduce el concepto
    del BNN del paper (ver nota de fidelidad en el docstring del modulo).

    Parameters
    ----------
    hidden : int           neuronas de la capa oculta (paper: 32).
    prior_sigma : float    desviacion del prior gaussiano sobre los pesos.
    kl_weight : float      tempering del termino KL (cold posterior). Con
                           datos escasos, valores pequenos dejan hablar a
                           los datos; grandes acercan al prior.
    lr, epochs : optimizacion (Adam, full-batch).
    n_samples_predict : muestras MC para la media/incertidumbre predictiva.
    random_state : reproducibilidad.
    """

    def __init__(self, hidden: int = 32, prior_sigma: float = 1.0,
                 kl_weight: float = 1e-3, lr: float = 1e-2,
                 epochs: int = 3000, n_samples_predict: int = 64,
                 random_state: int = 42, verbose: bool = False):
        self.hidden = hidden
        self.prior_sigma = prior_sigma
        self.kl_weight = kl_weight
        self.lr = lr
        self.epochs = epochs
        self.n_samples_predict = n_samples_predict
        self.random_state = random_state
        self.verbose = verbose

    # ---- entrenamiento -----------------------------------------------------
    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError(
                f"PyTorch no disponible: {(_TORCH_IMPORT_ERROR)!r}. "
                "Instala torch>=2.5 (compatible con numpy 2)."
            )
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.n_features_in_ = X.shape[1]

        # Escalado de y (X ya viene escalado por el Pipeline del harness).
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor((y - self.y_mean_) / self.y_std_, dtype=torch.float32)

        self.net_ = _BNNNet(self.n_features_in_, self.hidden, self.prior_sigma)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)

        self.net_.train()
        for ep in range(int(self.epochs)):
            opt.zero_grad()
            pred = self.net_(Xt)
            noise = F.softplus(self.net_.log_noise) + 1e-6
            # NLL gaussiano (suma sobre el dataset, full-batch).
            nll = 0.5 * torch.log(2.0 * math.pi * noise ** 2) \
                + 0.5 * ((yt - pred) ** 2) / (noise ** 2)
            nll = nll.sum()
            kl = self.net_.kl_sum()
            loss = nll + float(self.kl_weight) * kl
            loss.backward()
            opt.step()
            if self.verbose and (ep % 500 == 0 or ep == self.epochs - 1):
                print(f"  [BNN] epoch {ep:4d}  loss={loss.item():.3f}  "
                      f"nll={nll.item():.3f}  kl={kl.item():.1f}")
        return self

    # ---- prediccion --------------------------------------------------------
    def _forward_samples(self, X) -> np.ndarray:
        """Devuelve (n_samples_predict, n) en espacio ESCALADO de y."""
        Xt = torch.tensor(np.asarray(X, dtype=float), dtype=torch.float32)
        self.net_.train()  # mantener pesos estocasticos (muestreo activo)
        outs = []
        with torch.no_grad():
            for _ in range(int(self.n_samples_predict)):
                outs.append(self.net_(Xt).cpu().numpy())
        return np.stack(outs, axis=0)

    def predict(self, X) -> np.ndarray:
        """Media predictiva (en micras), compatible con el harness."""
        s = self._forward_samples(X)
        return s.mean(axis=0) * self.y_std_ + self.y_mean_

    def predict_dist(self, X):
        """Devuelve (media, std_epistemica) en micras. Para la Etapa 4 (UQ)."""
        s = self._forward_samples(X) * self.y_std_ + self.y_mean_
        mean = s.mean(axis=0)
        epi_std = s.std(axis=0)
        # Incertidumbre aleatoria (homoscedastica) aprendida, en micras.
        with torch.no_grad():
            ale_std = float(F.softplus(self.net_.log_noise).item()) * self.y_std_
        total_std = np.sqrt(epi_std ** 2 + ale_std ** 2)
        return mean, total_std

    def _more_tags(self):
        return {"non_deterministic": True, "requires_positive_X": False}
