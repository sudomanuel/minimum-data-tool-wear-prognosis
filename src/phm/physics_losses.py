"""
physics_losses.py — terminos de perdida fisica de la PINN de desgaste (P3).

Cada termino es una funcion pura de tensores torch, separada y testeable.
La PINN (pinn.py) los compone:

    L = data_loss
        + lambda_mono   * monotonicity_loss(df/dt)
        + lambda_smooth * smoothness_loss(d2f/dt2)
        + lambda_rate   * wear_rate_loss(df/dt, E_rot; a, b)
        + lambda_bound  * boundary_loss(f(t0), VB_0)

Justificacion fisica (ver reports/physics_spec.md):
  - monotonia: el desgaste de flanco nunca decrece (dVB/dt >= 0);
  - suavidad: la curva VB(t) es suave en regimen estable (sin saltos);
  - ley de tasa: dVB/dt sigue una funcion positiva y creciente de la energia
    vibracional rotacional, g(E) = a + softplus(b) * E  (a, b aprendibles);
  - boundary: SOLO ancla inicial. En T01 NO se usa ancla final de falla
    porque la herramienta no alcanza el umbral (max ~280 µm < 300 µm).

Las derivadas se obtienen por autodiff respecto del leaf `t` (helpers
compute_dvbdt / compute_d2vbdt2), nunca por diferencias finitas.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as _exc:  # pragma: no cover
    TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _exc


# =============================================================================
# Terminos de perdida
# =============================================================================
def data_loss(y_true, y_pred):
    """MSE entre VB real y VB predicho (ambos en espacio escalado)."""
    return F.mse_loss(y_pred, y_true)


def monotonicity_loss(dvbdt):
    """Penaliza tasa de desgaste negativa:  mean(ReLU(-dVB/dt)^2).

    Cero exactamente cuando dVB/dt >= 0 en todos los puntos evaluados
    (penalizacion one-sided: no empuja la tasa hacia arriba, solo prohibe
    que sea negativa).
    """
    return torch.mean(F.relu(-dvbdt) ** 2)


def smoothness_loss(d2vbdt2):
    """Penaliza curvatura excesiva:  mean((d2VB/dt2)^2).

    Regulariza la forma de la curva (evita oscilaciones que un MLP puede
    fabricar entre puntos de entrenamiento); NO fuerza convexidad (eso
    seria penalizar el signo, aqui se penaliza la magnitud).
    """
    return torch.mean(d2vbdt2 ** 2)


def wear_rate_loss(dvbdt, energy_driver, a, b):
    """Acopla la tasa aprendida a la energia rotacional:
        g(E) = a + softplus(b) * E      (g creciente en E si E>0)
        loss = mean((dVB/dt - g(E))^2)

    `a`, `b` son nn.Parameter aprendibles (la PINN los registra en el
    optimizador cuando lambda_rate > 0). `energy_driver` debe venir
    estandarizado y SOLO del train del fold.
    """
    g = a + F.softplus(b) * energy_driver
    return torch.mean((dvbdt - g) ** 2)


def boundary_loss(f_t0, vb_0):
    """Ancla inicial:  (f(x_0, t_0) - VB_0)^2.

    Re-pondera la observacion mas temprana del TRAIN del fold (la curva
    debe partir del desgaste inicial observado). Es el unico boundary
    permitido en T01; el ancla final de falla queda prohibida (la
    herramienta nunca llega al umbral dentro del dataset).
    """
    return (f_t0 - vb_0) ** 2


# =============================================================================
# Helpers de derivadas (autodiff respecto de t)
# =============================================================================
def compute_dvbdt(model, x, t, create_graph: bool = True):
    """df/dt por autodiff. `t` debe ser 1-D; se clona como leaf.

    Devuelve (f, df_dt, t_leaf). `create_graph=True` permite usar df_dt
    dentro de la loss (segundo backward) o derivar otra vez (d2f/dt2).
    """
    t_leaf = t.clone().requires_grad_(True)
    inp = torch.cat([x, t_leaf.unsqueeze(1)], dim=1)
    f = model(inp)
    df_dt = torch.autograd.grad(f.sum(), t_leaf, create_graph=create_graph)[0]
    return f, df_dt, t_leaf


def compute_d2vbdt2(model, x, t, create_graph: bool = True):
    """d2f/dt2 por doble autodiff. Devuelve (f, df_dt, d2f_dt2)."""
    f, df_dt, t_leaf = compute_dvbdt(model, x, t, create_graph=True)
    d2f_dt2 = torch.autograd.grad(df_dt.sum(), t_leaf,
                                  create_graph=create_graph)[0]
    return f, df_dt, d2f_dt2
