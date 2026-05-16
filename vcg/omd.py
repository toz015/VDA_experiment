"""Algorithm 4: OMD calibration step (Section 5.4).

Coordinate-wise exponentiated-gradient update with Bernoulli entropy:
    theta_hat^+ = theta * exp(eta g) / (theta * exp(eta g) + (1 - theta) * exp(-eta g)).
Equivalent to logistic-mirror ascent on Pi_k under the Bernoulli simplex.
"""

import numpy as np


def omd_update(
    theta: np.ndarray,
    g: np.ndarray,
    eta: float,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply the exponentiated-gradient update element-wise.

    Args:
        theta: array of current reports in (eps, 1-eps).
        g: array of same shape with gradients dPi_k/dtheta_hat^k_t.
        eta: scalar learning rate at this round.
        eps: clipping range to keep updated reports in the open interval.

    Returns updated theta of the same shape, clipped to [eps, 1-eps].
    """
    theta = np.asarray(theta, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    if theta.shape != g.shape:
        raise ValueError(f"shape mismatch: theta {theta.shape} vs g {g.shape}")

    e_pos = theta * np.exp(eta * g)
    e_neg = (1.0 - theta) * np.exp(-eta * g)
    denom = e_pos + e_neg
    # denom > 0 by construction.
    updated = e_pos / denom
    return np.clip(updated, eps, 1.0 - eps)


def lr_schedule(round_idx: int, eta_0: float = 0.3, p: float = 0.7) -> float:
    """Polynomial decay schedule eta_r = eta_0 / (r+1)^p (Section 7 default)."""
    return eta_0 / ((round_idx + 1) ** p)
