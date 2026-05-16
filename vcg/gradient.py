"""Algorithm 3: finite-difference gradient of Pi_k wrt theta_hat^k_t (Section 5.3).

Central differences. Each (k, t) gradient evaluation costs 2 full VCG solves;
a full KxT gradient therefore costs 2KT VCG payment computations -- this is
the dominant cost of OMD calibration. We expose both the single-entry helper
and a vectorised "full gradient matrix" routine.
"""

import numpy as np

from .allocation import solve_allocation
from .payment import compute_vcg_payments


def compute_gradient_at(
    theta: np.ndarray,
    k: int,
    t: int,
    *,
    delta: float = 1e-4,
    L: int = 100,
    tau: float = 1e-10,
    eps: float = 1e-6,
) -> float:
    """Approximate dPi_k / dtheta_hat^k_t via central differences."""
    theta = np.asarray(theta, dtype=np.float64)
    base = float(theta[k, t])
    th_plus = min(base + delta, 1.0 - eps)
    th_minus = max(base - delta, eps)

    theta_plus = theta.copy()
    theta_plus[k, t] = th_plus
    res_plus = solve_allocation(theta_plus, L=L, tau=tau, eps=eps)
    pay_plus = compute_vcg_payments(theta_plus, res_plus, L=L, tau=tau, eps=eps)

    theta_minus = theta.copy()
    theta_minus[k, t] = th_minus
    res_minus = solve_allocation(theta_minus, L=L, tau=tau, eps=eps)
    pay_minus = compute_vcg_payments(theta_minus, res_minus, L=L, tau=tau, eps=eps)

    denom = th_plus - th_minus
    if denom == 0.0:
        return 0.0
    return (pay_plus.payments[k] - pay_minus.payments[k]) / denom


def compute_gradient_matrix(
    theta: np.ndarray,
    *,
    delta: float = 1e-4,
    L: int = 100,
    tau: float = 1e-10,
    eps: float = 1e-6,
) -> np.ndarray:
    """Full (K, T) finite-difference gradient g[k, t] = dPi_k/dtheta_hat^k_t.

    Cost: 2 * K * T calls to compute_vcg_payments.
    """
    theta = np.asarray(theta, dtype=np.float64)
    K, T = theta.shape
    g = np.zeros_like(theta)
    for k in range(K):
        for t in range(T):
            g[k, t] = compute_gradient_at(
                theta, k, t, delta=delta, L=L, tau=tau, eps=eps
            )
    return g
