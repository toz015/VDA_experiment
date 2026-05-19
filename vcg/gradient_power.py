"""Power-kernel kernel-aware finite-difference gradient of VCG payments.

Computes  g[k, t] = d Pi_k^{(p, c)} / d theta_hat^k_t  by central finite
differences. Used by the OMD calibration loop in pipeline_power.py.

Cost per OMD round: O(K * T * K) = O(K^2 * T) Gauss-Seidel allocation solves
(each gradient entry requires two full payment computations, each is K
leave-one-out solves).
"""

import numpy as np

from .allocation_power import solve_allocation_power
from .payment_power import compute_power_payments


def compute_power_gradient_matrix(
    theta: np.ndarray,
    *,
    c: float = 0.05,
    p: float = 1.0,
    delta: float = 1e-4,
    L: int = 100,
    tau: float = 1e-10,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return (K, T) gradient: g[k, t] = d Pi_k^{(p, c)} / d theta_hat^k_t.

    Central differences. For each (k, t), perturb theta[k, t] by +/- delta,
    re-solve the full power-kernel allocation and payments, take
    (Pi_k(theta + delta) - Pi_k(theta - delta)) / (2 delta).
    """
    theta = np.clip(np.asarray(theta, dtype=np.float64), eps, 1.0 - eps)
    K, T = theta.shape
    g = np.zeros((K, T), dtype=np.float64)

    for k in range(K):
        for t in range(T):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[k, t] = min(theta_p[k, t] + delta, 1.0 - eps)
            theta_m[k, t] = max(theta_m[k, t] - delta, eps)

            alloc_p = solve_allocation_power(theta_p, c=c, p=p, L=L, tau=tau, eps=eps)
            alloc_m = solve_allocation_power(theta_m, c=c, p=p, L=L, tau=tau, eps=eps)
            pay_p = compute_power_payments(theta_p, alloc_p, c=c, p=p, L=L, tau=tau, eps=eps)
            pay_m = compute_power_payments(theta_m, alloc_m, c=c, p=p, L=L, tau=tau, eps=eps)

            denom = (theta_p[k, t] - theta_m[k, t])  # robust to boundary clip
            if denom > 0:
                g[k, t] = (pay_p.payments[k] - pay_m.payments[k]) / denom

    return g