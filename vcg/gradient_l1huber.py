"""L1-Huber kernel-aware finite-difference gradient of VCG payments.

Computes  g[k, t] = d Pi_k^{L1H} / d theta_hat^k_t  by central finite
differences. Used by the OMD calibration loop in pipeline_l1huber.py.

Each gradient entry requires two full L1-Huber payment computations (one at
theta + delta * e_{kt}, one at theta - delta * e_{kt}), each of which
re-solves K leave-one-out allocations. Total cost per OMD round:
O(K * T * K) = O(K^2 * T) Gauss-Seidel allocation solves.
"""

import numpy as np

from .allocation_l1huber import solve_allocation_l1huber
from .payment_l1huber import compute_l1huber_payments


def compute_l1huber_gradient_matrix(
    theta: np.ndarray,
    *,
    c: float = 0.05,
    delta: float = 1e-4,
    L: int = 100,
    tau: float = 1e-10,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return (K, T) gradient: g[k, t] = d Pi_k^{L1H} / d theta_hat^k_t.

    Args:
        theta: (K, T) current reports.
        c: L1-Huber bandwidth.
        delta: finite-diff step size.
        L, tau, eps: passed to allocation/payment solvers.

    Implementation: central differences. For each (k, t), perturb
    theta[k, t] by +/- delta, re-solve full L1-Huber allocation and
    re-compute payments, take (Pi_k(theta + delta) - Pi_k(theta - delta)) / (2 delta).
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

            alloc_p = solve_allocation_l1huber(theta_p, c=c, L=L, tau=tau, eps=eps)
            alloc_m = solve_allocation_l1huber(theta_m, c=c, L=L, tau=tau, eps=eps)
            pay_p = compute_l1huber_payments(theta_p, alloc_p, c=c, L=L, tau=tau, eps=eps)
            pay_m = compute_l1huber_payments(theta_m, alloc_m, c=c, L=L, tau=tau, eps=eps)

            denom = (theta_p[k, t] - theta_m[k, t])  # robust to clipping at boundary
            if denom > 0:
                g[k, t] = (pay_p.payments[k] - pay_m.payments[k]) / denom

    return g