"""L1-Huber VCG payment computation (mirrors vcg/payment.py).

Pi_k^{L1H} = S_{-k}(d*_{-k}) - S_{-k}(d*),
S_{-k}(d) = sum_{j != k} v_j(d, theta^j),
v_j(d, theta^j) = prod_t kappa_c(d_t - theta^j_t),
kappa_c(x) = 1 - min(|x|, c).

For each k we re-solve the L1-Huber allocation with D_k removed
(leave-one-out, warm-started from the full-set d*) and take the difference
in "others' welfare".
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from .allocation_l1huber import L1HuberAllocationResult as AllocationResult
from .allocation_l1huber import solve_allocation_l1huber


@dataclass
class PaymentResult:
    payments: np.ndarray
    raw_payments: np.ndarray
    leave_one_out: List[AllocationResult]


def _l1h_per_disc_products(d: np.ndarray, theta: np.ndarray, c: float) -> np.ndarray:
    """v_k = prod_t kappa_c(d_t - theta^k_t), shape (K,)."""
    factors = 1.0 - np.minimum(np.abs(d[None, :] - theta), c)
    return factors.prod(axis=1)


def compute_l1huber_payments(
    theta: np.ndarray,
    full_alloc: AllocationResult,
    *,
    c: float = 0.05,
    L: int = 100,
    tau: float = 1e-10,
    eps: float = 1e-6,
) -> PaymentResult:
    """Compute Pi_k^{L1H} for each discriminator k.

    Args:
        theta: (K, T) reports.
        full_alloc: result of solve_allocation_l1huber on the full theta.
        c: L1-Huber bandwidth.
        L, tau, eps: passed to leave-one-out solver.

    Returns PaymentResult with payments (length K), raw payments, and LOO solves.
    """
    theta = np.asarray(theta, dtype=np.float64)
    K, T = theta.shape
    d_full = full_alloc.d
    v_full = _l1h_per_disc_products(d_full, theta, c)

    raw = np.zeros(K, dtype=np.float64)
    loo_results: List[AllocationResult] = []
    for k in range(K):
        mask = np.ones(K, dtype=bool)
        mask[k] = False
        theta_minus_k = theta[mask]

        res_mk = solve_allocation_l1huber(
            theta_minus_k, c=c, L=L, tau=tau, eps=eps, d_init=d_full
        )
        loo_results.append(res_mk)

        v_minus_at_dmk   = _l1h_per_disc_products(res_mk.d, theta_minus_k, c).sum()
        v_minus_at_dfull = float(v_full[mask].sum())
        raw[k] = v_minus_at_dmk - v_minus_at_dfull

    # Theory: Pi_k >= 0. Numerical slack: clip negatives to zero. Note that
    # the L1-Huber FOC at a Gauss-Seidel optimum is the weighted median, and
    # removing one discriminator can flip the median which makes raw[k]
    # negative more often than under squared loss. We allow a slightly larger
    # tolerance than the squared-kernel assertion in vcg/payment.py.
    assert (raw >= -1e-4).all(), (
        f"L1-Huber VCG payment negative beyond tolerance: {raw}"
    )
    payments = np.maximum(raw, 0.0)
    return PaymentResult(payments=payments, raw_payments=raw, leave_one_out=loo_results)