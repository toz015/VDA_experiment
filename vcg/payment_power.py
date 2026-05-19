"""Power-kernel VCG payment computation (mirrors vcg/payment.py).

Pi_k^{(p, c)} = S_{-k}^{(p, c)}(d*_{-k}) - S_{-k}^{(p, c)}(d*),
S_{-k}^{(p, c)}(d) = sum_{j != k} prod_t kappa_{c, p}(d_t - theta^j_t),
kappa_{c, p}(x) = 1 - min(|x|^p, c^p).

For each k we re-solve the power-kernel allocation with D_k removed
(leave-one-out, warm-started from the full-set d*) and take the difference
in "others' welfare".
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from .allocation_power import PowerAllocationResult, solve_allocation_power


@dataclass
class PaymentResult:
    payments: np.ndarray
    raw_payments: np.ndarray
    leave_one_out: List[PowerAllocationResult]


def _power_per_disc_products(
    d: np.ndarray, theta: np.ndarray, c: float, p: float
) -> np.ndarray:
    """v_k = prod_t kappa_{c, p}(d_t - theta^k_t), shape (K,)."""
    factors = 1.0 - np.minimum(np.abs(d[None, :] - theta) ** p, c ** p)
    return factors.prod(axis=1)


def compute_power_payments(
    theta: np.ndarray,
    full_alloc: PowerAllocationResult,
    *,
    c: float = 0.05,
    p: float = 1.0,
    L: int = 100,
    tau: float = 1e-10,
    eps: float = 1e-6,
) -> PaymentResult:
    """Compute Pi_k^{(p, c)} for each discriminator k.

    Args:
        theta: (K, T) reports.
        full_alloc: result of solve_allocation_power on the full theta.
        c: kernel truncation.
        p: kernel exponent.
        L, tau, eps: passed to leave-one-out solver.

    Returns PaymentResult with payments (length K), raw payments, and
    LOO solves. Negative values clipped to zero (Pi_k >= 0 in theory; small
    numerical slack allowed).
    """
    theta = np.asarray(theta, dtype=np.float64)
    K, T = theta.shape
    d_full = full_alloc.d
    v_full = _power_per_disc_products(d_full, theta, c, p)

    raw = np.zeros(K, dtype=np.float64)
    loo_results: List[PowerAllocationResult] = []
    for k in range(K):
        mask = np.ones(K, dtype=bool)
        mask[k] = False
        theta_minus_k = theta[mask]

        res_mk = solve_allocation_power(
            theta_minus_k, c=c, p=p, L=L, tau=tau, eps=eps, d_init=d_full
        )
        loo_results.append(res_mk)

        v_minus_at_dmk = _power_per_disc_products(res_mk.d, theta_minus_k, c, p).sum()
        v_minus_at_dfull = float(v_full[mask].sum())
        raw[k] = v_minus_at_dmk - v_minus_at_dfull

    # Theory: Pi_k >= 0. Numerical slack allowed; we use the same
    # tolerance as the L1-Huber payment.
    assert (raw >= -1e-4).all(), (
        f"Power-VCG payment negative beyond tolerance: {raw}"
    )
    payments = np.maximum(raw, 0.0)
    return PaymentResult(payments=payments, raw_payments=raw, leave_one_out=loo_results)