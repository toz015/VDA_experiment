"""Inference-aligned OMD gradient computation for Power-VCG.

The original Tong's OMD uses payment-ascent: g[k, t] = d Pi_k / d theta^k_t.
This file provides alternative gradient signals that are aligned with the
inference goal (identifying the mistake step) rather than the equilibrium
goal (Nash of the VCG game):

  - 'entropy':  g = - d H(d_hat^*) / d theta^k_t,
                where d_hat^* is the normalised allocation d^* / sum(d^*).
                Descent on entropy = ascent on concentration.

  - 'margin':   g = + d (max_t d_t - second_max_t d_t) / d theta^k_t.
                Ascent on top1-top2 gap.

  - 'payment':  recovers vcg.gradient_power.compute_power_gradient_matrix.

All gradients are computed by central finite differences in this file for
consistency with the existing payment-gradient implementation. A future
analytical implementation would be faster.
"""

import numpy as np

from .allocation_power import solve_allocation_power


def normalised_entropy(d: np.ndarray) -> float:
    """H(d_hat) where d_hat = d / sum(d)."""
    s = d.sum()
    if s <= 1e-12:
        return 0.0
    p = d / s
    p_safe = np.clip(p, 1e-12, 1.0)
    return float(-(p_safe * np.log(p_safe)).sum())


def top1_minus_top2(d: np.ndarray) -> float:
    sorted_desc = -np.sort(-d)
    if sorted_desc.size < 2:
        return float(sorted_desc[0]) if sorted_desc.size > 0 else 0.0
    return float(sorted_desc[0] - sorted_desc[1])


def compute_inference_gradient_matrix(
    theta: np.ndarray,
    *,
    objective: str = "entropy",
    c: float = 0.05,
    p: float = 1.0,
    delta: float = 1e-4,
    L: int = 100,
    tau: float = 1e-10,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return (K, T) gradient of the chosen inference objective.

    For 'entropy', returns g[k, t] = -d H(d_hat^*) / d theta^k_t  (so that
    GRADIENT ASCENT on g moves toward lower entropy, i.e.\\ more concentrated
    allocation, i.e.\\ closer to the one-hot ground-truth allocation).

    For 'margin', returns g[k, t] = +d (top1 - top2)(d^*) / d theta^k_t (so
    that GRADIENT ASCENT moves toward larger margin).

    Both signs are chosen so the standard "ascent on g" OMD update with
    eta > 0 makes the allocation more concentrated.
    """
    if objective not in {"entropy", "margin"}:
        raise ValueError(f"objective must be 'entropy' or 'margin', got {objective}")

    theta = np.clip(np.asarray(theta, dtype=np.float64), eps, 1.0 - eps)
    K, T = theta.shape
    g = np.zeros((K, T), dtype=np.float64)

    def obj_of(th):
        d = solve_allocation_power(th, c=c, p=p, L=L, tau=tau, eps=eps).d
        if objective == "entropy":
            return normalised_entropy(d)
        else:
            return top1_minus_top2(d)

    for k in range(K):
        for t in range(T):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[k, t] = min(theta_p[k, t] + delta, 1.0 - eps)
            theta_m[k, t] = max(theta_m[k, t] - delta, eps)

            obj_p = obj_of(theta_p)
            obj_m = obj_of(theta_m)
            denom = theta_p[k, t] - theta_m[k, t]
            if denom <= 0:
                continue

            raw_grad = (obj_p - obj_m) / denom
            # For entropy: ASCENT on g should DECREASE entropy ->
            #   we want g = - d(entropy)/d(theta).
            # For margin: ASCENT on g should INCREASE margin ->
            #   we want g = + d(margin)/d(theta).
            if objective == "entropy":
                g[k, t] = -raw_grad
            else:
                g[k, t] = +raw_grad

    return g