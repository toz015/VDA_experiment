"""Algorithm 5/6: complete VDA pipeline (post-Phase-1 / Method 2 reports).

Given an already-computed theta_hat^raw matrix (K x T) from Method 2, this
runs Phase 1b (initial allocation), Phase 2 (OMD calibration; optional),
Phase 3 (final aggregation + cumulative payments), and Phase 4 (blame).

Phase 1 (LLM evaluation) is handled separately in scripts/generate_reports.py.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .allocation import AllocationResult, solve_allocation
from .blame import BlameResult, attribute_blame
from .gradient import compute_gradient_matrix
from .omd import lr_schedule, omd_update
from .payment import PaymentResult, compute_vcg_payments


@dataclass
class PipelineResult:
    theta_hat_raw: np.ndarray             # (K, T) Phase-1 reports
    theta_hat_final: np.ndarray           # (K, T) after R OMD rounds (== raw if R=0)
    initial_alloc: AllocationResult       # Phase 1b
    final_alloc: AllocationResult         # Phase 3
    theta_bar: np.ndarray                 # (T,) == final_alloc.d
    omd_history: List[np.ndarray] = field(default_factory=list)   # one (K,T) per OMD round
    payment_history: List[PaymentResult] = field(default_factory=list)
    payments_total: Optional[np.ndarray] = None                   # (K,) sum of Pi over OMD rounds
    blame: Optional[BlameResult] = None                           # Phase 4 (if step_agents given)
    omd_converged: bool = False


def run_pipeline(
    theta_hat_raw: np.ndarray,
    *,
    step_agents: Optional[Sequence[str]] = None,
    R: int = 0,
    eta_0: float = 0.3,
    eta_p: float = 0.7,
    omd_tol: float = 1e-5,
    delta: float = 1e-4,
    L: int = 100,
    tau: float = 1e-10,
    eps: float = 1e-6,
    report_eps: Optional[float] = None,
    c_t: float = 0.5,
    omd_direction: str = "ascent",
) -> PipelineResult:
    """Run Phase 1b -> 2 -> 3 -> 4.

    Args:
        theta_hat_raw: (K, T) discriminator reports from Method 2.
        step_agents: optional length-T agent labels; required for Phase 4.
        R: OMD rounds (0 skips Phase 2).
        eta_0, eta_p: lr schedule; eta_r = eta_0 / (r+1)^p.
        omd_tol: early-stop on ||theta_hat^{r+1} - theta_hat^r||_inf.
        delta, L, tau, eps: forwarded to allocation / gradient / payment.
        c_t: blame threshold.
        omd_direction: "ascent" applies the note's literal update (g as-is);
            "descent" applies -g (useful diagnostic — see note discrepancy
            re: outlier gradients).
        report_eps: optional wider clip applied to the raw reports (before
            allocation/OMD). Defaults to eps. Larger values (e.g. 0.05)
            soften LLM saturated outputs and let mirror-descent updates move.
    """
    if omd_direction not in {"ascent", "descent"}:
        raise ValueError(f"omd_direction must be 'ascent' or 'descent'; got {omd_direction!r}")
    if report_eps is None:
        report_eps = eps

    theta = np.clip(np.asarray(theta_hat_raw, dtype=np.float64), report_eps, 1.0 - report_eps)
    K, T = theta.shape

    # Phase 1b — initial Gauss-Seidel allocation on raw reports.
    init_alloc = solve_allocation(theta, L=L, tau=tau, eps=eps)

    # Phase 2 — OMD calibration.
    theta_curr = theta.copy()
    omd_history: List[np.ndarray] = [theta_curr.copy()]
    payment_history: List[PaymentResult] = []
    omd_converged = False
    payments_total = np.zeros(K, dtype=np.float64)

    for r in range(R):
        eta = lr_schedule(r, eta_0=eta_0, p=eta_p)
        alloc_r = solve_allocation(theta_curr, L=L, tau=tau, eps=eps, d_init=init_alloc.d)
        pay_r = compute_vcg_payments(theta_curr, alloc_r, L=L, tau=tau, eps=eps)
        payments_total += pay_r.payments
        payment_history.append(pay_r)
        g = compute_gradient_matrix(theta_curr, delta=delta, L=L, tau=tau, eps=eps)
        if omd_direction == "descent":
            g = -g
        theta_next = omd_update(theta_curr, g, eta=eta, eps=eps)
        omd_history.append(theta_next.copy())
        if np.max(np.abs(theta_next - theta_curr)) < omd_tol:
            theta_curr = theta_next
            omd_converged = True
            break
        theta_curr = theta_next

    # Phase 3 — final allocation on calibrated reports.
    final_alloc = solve_allocation(theta_curr, L=L, tau=tau, eps=eps, d_init=init_alloc.d)

    # Phase 4 — blame attribution.
    blame: Optional[BlameResult] = None
    if step_agents is not None:
        blame = attribute_blame(final_alloc.d, step_agents, c_t=c_t)

    return PipelineResult(
        theta_hat_raw=theta,
        theta_hat_final=theta_curr,
        initial_alloc=init_alloc,
        final_alloc=final_alloc,
        theta_bar=final_alloc.d,
        omd_history=omd_history,
        payment_history=payment_history,
        payments_total=payments_total if R > 0 else None,
        blame=blame,
        omd_converged=omd_converged,
    )
