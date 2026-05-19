"""L1-Huber multiplicative VCG pipeline (mirrors vcg/pipeline.py).

The only difference vs vcg/pipeline.py is the kernel:
  kappa_c(x) = 1 - min(|x|, c)        (this file, L1-Huber)
  kappa(x)   = 1 - x^2                (vcg/pipeline.py, squared)

Phases:
  1b. Initial L1-Huber Gauss-Seidel allocation on the (clipped) raw reports.
   2. OMD calibration: R rounds of mirror-descent on the reports
      theta_hat^k_t, using kernel-aware finite-difference gradients of the
      L1-Huber payments. Reuses vcg/omd.py's lr_schedule and omd_update
      (the mirror-descent step itself is kernel-independent).
   3. Final L1-Huber allocation on the calibrated reports.
   4. Blame attribution via vcg/blame.py (kernel-independent).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .allocation_l1huber import L1HuberAllocationResult as AllocationResult
from .allocation_l1huber import solve_allocation_l1huber
from .blame import BlameResult, attribute_blame
from .gradient_l1huber import compute_l1huber_gradient_matrix
from .omd import lr_schedule, omd_update
from .payment_l1huber import PaymentResult, compute_l1huber_payments


@dataclass
class PipelineResult:
    theta_hat_raw: np.ndarray
    theta_hat_final: np.ndarray
    initial_alloc: AllocationResult
    final_alloc: AllocationResult
    theta_bar: np.ndarray
    omd_history: List[np.ndarray] = field(default_factory=list)
    payment_history: List[PaymentResult] = field(default_factory=list)
    payments_total: Optional[np.ndarray] = None
    blame: Optional[BlameResult] = None
    omd_converged: bool = False


def run_pipeline_l1huber(
    theta_hat_raw: np.ndarray,
    *,
    step_agents: Optional[Sequence[str]] = None,
    c: float = 0.05,
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
    """Run L1-Huber Phase 1b -> 2 -> 3 -> 4.

    Args:
        theta_hat_raw: (K, T) discriminator reports from Method 2.
        step_agents: optional length-T agent labels; required for Phase 4.
        c: L1-Huber bandwidth.
        R: OMD rounds (0 skips Phase 2).
        eta_0, eta_p: lr schedule; eta_r = eta_0 / (r+1)^p.
        omd_tol: early-stop on ||theta_hat^{r+1} - theta_hat^r||_inf.
        delta, L, tau, eps: forwarded to allocation / gradient / payment.
        report_eps: external clip on raw reports before allocation/OMD.
            Defaults to eps. To match Tong's reported baseline use 0.10.
        c_t: blame threshold (kernel-independent).
        omd_direction: 'ascent' applies the literal update (g as-is);
            'descent' applies -g (diagnostic).
    """
    if omd_direction not in {"ascent", "descent"}:
        raise ValueError(
            f"omd_direction must be 'ascent' or 'descent'; got {omd_direction!r}"
        )
    if report_eps is None:
        report_eps = eps

    theta = np.clip(
        np.asarray(theta_hat_raw, dtype=np.float64),
        report_eps,
        1.0 - report_eps,
    )
    K, T = theta.shape

    # Phase 1b — initial Gauss-Seidel L1-Huber allocation on raw reports.
    init_alloc = solve_allocation_l1huber(theta, c=c, L=L, tau=tau, eps=eps)

    # Phase 2 — OMD calibration (reuse Tong's omd.py; kernel-independent).
    theta_curr = theta.copy()
    omd_history: List[np.ndarray] = [theta_curr.copy()]
    payment_history: List[PaymentResult] = []
    omd_converged = False
    payments_total = np.zeros(K, dtype=np.float64)

    for r in range(R):
        eta = lr_schedule(r, eta_0=eta_0, p=eta_p)
        alloc_r = solve_allocation_l1huber(
            theta_curr, c=c, L=L, tau=tau, eps=eps, d_init=init_alloc.d
        )
        pay_r = compute_l1huber_payments(
            theta_curr, alloc_r, c=c, L=L, tau=tau, eps=eps
        )
        payments_total += pay_r.payments
        payment_history.append(pay_r)
        g = compute_l1huber_gradient_matrix(
            theta_curr, c=c, delta=delta, L=L, tau=tau, eps=eps
        )
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
    final_alloc = solve_allocation_l1huber(
        theta_curr, c=c, L=L, tau=tau, eps=eps, d_init=init_alloc.d
    )

    # Phase 4 — blame attribution (kernel-independent).
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