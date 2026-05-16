"""VCG mechanism for failure attribution (implementation-note-2026-04-07.pdf §5).

Phases:
  1b. solve_allocation        — Gauss-Seidel multiplicative consensus
  2.  compute_vcg_payments    — leave-one-out informational contributions
  3.  compute_gradient        — finite-difference dPi_k/dtheta_hat
  4.  omd_update              — exponentiated-gradient calibration step
  5.  run_pipeline            — full Phase 1b -> 2 -> 3 -> 4
"""

from .allocation import AllocationResult, compute_weights, social_welfare, solve_allocation
from .blame import BlameResult, attribute_blame
from .gradient import compute_gradient_at, compute_gradient_matrix
from .omd import lr_schedule, omd_update
from .payment import PaymentResult, compute_vcg_payments
from .pipeline import PipelineResult, run_pipeline

__all__ = [
    "AllocationResult",
    "BlameResult",
    "PaymentResult",
    "PipelineResult",
    "attribute_blame",
    "compute_gradient_at",
    "compute_gradient_matrix",
    "compute_vcg_payments",
    "compute_weights",
    "lr_schedule",
    "omd_update",
    "run_pipeline",
    "social_welfare",
    "solve_allocation",
]
