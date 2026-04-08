"""Hyperparameters and defaults for VDA (matches Section 7 of the implementation note)."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class VDAConfig:
    # Discriminators (Stage 1: single model, K temperatures)
    K: int = 3
    openai_model: str = "gpt-4o-mini"
    temperatures: List[float] = field(default_factory=lambda: [0.0, 0.7, 1.0])

    # OMD calibration (Stage 2)
    R: int = 15
    eta_0: float = 0.3
    eta_p: float = 0.7
    omd_tol: float = 1e-5

    # Gauss–Seidel solver
    L: int = 100
    tau: float = 1e-10

    # Gradient
    delta: float = 1e-4
    eps: float = 1e-6

    # Blame
    c_t: float = 0.5

    # Prompt
    prompt_max_tokens_per_prior_step: int = 500

    def lr(self, r: int) -> float:
        """Learning rate at OMD round r (0-indexed). Matches Algorithm 5 line 15."""
        return self.eta_0 / ((r + 1) ** self.eta_p)
