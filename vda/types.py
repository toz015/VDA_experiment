"""Data types for VDA. Plain dataclasses, no business logic."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TraceStep:
    t: int
    agent_name: str
    action: str              # history[t]["content"], untruncated raw content
    prior_context: str       # formatted prior steps (structured or raw)
    task_description: str
    ground_truth: str
    action_type: str = ""    # e.g. "execute", "search", "plan", "inform", ...
    state: str = ""          # brief summary of what the step does/produces


@dataclass
class Reports:
    theta_hat: np.ndarray    # shape (K, T), values in [eps, 1-eps]
    model_ids: list          # length K, e.g. ["gpt-4o-mini@T=0.0", ...]


@dataclass
class BlameResult:
    theta_bar: np.ndarray              # shape (T,)
    blame_set: list                    # list[int]
    agent_blame: dict                  # dict[str, float]
    predicted_agent: str
    predicted_step: int
    vcg_payments: Optional[np.ndarray] # shape (K,), None in Stage 1
    solver_diagnostics: dict
