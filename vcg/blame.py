"""Phase 4: Blame attribution (Algorithm 6 lines 25-31).

Given the final consensus theta_bar and per-step agent assignment i_t,
- S* = {t : theta_bar_t > c_t}
- Pi_agent_i = sum_{t in S* and i_t == i} theta_bar_t
- predicted blame agent = argmax_i Pi_agent_i
- predicted blame step  = argmax_t theta_bar_t
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class BlameResult:
    blame_set: List[int]            # S* — step indices above threshold
    agent_blame: Dict[str, float]   # name -> Pi_agent_i
    predicted_agent: Optional[str]  # argmax (None when S* is empty)
    predicted_step: int             # argmax_t theta_bar_t (always defined)


def attribute_blame(
    theta_bar: np.ndarray,
    step_agents: Sequence[str],
    *,
    c_t: float = 0.5,
) -> BlameResult:
    """Phase 4 of Algorithm 5/6.

    Args:
        theta_bar: (T,) final consensus per step.
        step_agents: length-T sequence of agent identifiers for each step.
        c_t: scalar threshold (Section 7 default 0.5). Per-step thresholds
            can be supplied by callers that vary c_t along t.

    Returns BlameResult.
    """
    theta_bar = np.asarray(theta_bar, dtype=np.float64)
    if theta_bar.ndim != 1:
        raise ValueError(f"theta_bar must be 1-D; got shape {theta_bar.shape}")
    if len(step_agents) != theta_bar.shape[0]:
        raise ValueError(
            f"step_agents length {len(step_agents)} != theta_bar length {theta_bar.shape[0]}"
        )

    blame_set = [int(t) for t in range(theta_bar.shape[0]) if theta_bar[t] > c_t]

    agent_blame: Dict[str, float] = {}
    for t in blame_set:
        a = step_agents[t]
        agent_blame[a] = agent_blame.get(a, 0.0) + float(theta_bar[t])

    predicted_agent = (
        max(agent_blame.items(), key=lambda kv: kv[1])[0] if agent_blame else None
    )
    predicted_step = int(np.argmax(theta_bar))
    return BlameResult(
        blame_set=blame_set,
        agent_blame=agent_blame,
        predicted_agent=predicted_agent,
        predicted_step=predicted_step,
    )
