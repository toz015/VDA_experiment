"""
Who&When dataset loader (Section 11).
Source: HuggingFace Kevin355/Who_and_When
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# Roles that actually take actions in the environment (not Orchestrator internals)
ACTION_ROLES = frozenset({"WebSurfer", "Websurfer", "Assistant", "FileSurfer", "ComputerTerminal"})


@dataclass
class WhoAndWhenTrace:
    """A single failure trace from Who&When."""
    trace_id: int
    question: str
    ground_truth: str
    history: List[Dict]        # list of {name, role, content} — full history
    mistake_agent: str
    mistake_step: int          # index into full history[]
    mistake_reason: str
    T: int = 0                 # total steps (full history)
    M: int = 0                 # number of unique agents
    agents: List[str] = field(default_factory=list)
    agent_steps: Dict[str, List[int]] = field(default_factory=dict)

    # Action-only view (set after __post_init__)
    action_history: List[Dict] = field(default_factory=list)  # filtered to ACTION_ROLES
    T_action: int = 0          # number of action steps
    mistake_step_action: Optional[int] = None  # index into action_history, or None

    def __post_init__(self):
        self.T = len(self.history)
        self.agents = list(dict.fromkeys(h["role"] for h in self.history))
        self.M = len(self.agents)
        self.agent_steps = {}
        for t, h in enumerate(self.history):
            name = h["role"]
            if name not in self.agent_steps:
                self.agent_steps[name] = []
            self.agent_steps[name].append(t)

        # Build action-only view
        self.action_history = [h for h in self.history if h["role"] in ACTION_ROLES]
        self.T_action = len(self.action_history)

        # Remap mistake_step to action index
        if 0 <= self.mistake_step < self.T:
            ms_role = self.history[self.mistake_step]["role"]
            if ms_role in ACTION_ROLES:
                # Find position in action_history
                action_idx = 0
                for orig_t, h in enumerate(self.history):
                    if h["role"] in ACTION_ROLES:
                        if orig_t == self.mistake_step:
                            self.mistake_step_action = action_idx
                            break
                        action_idx += 1


def load_who_and_when(
    subset: str = "Algorithm-Generated",
    action_only: bool = True,
) -> List[WhoAndWhenTrace]:
    """
    Load Who&When dataset from HuggingFace.

    Args:
        subset: "Algorithm-Generated" (126 traces) or "Hand-Crafted" (58 traces)
        action_only: if True (default), skip traces whose ground-truth mistake step
                     does not land on an action-taking role. Keeps only traces
                     where we have a clean ground truth in the filtered step sequence.
    """
    from datasets import load_dataset
    ds = load_dataset("Kevin355/Who_and_When", subset)
    traces = []

    for i, row in enumerate(ds["train"]):
        trace = WhoAndWhenTrace(
            trace_id=i,
            question=row["question"],
            ground_truth=row["groundtruth"],
            history=row["history"],
            mistake_agent=row["mistake_agent"],
            mistake_step=int(row["mistake_step"]),
            mistake_reason=row.get("mistake_reason", ""),
        )
        if action_only and trace.mistake_step_action is None:
            continue
        traces.append(trace)

    return traces


def trace_to_steps(trace: WhoAndWhenTrace, max_tokens_per_prior_step: int = 500):
    """Convert a Who&When trace to a list of vda.types.TraceStep objects.

    Uses action_history (ACTION_ROLES only). Step indices t refer to positions
    in this filtered list, NOT the original full history.
    """
    from vda.types import TraceStep

    steps = []
    for t, h in enumerate(trace.action_history):
        prior_parts = []
        for prev_t, ph in enumerate(trace.action_history[:t]):
            content = ph["content"][:max_tokens_per_prior_step]
            prior_parts.append(f"--- Step {prev_t} ({ph['role']}) ---\n{content}")
        prior_context = "\n".join(prior_parts)

        step = TraceStep(
            t=t,
            agent_name=h["role"],
            action=h["content"],
            prior_context=prior_context,
            task_description=trace.question,
            ground_truth=trace.ground_truth,
        )
        steps.append(step)

    return steps


def get_ground_truth_vector(trace: WhoAndWhenTrace) -> np.ndarray:
    """
    Get ground truth theta vector for the action-only step sequence.
    Returns (T_action,) with 1.0 at mistake_step_action, 0.0 elsewhere.
    """
    gt = np.zeros(trace.T_action)
    if trace.mistake_step_action is not None and 0 <= trace.mistake_step_action < trace.T_action:
        gt[trace.mistake_step_action] = 1.0
    return gt
