"""
Who&When dataset loader (Section 11).
Source: HuggingFace Kevin355/Who_and_When
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class WhoAndWhenTrace:
    """A single failure trace from Who&When."""
    trace_id: int
    question: str
    ground_truth: str
    history: List[Dict]        # list of {name, role, content}
    mistake_agent: str
    mistake_step: int
    mistake_reason: str
    T: int = 0                 # number of steps
    M: int = 0                 # number of agents
    agents: List[str] = field(default_factory=list)
    agent_steps: Dict[str, List[int]] = field(default_factory=dict)

    def __post_init__(self):
        self.T = len(self.history)
        self.agents = list(dict.fromkeys(h["name"] for h in self.history))
        self.M = len(self.agents)
        self.agent_steps = {}
        for t, h in enumerate(self.history):
            name = h["name"]
            if name not in self.agent_steps:
                self.agent_steps[name] = []
            self.agent_steps[name].append(t)


def load_who_and_when(subset: str = "Algorithm-Generated") -> List[WhoAndWhenTrace]:
    """
    Load Who&When dataset from HuggingFace.

    Args:
        subset: "Algorithm-Generated" (126 traces) or "Hand-Crafted" (58 traces)
    """
    from datasets import load_dataset
    ds = load_dataset("Kevin355/Who_and_When", subset)
    traces = []

    for i, row in enumerate(ds["train"]):
        trace = WhoAndWhenTrace(
            trace_id=i,
            question=row["question"],
            ground_truth=row["ground_truth"],
            history=row["history"],
            mistake_agent=row["mistake_agent"],
            mistake_step=int(row["mistake_step"]),
            mistake_reason=row.get("mistake_reason", ""),
        )
        traces.append(trace)

    return traces


def trace_to_steps(trace: WhoAndWhenTrace):
    """Convert a Who&When trace to a list of TraceStep objects for discriminators."""
    from vda.discriminator import TraceStep

    steps = []
    for t in range(trace.T):
        h = trace.history[t]
        # Build state = prefix context (all steps up to t)
        context_parts = []
        for prev_t in range(t + 1):
            ph = trace.history[prev_t]
            content = ph["content"][:500] if prev_t < t else ph["content"]
            context_parts.append(f"--- Step {prev_t} ({ph['name']}) ---\n{content}")
        state = "\n".join(context_parts[:-1]) if t > 0 else "(start of trace)"

        step = TraceStep(
            t=t,
            agent_name=h["name"],
            state=state,
            action=h["content"],
            task_description=trace.question,
            ground_truth=trace.ground_truth,
            total_steps=trace.T,
        )
        steps.append(step)

    return steps


def get_ground_truth_vector(trace: WhoAndWhenTrace) -> np.ndarray:
    """
    Get ground truth theta vector. For Who&When, only one step is labeled.
    Returns (T,) with 1.0 at mistake_step, 0.0 elsewhere.
    """
    gt = np.zeros(trace.T)
    if 0 <= trace.mistake_step < trace.T:
        gt[trace.mistake_step] = 1.0
    return gt
