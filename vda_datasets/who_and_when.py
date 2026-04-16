"""
Who&When dataset loader (Section 11).
Source: HuggingFace Kevin355/Who_and_When
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# Legacy role-based filter (kept as fallback when no classification is available)
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

    # Action-only view (set after __post_init__ or classify_steps)
    action_history: List[Dict] = field(default_factory=list)  # filtered action steps
    T_action: int = 0          # number of action steps
    mistake_step_action: Optional[int] = None  # index into action_history, or None

    # Structured classification (populated by classify_steps())
    classified_steps: List[Dict] = field(default_factory=list)  # [{agent, action_type, state, original_index}, ...]
    _classified: bool = False

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

        # Default: use legacy role-based filtering
        # This gets overridden when classify_steps() is called
        if not self._classified:
            self._build_action_view_by_role()

    def _build_action_view_by_role(self):
        """Legacy: filter steps by ACTION_ROLES."""
        self.action_history = [h for h in self.history if h["role"] in ACTION_ROLES]
        self.T_action = len(self.action_history)
        self._remap_mistake_step()

    def _build_action_view_by_classification(self):
        """Build action view using LLM-classified steps.

        Keeps ALL steps (no filtering by action_type) because meta steps like
        plan, instruct, verify, and conclude can be root-cause mistakes in the
        Who&When dataset. The classification is used for structured prompt
        formatting, not filtering.
        """
        self.action_history = []
        for cls in self.classified_steps:
            orig_idx = cls["original_index"]
            h = dict(self.history[orig_idx])
            h["action_type"] = cls["action_type"]
            h["state"] = cls["state"]
            h["original_index"] = orig_idx
            self.action_history.append(h)

        self.T_action = len(self.action_history)
        self._remap_mistake_step()

    def _remap_mistake_step(self):
        """Remap mistake_step from full history index to action_history index."""
        self.mistake_step_action = None
        if 0 <= self.mistake_step < self.T:
            for action_idx, h in enumerate(self.action_history):
                orig_idx = h.get("original_index")
                if orig_idx is not None:
                    if orig_idx == self.mistake_step:
                        self.mistake_step_action = action_idx
                        return
                else:
                    # Legacy path: match by position in role-filtered view
                    pass

            # Legacy fallback for role-based filtering (no original_index)
            if self.mistake_step_action is None:
                ms_role = self.history[self.mistake_step]["role"]
                if ms_role in ACTION_ROLES:
                    action_idx = 0
                    for orig_t, h in enumerate(self.history):
                        if h["role"] in ACTION_ROLES:
                            if orig_t == self.mistake_step:
                                self.mistake_step_action = action_idx
                                return
                            action_idx += 1

    def classify_steps(self, client=None, model: str = "gpt-4o-mini",
                       subset: str = "", cache_dir=None):
        """Run LLM-based step classification and rebuild the action view."""
        from vda.step_classifier import classify_trace

        self.classified_steps = classify_trace(
            history=self.history,
            trace_id=self.trace_id,
            subset=subset,
            cache_dir=cache_dir,
            client=client,
            model=model,
        )
        self._classified = True
        self._build_action_view_by_classification()


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
            ground_truth=row.get("groundtruth", row.get("ground_truth", "")),
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

    Uses action_history (filtered by classification or ACTION_ROLES).
    Step indices t refer to positions in this filtered list, NOT the original full history.

    When classification data is available, prior steps use compact structured format
    (action_type + state only). The current step always includes full raw content.
    """
    from vda.types import TraceStep
    from vda.prompt import format_prior_step

    use_structured = trace._classified

    steps = []
    for t, h in enumerate(trace.action_history):
        prior_parts = []
        for prev_t, ph in enumerate(trace.action_history[:t]):
            if use_structured:
                # Compact structured line for prior steps (no raw content)
                prior_parts.append(format_prior_step(
                    prev_t, ph["role"],
                    ph.get("action_type", ""), ph.get("state", ""),
                ))
            else:
                # Legacy: truncated raw content
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
            action_type=h.get("action_type", ""),
            state=h.get("state", ""),
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
