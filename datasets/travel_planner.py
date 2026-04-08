"""
TravelPlanner dataset loader.
Source: HuggingFace osunlp/TravelPlanner

Each task has a travel query with multiple constraints (budget, cuisine, transportation, etc.).
Plans can violate multiple constraints = multi-error ground truth.
We treat each constraint check as a "step" for VDA evaluation.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class TravelPlannerTrace:
    """A single travel planning trace with constraint violations."""
    trace_id: int
    query: str
    plan: str
    constraints: List[str]         # constraint descriptions
    violations: List[int]          # 1 = violated, 0 = satisfied
    T: int = 0
    n_errors: int = 0

    def __post_init__(self):
        self.T = len(self.constraints)
        self.n_errors = sum(self.violations)


def load_travel_planner(
    max_traces: Optional[int] = None,
    min_errors: int = 1,
    max_constraints: int = 20,
    split: str = "validation",
) -> List[TravelPlannerTrace]:
    """
    Load TravelPlanner from HuggingFace.

    The dataset has queries + reference plans + constraint annotations.
    We extract per-constraint pass/fail as multi-error ground truth.

    Args:
        max_traces: limit number of traces loaded
        min_errors: minimum constraint violations per trace
        max_constraints: skip traces with more constraints than this
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("osunlp/TravelPlanner", split=split)
    except Exception:
        # Fallback: try test split
        ds = load_dataset("osunlp/TravelPlanner", split="test")

    # Standard constraint categories for TravelPlanner
    CONSTRAINT_KEYS = [
        "budget", "cuisine", "room_type", "transportation",
        "attractions", "accommodation", "restaurants",
        "duration", "city_count", "diversity",
    ]

    traces = []
    for i, row in enumerate(ds):
        if max_traces and len(traces) >= max_traces:
            break

        query = row.get("query", row.get("question", ""))
        plan = row.get("plan", row.get("output", ""))
        annotations = row.get("annotations", row.get("constraint_results", {}))

        if not query or not annotations:
            continue

        # Extract constraint pass/fail
        constraints = []
        violations = []

        if isinstance(annotations, dict):
            for key in CONSTRAINT_KEYS:
                if key in annotations:
                    val = annotations[key]
                    constraints.append(key)
                    # Various formats: bool, int, string
                    if isinstance(val, bool):
                        violations.append(0 if val else 1)
                    elif isinstance(val, (int, float)):
                        violations.append(0 if val >= 1.0 else 1)
                    elif isinstance(val, str):
                        violations.append(0 if val.lower() in ("pass", "true", "1", "yes") else 1)
                    else:
                        violations.append(0)
        elif isinstance(annotations, list):
            # List of constraint results
            for j, ann in enumerate(annotations):
                if isinstance(ann, dict):
                    name = ann.get("name", ann.get("constraint", f"constraint_{j}"))
                    passed = ann.get("passed", ann.get("result", True))
                    constraints.append(name)
                    violations.append(0 if passed else 1)
                else:
                    constraints.append(f"constraint_{j}")
                    violations.append(0 if ann else 1)

        if len(constraints) == 0 or len(constraints) > max_constraints:
            continue
        if sum(violations) < min_errors:
            continue

        trace = TravelPlannerTrace(
            trace_id=i,
            query=query,
            plan=plan,
            constraints=constraints,
            violations=violations,
        )
        traces.append(trace)

    return traces


def trace_to_steps(trace: TravelPlannerTrace):
    """Convert a TravelPlanner trace to TraceStep objects."""
    from vda.discriminator import TraceStep

    steps = []
    for t in range(trace.T):
        step = TraceStep(
            t=t,
            agent_name="planner",
            state=f"Query: {trace.query}\nPlan: {trace.plan[:500]}",
            action=f"Constraint check [{trace.constraints[t]}]: Does the plan satisfy the {trace.constraints[t]} constraint?",
            task_description=trace.query,
            total_steps=trace.T,
        )
        steps.append(step)

    return steps


def get_ground_truth_vector(trace: TravelPlannerTrace) -> np.ndarray:
    """Get multi-error ground truth. Returns (T,) with 1.0 at violated constraints."""
    return np.array(trace.violations, dtype=float)
