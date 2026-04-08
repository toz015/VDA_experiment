"""
Math-Shepherd dataset loader.
Source: HuggingFace peiyi9979/Math-Shepherd

Each row: problem + step-by-step solution where every step is labeled +/-.
Multiple steps can be labeled "-" = multi-error ground truth.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MathShepherdTrace:
    """A single math solution trace."""
    trace_id: int
    problem: str
    steps: List[str]           # step text
    labels: List[int]          # 1 = error ("-"), 0 = correct ("+")
    T: int = 0
    n_errors: int = 0

    def __post_init__(self):
        self.T = len(self.steps)
        self.n_errors = sum(self.labels)


def parse_solution(input_text: str, label_text: str) -> Tuple[List[str], List[int]]:
    """
    Parse Math-Shepherd format.
    Steps are separated by "Step N:" pattern.
    Labels use "+" for correct, "-" for incorrect per step.
    """
    # Split by "Step N:" pattern
    step_pattern = re.compile(r'Step \d+:', re.IGNORECASE)
    parts = step_pattern.split(label_text)

    # Also extract labels from the label field
    # Math-Shepherd format: "Step 1: ... +\nStep 2: ... -\n..."
    steps = []
    labels = []

    lines = label_text.strip().split('\n')
    current_step = []
    current_label = 0

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check if line ends with label marker
        if line_stripped.endswith('+') or line_stripped.endswith('-'):
            label_char = line_stripped[-1]
            step_text = line_stripped[:-1].strip()
            # Remove "Step N:" prefix
            step_text = re.sub(r'^Step \d+:\s*', '', step_text)

            if current_step:
                current_step.append(step_text)
                steps.append(' '.join(current_step))
            else:
                steps.append(step_text)

            labels.append(1 if label_char == '-' else 0)
            current_step = []
        else:
            # Multi-line step, accumulate
            step_text = re.sub(r'^Step \d+:\s*', '', line_stripped)
            current_step.append(step_text)

    # Handle remaining text
    if current_step:
        steps.append(' '.join(current_step))
        labels.append(0)  # default to correct if no label

    return steps, labels


def load_math_shepherd(
    max_traces: Optional[int] = None,
    min_errors: int = 1,
    max_steps: int = 20,
    split: str = "train",
) -> List[MathShepherdTrace]:
    """
    Load Math-Shepherd from HuggingFace.

    Args:
        max_traces: limit number of traces loaded
        min_errors: minimum number of error steps per trace (1+ for multi-error)
        max_steps: skip traces with more steps than this
    """
    from datasets import load_dataset
    ds = load_dataset("peiyi9979/Math-Shepherd", split=split)

    traces = []
    for i, row in enumerate(ds):
        if max_traces and len(traces) >= max_traces:
            break

        problem = row.get("input", row.get("question", ""))
        label_text = row.get("label", row.get("output", ""))

        if not label_text:
            continue

        steps, labels = parse_solution(problem, label_text)

        if len(steps) == 0 or len(steps) > max_steps:
            continue
        if len(steps) != len(labels):
            continue
        if sum(labels) < min_errors:
            continue

        trace = MathShepherdTrace(
            trace_id=i,
            problem=problem,
            steps=steps,
            labels=labels,
        )
        traces.append(trace)

    return traces


def trace_to_steps(trace: MathShepherdTrace):
    """Convert a Math-Shepherd trace to TraceStep objects."""
    from vda.discriminator import TraceStep

    steps = []
    for t in range(trace.T):
        # State = all previous steps
        prev_steps = "\n".join(
            f"Step {i+1}: {trace.steps[i]}" for i in range(t)
        )
        state = prev_steps if prev_steps else "(start of solution)"

        step = TraceStep(
            t=t,
            agent_name="solver",
            state=state,
            action=f"Step {t+1}: {trace.steps[t]}",
            task_description=trace.problem,
            total_steps=trace.T,
        )
        steps.append(step)

    return steps


def get_ground_truth_vector(trace: MathShepherdTrace) -> np.ndarray:
    """Get multi-error ground truth. Returns (T,) with 1.0 at error steps."""
    return np.array(trace.labels, dtype=float)
