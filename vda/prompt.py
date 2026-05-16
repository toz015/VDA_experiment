"""Prompt builder matching Section 6 of the implementation note."""

from vda.types import TraceStep


def build_json_discriminator_prompt(step: TraceStep) -> str:
    """JSON-output variant of the discriminator prompt.

    Asks the model to emit {"answer": "A"|"B", "confidence": 0-1}. Use this
    when the discriminator backend cannot supply per-token logprobs (e.g.
    Vertex MaaS endpoints for Qwen / GPT-OSS). The mapping back to θ is:
        θ = confidence       if answer == "A"
        θ = 1 - confidence   if answer == "B"
    """
    base = build_discriminator_prompt(step)
    return base + (
        "\n\nRespond with EXACTLY one JSON object and nothing else, in this format:\n"
        '{"answer": "A" or "B", "confidence": <float between 0 and 1>}\n'
        "where confidence is your confidence in the chosen answer."
    )


def build_trial_discriminator_prompt(
    task_description: str,
    ground_truth: str,
    trial_index: int,
    n_trials: int,
    trial_messages: list,
    prior_trials_summary: str = "",
) -> str:
    """Trial-level discriminator prompt (DoVer Stage A).

    Asks whether the root-cause error lies somewhere within this trial.
    `trial_messages` is a list of dicts with keys role, action_type, state, content.
    """
    lines = []
    for i, m in enumerate(trial_messages):
        header = _format_step_header(
            m.get("original_index", i),
            m.get("role", "?"),
            m.get("action_type", ""),
            m.get("state", ""),
        )
        content = (m.get("content", "") or "")[:600]
        lines.append(f"{header}\n{content}")
    trial_block = "\n".join(lines)

    prior_block = (
        f"[PRIOR TRIALS SUMMARY]\n{prior_trials_summary}\n"
        if prior_trials_summary else ""
    )

    return (
        "You are evaluating a TRIAL (a contiguous span of messages between two "
        "planning events) in a multi-agent task execution that ultimately FAILED.\n"
        "Exactly ONE trial contains the single root-cause mistake. If the mistake "
        "in that trial had been corrected, the task would have succeeded.\n"
        f"[TASK] {task_description}\n"
        f"[CORRECT ANSWER] {ground_truth}\n"
        f"{prior_block}"
        f"[TRIAL {trial_index} of {n_trials} — messages below]\n"
        f"{trial_block}\n\n"
        f"Question: Does trial {trial_index} contain the single root-cause mistake?\n"
        "Answer (A) Yes - the root-cause is somewhere in this trial, or "
        "(B) No - the root-cause is in a different trial.\n\n"
        "Respond with EXACTLY one JSON object and nothing else, in this format:\n"
        '{"answer": "A" or "B", "confidence": <float between 0 and 1>}\n'
        "where confidence is your confidence in the chosen answer."
    )


def build_discriminator_prompt(step: TraceStep) -> str:
    """Build the discriminator query prompt for a single trace step.

    Uses a hybrid format:
      - Prior steps: compact structured triples (agent, action_type, state)
      - Current step: structured header + full raw content

    This keeps the context short for prior steps while giving the discriminator
    full detail for the step under evaluation.

    P("A") from the model's logprobs is interpreted as P(step t is the root-cause).
    """
    prior_block = step.prior_context.strip()

    # Current step: structured header + full raw content
    current_header = _format_step_header(step.t, step.agent_name, step.action_type, step.state)
    current_block = f"{current_header}\nFull output:\n{step.action}"

    trace_context = (f"{prior_block}\n\n{current_block}") if prior_block else current_block

    action_desc = f"{step.action_type} — \"{step.state}\"" if step.action_type else "this action"

    return (
        "You are evaluating a step in a multi-agent task execution that ultimately FAILED.\n"
        "Exactly ONE step is the single root-cause mistake. If that step had been done correctly, "
        "the task would have succeeded. All other steps are correct.\n"
        f"[TASK] {step.task_description}\n"
        f"[CORRECT ANSWER] {step.ground_truth}\n"
        f"[TRACE CONTEXT (steps 0 to {step.t})]\n"
        f"{trace_context}\n"
        f"Question: Is the action at step {step.t} (by agent \"{step.agent_name}\", "
        f"{action_desc}) "
        "the single root-cause mistake — i.e., if this step had been done correctly, "
        "would the task have succeeded?\n"
        "Answer (A) Yes - this is the root-cause mistake, or (B) No - this step is correct."
    )


def _format_step_header(t: int, agent_name: str, action_type: str, state: str) -> str:
    """Format a structured step header line."""
    if action_type and state:
        return f"--- Step {t}: [{agent_name}] {action_type} — \"{state}\" ---"
    elif action_type:
        return f"--- Step {t}: [{agent_name}] {action_type} ---"
    else:
        return f"--- Step {t} ({agent_name}) ---"


def format_prior_step(t: int, agent_name: str, action_type: str, state: str) -> str:
    """Format a single prior step as a compact structured line (no raw content)."""
    return _format_step_header(t, agent_name, action_type, state)
