"""Prompt builder matching Section 6 of the implementation note."""

from vda.types import TraceStep


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
