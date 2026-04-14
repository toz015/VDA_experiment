"""Prompt builder matching Section 6 of the implementation note."""

from vda.types import TraceStep


def build_discriminator_prompt(step: TraceStep) -> str:
    """Build the discriminator query prompt for a single trace step.

    Asks whether this specific step is the single root-cause mistake in the trace.
    P("A") from the model's logprobs is interpreted as P(step t is the root-cause).
    """
    prior_block = step.prior_context.strip()
    current_block = f"--- Step {step.t} ({step.agent_name}) ---\n{step.action}"
    trace_context = (f"{prior_block}\n{current_block}") if prior_block else current_block

    return (
        "You are evaluating a step in a multi-agent task execution that ultimately FAILED.\n"
        "Exactly ONE step is the single root-cause mistake. If that step had been done correctly, "
        "the task would have succeeded. All other steps are correct.\n"
        f"[TASK] {step.task_description}\n"
        f"[CORRECT ANSWER] {step.ground_truth}\n"
        f"[TRACE CONTEXT (steps 0 to {step.t})]\n"
        f"{trace_context}\n"
        f"Question: Is the action at step {step.t} (by agent \"{step.agent_name}\") "
        "the single root-cause mistake — i.e., if this step had been done correctly, "
        "would the task have succeeded?\n"
        "Answer (A) Yes - this is the root-cause mistake, or (B) No - this step is correct."
    )
