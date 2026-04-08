"""Prompt builder matching Section 6 of the implementation note."""

from vda.types import TraceStep


def build_discriminator_prompt(step: TraceStep) -> str:
    """Build the discriminator query prompt for a single trace step.

    Format is verbatim from Section 6 of the implementation note.
    """
    prior_block = step.prior_context.strip()
    current_block = f"--- Step {step.t} ({step.agent_name}) ---\n{step.action}"

    if prior_block:
        trace_context = f"{prior_block}\n{current_block}"
    else:
        trace_context = current_block

    return (
        "You are evaluating a step in a multi-agent task execution that ultimately FAILED.\n"
        f"[TASK] {step.task_description}\n"
        f"[TRACE CONTEXT (steps 0 to {step.t})]\n"
        f"{trace_context}\n"
        f"[TASK OUTCOME: FAILED. Ground truth: {step.ground_truth}]\n"
        f'Question: Does the action at step {step.t} (by agent "{step.agent_name}") '
        "contain an error that contributed to the task failure?\n"
        "Answer (A) Yes or (B) No."
    )
