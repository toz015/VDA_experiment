from vda.types import TraceStep
from vda.prompt import build_discriminator_prompt


def test_prompt_contains_all_sections():
    step = TraceStep(
        t=2,
        agent_name="agent_B",
        action="I will compute sqrt(16) = 5.",
        prior_context="--- Step 0 (agent_A) ---\nPlan: use sqrt\n--- Step 1 (agent_B) ---\nOK",
        task_description="What is sqrt(16)?",
        ground_truth="4",
    )
    p = build_discriminator_prompt(step)

    assert "multi-agent task execution that ultimately" in p
    assert "FAILED" in p
    assert "What is sqrt(16)?" in p
    assert "[TRACE CONTEXT (steps 0 to 2)]" in p
    assert "--- Step 0 (agent_A) ---" in p
    assert "--- Step 1 (agent_B) ---" in p
    assert "--- Step 2 (agent_B) ---" in p
    assert "I will compute sqrt(16) = 5." in p
    assert "Ground truth: 4" in p
    assert 'agent "agent_B"' in p
    assert "(A) Yes or (B) No" in p


def test_prompt_at_step_zero():
    step = TraceStep(
        t=0,
        agent_name="agent_A",
        action="First action",
        prior_context="",  # no prior steps
        task_description="Task",
        ground_truth="GT",
    )
    p = build_discriminator_prompt(step)
    assert "--- Step 0 (agent_A) ---" in p
    assert "First action" in p
