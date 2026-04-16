from vda.types import TraceStep
from vda.prompt import build_discriminator_prompt, format_prior_step


def test_prompt_contains_all_sections():
    """Legacy format (no action_type/state): prior context passed as-is."""
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
    assert "I will compute sqrt(16) = 5." in p
    assert "4" in p                        # ground truth visible
    assert 'agent "agent_B"' in p
    assert "(A) Yes" in p
    assert "(B) No" in p


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
    assert "Step 0" in p
    assert "agent_A" in p
    assert "First action" in p


def test_prompt_structured_format():
    """With action_type and state, prompt uses structured header + full raw content."""
    prior = format_prior_step(0, "Agent_A", "plan", "identify ingredients")
    step = TraceStep(
        t=1,
        agent_name="Agent_B",
        action="Running script: import pandas as pd\ndf = pd.read_csv('data.csv')",
        prior_context=prior,
        task_description="Analyze the data",
        ground_truth="42",
        action_type="execute",
        state="run data analysis script",
    )
    p = build_discriminator_prompt(step)

    # Prior step is compact structured line (no raw content)
    assert "[Agent_A] plan" in p
    assert "identify ingredients" in p

    # Current step has structured header AND full raw content
    assert "[Agent_B] execute" in p
    assert "run data analysis script" in p
    assert "Full output:" in p
    assert "import pandas as pd" in p

    # Question includes action description
    assert 'execute — "run data analysis script"' in p


def test_format_prior_step():
    line = format_prior_step(3, "WebSurfer", "search", "search for GPT-4 release date")
    assert "Step 3" in line
    assert "[WebSurfer]" in line
    assert "search" in line
    assert "GPT-4 release date" in line

    # Without state
    line2 = format_prior_step(0, "Agent", "plan", "")
    assert "[Agent] plan" in line2

    # Without action_type (legacy fallback)
    line3 = format_prior_step(0, "Agent", "", "")
    assert "(Agent)" in line3
