import numpy as np
from vda.types import TraceStep, Reports, BlameResult


def test_trace_step_fields():
    s = TraceStep(
        t=0, agent_name="alice", action="do X", prior_context="(start of trace)",
        task_description="solve Y", ground_truth="42",
    )
    assert s.t == 0
    assert s.agent_name == "alice"


def test_reports_shape():
    theta = np.array([[0.1, 0.9], [0.2, 0.8], [0.15, 0.85]])
    r = Reports(theta_hat=theta, model_ids=["m@0", "m@1", "m@2"])
    assert r.theta_hat.shape == (3, 2)
    assert len(r.model_ids) == 3


def test_blame_result_fields():
    br = BlameResult(
        theta_bar=np.array([0.2, 0.8]),
        blame_set=[1],
        agent_blame={"alice": 0.0, "bob": 0.8},
        predicted_agent="bob",
        predicted_step=1,
        vcg_payments=None,
        solver_diagnostics={"sweeps": 5},
    )
    assert br.predicted_agent == "bob"
    assert br.vcg_payments is None
