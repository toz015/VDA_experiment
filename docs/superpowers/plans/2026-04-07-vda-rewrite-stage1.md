# VDA Rewrite Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Reputation-Weighted Multiplicative VCG mechanism on the Who&When benchmark. Stage 1a generates and saves LLM discriminator probability matrices to disk. Stage 1b implements the Gauss–Seidel allocation and blame attribution offline on the saved matrices. OMD calibration (Stage 2) is a follow-on plan.

**Architecture:** Flat modules with a pure-numpy math core (`vda/core.py`) decoupled from LLM plumbing (`vda/discriminator.py`) and orchestration (`vda/pipeline.py`). All expensive LLM calls happen once in Stage 1a and are cached to disk as `.npz` files; Stage 1b and all later stages read those cached files offline.

**Tech Stack:** Python 3.11+, numpy, openai SDK (logprobs API), HuggingFace `datasets` (for `Kevin355/Who_and_When`), pytest.

**Spec:** `docs/superpowers/specs/2026-04-07-vda-rewrite-design.md`

---

## Task 0: Wipe old code and set up directory structure

**Files:**
- Delete: `vda/` (entire directory)
- Delete: `scripts/` (entire directory)
- Delete: `evaluation/metrics.py`, `evaluation/runner.py` (keep `evaluation/__init__.py`)
- Keep: `datasets/who_and_when.py`, `datasets/__init__.py`, `config.py`, `requirements.txt`, `implementation-note-2026-04-07.pdf`
- Create: empty directories `vda/`, `scripts/`, `tests/`, `data/reports/`

- [ ] **Step 1: Inspect what will be deleted**

Run: `ls vda/ scripts/ evaluation/`
Expected: shows old files. Note: `datasets/who_and_when.py` and `config.py` are NOT in the delete list.

- [ ] **Step 2: Delete old code**

```bash
cd /Users/wanghd/Desktop/Research/Dai/vda_experiment
rm -rf vda scripts
rm -f evaluation/metrics.py evaluation/runner.py
```

- [ ] **Step 3: Create new directory skeleton**

```bash
mkdir -p vda scripts tests data/reports
touch vda/__init__.py tests/__init__.py
```

- [ ] **Step 4: Update requirements.txt**

Replace entire content:

```
numpy>=1.24
datasets>=2.14
openai>=1.0
tqdm>=4.65
pytest>=7.4
```

(Dropped: `scipy`, `scikit-learn`, `anthropic` — not needed for Stage 1.)

- [ ] **Step 5: Verify structure**

Run: `ls vda scripts tests data`
Expected: all four directories present, `vda/__init__.py` and `tests/__init__.py` exist.

---

## Task 1: Update config.py to match Section 7 of the note

**Files:**
- Modify: `config.py` (full rewrite)

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from config import VDAConfig


def test_default_config_matches_note():
    cfg = VDAConfig()
    assert cfg.K == 3
    assert cfg.openai_model == "gpt-4o-mini"
    assert cfg.temperatures == [0.0, 0.7, 1.0]
    assert cfg.R == 15
    assert cfg.eta_0 == 0.3
    assert cfg.eta_p == 0.7
    assert cfg.L == 100
    assert cfg.tau == 1e-10
    assert cfg.delta == 1e-4
    assert cfg.eps == 1e-6
    assert cfg.c_t == 0.5
    assert cfg.prompt_max_tokens_per_prior_step == 500


def test_learning_rate_schedule():
    cfg = VDAConfig()
    # r is 0-indexed; formula is eta_0 / (r+1)^p
    assert abs(cfg.lr(0) - 0.3) < 1e-12                # 0.3 / 1^0.7
    assert abs(cfg.lr(1) - 0.3 / (2 ** 0.7)) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — old `VDAConfig` has different defaults / signatures.

- [ ] **Step 3: Rewrite config.py**

Replace entire file:

```python
"""Hyperparameters and defaults for VDA (matches Section 7 of the implementation note)."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class VDAConfig:
    # Discriminators (Stage 1: single model, K temperatures)
    K: int = 3
    openai_model: str = "gpt-4o-mini"
    temperatures: List[float] = field(default_factory=lambda: [0.0, 0.7, 1.0])

    # OMD calibration (Stage 2)
    R: int = 15
    eta_0: float = 0.3
    eta_p: float = 0.7
    omd_tol: float = 1e-5

    # Gauss–Seidel solver
    L: int = 100
    tau: float = 1e-10

    # Gradient
    delta: float = 1e-4
    eps: float = 1e-6

    # Blame
    c_t: float = 0.5

    # Prompt
    prompt_max_tokens_per_prior_step: int = 500

    def lr(self, r: int) -> float:
        """Learning rate at OMD round r (0-indexed). Matches Algorithm 5 line 15."""
        return self.eta_0 / ((r + 1) ** self.eta_p)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git init -q 2>/dev/null || true
git add -A
git commit -m "chore: wipe old vda code, refresh config and requirements"
```

---

## Task 2: Data types (vda/types.py)

**Files:**
- Create: `vda/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_types.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py -v`
Expected: FAIL — `vda.types` does not exist.

- [ ] **Step 3: Implement vda/types.py**

Create `vda/types.py`:

```python
"""Data types for VDA. Plain dataclasses, no business logic."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TraceStep:
    t: int
    agent_name: str
    action: str              # history[t]["content"], untruncated
    prior_context: str       # "--- Step 0 (...) ---\n<content[:500]>\n..." up to step t-1
    task_description: str
    ground_truth: str


@dataclass
class Reports:
    theta_hat: np.ndarray    # shape (K, T), values in [eps, 1-eps]
    model_ids: list          # length K, e.g. ["gpt-4o-mini@T=0.0", ...]


@dataclass
class BlameResult:
    theta_bar: np.ndarray              # shape (T,)
    blame_set: list                    # list[int]
    agent_blame: dict                  # dict[str, float]
    predicted_agent: str
    predicted_step: int
    vcg_payments: Optional[np.ndarray] # shape (K,), None in Stage 1
    solver_diagnostics: dict
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_types.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add vda/types.py tests/test_types.py
git commit -m "feat: add vda.types dataclasses"
```

---

## Task 3: Prompt builder (vda/prompt.py)

**Files:**
- Create: `vda/prompt.py`
- Create: `tests/test_prompt.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompt.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement vda/prompt.py**

Create `vda/prompt.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add vda/prompt.py tests/test_prompt.py
git commit -m "feat: add discriminator prompt builder (Section 6 template)"
```

---

## Task 4: Discriminator with logprob extraction (vda/discriminator.py)

**Files:**
- Create: `vda/discriminator.py`
- Create: `tests/test_discriminator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_discriminator.py`:

```python
import math
from unittest.mock import MagicMock
import pytest

from vda.discriminator import OpenAIDiscriminator, extract_probability_a


def _make_logprob_entry(token, logprob):
    m = MagicMock()
    m.token = token
    m.logprob = logprob
    return m


def test_extract_probability_a_basic():
    # p(A) = exp(-0.1), p(B) = exp(-1.0); after renormalization p_A / (p_A + p_B)
    entries = [
        _make_logprob_entry(" A", -0.1),
        _make_logprob_entry(" B", -1.0),
        _make_logprob_entry(" C", -5.0),
    ]
    p, fallback = extract_probability_a(entries)
    p_a = math.exp(-0.1)
    p_b = math.exp(-1.0)
    expected = p_a / (p_a + p_b)
    assert abs(p - expected) < 1e-10
    assert fallback is False


def test_extract_matches_whitespace_and_case():
    for token in ["A", " A", "a", " a"]:
        entries = [_make_logprob_entry(token, -0.2), _make_logprob_entry("B", -0.8)]
        p, fallback = extract_probability_a(entries)
        assert 0.0 < p < 1.0
        assert fallback is False


def test_extract_fallback_when_neither_present():
    entries = [_make_logprob_entry("X", -0.1), _make_logprob_entry("Y", -0.2)]
    p, fallback = extract_probability_a(entries)
    assert p == 0.5
    assert fallback is True


def test_extract_only_a_present():
    # Only "A" token seen → p(A) = 1 before clipping
    entries = [_make_logprob_entry("A", -0.1), _make_logprob_entry("Z", -0.2)]
    p, fallback = extract_probability_a(entries)
    assert p == 1.0
    assert fallback is False


def test_discriminator_query_calls_openai_and_returns_prob(monkeypatch):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.logprobs.content = [
        MagicMock(top_logprobs=[
            _make_logprob_entry(" A", -0.1),
            _make_logprob_entry(" B", -1.0),
        ])
    ]
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    disc = OpenAIDiscriminator(model="gpt-4o-mini", temperature=0.7, client=mock_client)
    p = disc.query("some prompt")

    assert 0.0 < p < 1.0
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 1
    assert call_kwargs["logprobs"] is True
    assert call_kwargs["top_logprobs"] == 20
    assert disc.id == "gpt-4o-mini@T=0.7"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_discriminator.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement vda/discriminator.py**

Create `vda/discriminator.py`:

```python
"""OpenAI discriminator that extracts P('A') from logprobs."""

import math
from typing import Optional, Tuple, List

from config import VDAConfig


def extract_probability_a(top_logprob_entries) -> Tuple[float, bool]:
    """Extract P('A') / (P('A') + P('B')) from a list of top-logprob entries.

    Each entry must expose `.token` and `.logprob`. Tokens are normalized with
    `.strip().upper()` before matching. Returns (probability, fallback_flag);
    fallback_flag is True iff neither 'A' nor 'B' was found (p=0.5 default).
    """
    logprob_a = None
    logprob_b = None
    for e in top_logprob_entries:
        key = e.token.strip().upper()
        if key == "A" and logprob_a is None:
            logprob_a = e.logprob
        elif key == "B" and logprob_b is None:
            logprob_b = e.logprob

    if logprob_a is None and logprob_b is None:
        return 0.5, True
    if logprob_a is None:
        return 0.0, False
    if logprob_b is None:
        return 1.0, False

    # Softmax-normalize the two tokens.
    m = max(logprob_a, logprob_b)
    e_a = math.exp(logprob_a - m)
    e_b = math.exp(logprob_b - m)
    return e_a / (e_a + e_b), False


class OpenAIDiscriminator:
    """One discriminator instance = one (model, temperature) pair."""

    def __init__(
        self,
        model: str,
        temperature: float,
        client=None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.id = f"{model}@T={temperature}"
        self.fallback_count = 0

        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)

    def query(self, prompt: str) -> float:
        """Query the LLM and return P('A') ∈ [0, 1]."""
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            messages=[{"role": "user", "content": prompt}],
        )
        top = resp.choices[0].logprobs.content[0].top_logprobs
        p, fallback = extract_probability_a(top)
        if fallback:
            self.fallback_count += 1
        return p


def build_ensemble(
    config: VDAConfig,
    api_key: Optional[str] = None,
) -> List[OpenAIDiscriminator]:
    """Build K discriminators from a single model at K different temperatures."""
    return [
        OpenAIDiscriminator(model=config.openai_model, temperature=t, api_key=api_key)
        for t in config.temperatures
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_discriminator.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add vda/discriminator.py tests/test_discriminator.py
git commit -m "feat: add OpenAIDiscriminator with logprob extraction"
```

---

## Task 5: Report generation script (scripts/generate_reports.py)

**Files:**
- Create: `scripts/generate_reports.py`
- Create: `scripts/__init__.py` (empty, to allow module imports in tests)

**Note:** This task has no unit test because the output depends on real LLM calls. The test is the manual smoke run in Task 6 and the Stage 1a acceptance criteria.

- [ ] **Step 1: Create scripts package marker**

```bash
touch scripts/__init__.py
```

- [ ] **Step 2: Implement generate_reports.py**

Create `scripts/generate_reports.py`:

```python
"""Stage 1a: Generate discriminator probability matrices for every trace in a subset.

For each trace, runs KT LLM calls and saves a .npz file with:
  - theta_hat: (K, T) numpy array, clipped to [eps, 1-eps]
  - model_ids: list of K strings
  - fallback_counts: (K,) int array of per-discriminator fallback events
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running as `python scripts/generate_reports.py` from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import VDAConfig
from datasets.who_and_when import load_who_and_when, trace_to_steps, WhoAndWhenTrace
from vda.discriminator import build_ensemble
from vda.prompt import build_discriminator_prompt


def generate_for_trace(trace: WhoAndWhenTrace, discriminators, config: VDAConfig):
    """Run KT queries for one trace. Returns theta_hat (K,T) and fallback counts (K,)."""
    steps = trace_to_steps(trace)
    K, T = len(discriminators), trace.T
    theta = np.empty((K, T), dtype=np.float64)

    for k, disc in enumerate(discriminators):
        start_fb = disc.fallback_count
        for t, step in enumerate(steps):
            prompt = build_discriminator_prompt(step)
            theta[k, t] = disc.query(prompt)

    theta = np.clip(theta, config.eps, 1.0 - config.eps)
    fallback_counts = np.array(
        [d.fallback_count for d in discriminators], dtype=np.int64
    )
    return theta, fallback_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="Hand-Crafted",
                        choices=["Hand-Crafted", "Algorithm-Generated"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N traces")
    parser.add_argument("--trace-ids", type=str, default=None,
                        help="Comma-separated trace ids to process (overrides --limit)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip traces whose .npz file already exists")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to data/reports/<subset>")
    args = parser.parse_args()

    config = VDAConfig()
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "data" / "reports" / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading subset '{args.subset}'...")
    traces = load_who_and_when(subset=args.subset)
    print(f"Loaded {len(traces)} traces.")

    if args.trace_ids:
        wanted = {int(x) for x in args.trace_ids.split(",")}
        traces = [t for t in traces if t.trace_id in wanted]
    elif args.limit is not None:
        traces = traces[: args.limit]

    discriminators = build_ensemble(config)
    print(f"Built {len(discriminators)} discriminators: {[d.id for d in discriminators]}")

    manifest = {
        "subset": args.subset,
        "model": config.openai_model,
        "temperatures": config.temperatures,
        "K": config.K,
        "eps": config.eps,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "traces": [],
    }
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    existing_ids = {entry["trace_id"] for entry in manifest.get("traces", [])}

    for trace in tqdm(traces, desc="traces"):
        out_path = out_dir / f"{trace.trace_id}.npz"
        if args.resume and out_path.exists() and trace.trace_id in existing_ids:
            continue

        theta, fb = generate_for_trace(trace, discriminators, config)

        np.savez(
            out_path,
            theta_hat=theta,
            model_ids=np.array([d.id for d in discriminators]),
            fallback_counts=fb,
            mistake_agent=trace.mistake_agent,
            mistake_step=trace.mistake_step,
        )

        manifest["traces"] = [e for e in manifest.get("traces", []) if e["trace_id"] != trace.trace_id]
        manifest["traces"].append({
            "trace_id": trace.trace_id,
            "T": trace.T,
            "agents": trace.agents,
            "mistake_agent": trace.mistake_agent,
            "mistake_step": trace.mistake_step,
            "fallback_counts": fb.tolist(),
        })
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"Done. Reports in {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Dry-lint by importing**

Run: `python -c "import scripts.generate_reports"` from the project root.
Expected: no ImportError (warnings are fine).

- [ ] **Step 4: Commit**

```bash
git add scripts/__init__.py scripts/generate_reports.py
git commit -m "feat: add scripts/generate_reports.py for Stage 1a data generation"
```

---

## Task 6: Report inspection script (scripts/inspect_reports.py)

**Files:**
- Create: `scripts/inspect_reports.py`

- [ ] **Step 1: Implement inspect_reports.py**

Create `scripts/inspect_reports.py`:

```python
"""Stage 1a: Sanity-check a directory of generated report .npz files."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="Hand-Crafted")
    parser.add_argument("--dir", default=None)
    args = parser.parse_args()

    report_dir = Path(args.dir) if args.dir else ROOT / "data" / "reports" / args.subset
    if not report_dir.exists():
        print(f"ERROR: {report_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    files = sorted(report_dir.glob("*.npz"))
    print(f"Found {len(files)} report files in {report_dir}")

    if not files:
        return

    all_theta_per_k = None
    total_fallback = None
    nan_count = 0
    shape_problems = 0
    K_seen = None

    for f in files:
        data = np.load(f, allow_pickle=True)
        theta = data["theta_hat"]
        fb = data["fallback_counts"]

        if K_seen is None:
            K_seen = theta.shape[0]
            all_theta_per_k = [[] for _ in range(K_seen)]
            total_fallback = np.zeros(K_seen, dtype=np.int64)

        if theta.shape[0] != K_seen:
            shape_problems += 1
            continue

        if np.isnan(theta).any():
            nan_count += 1

        for k in range(K_seen):
            all_theta_per_k[k].extend(theta[k].tolist())
        total_fallback += fb

    print(f"\n=== Shape/NaN check ===")
    print(f"NaN-containing traces: {nan_count}")
    print(f"Shape-mismatch traces: {shape_problems}")

    print(f"\n=== Per-discriminator distribution ===")
    total_queries_per_k = sum(len(x) for x in all_theta_per_k) // K_seen if K_seen else 0
    for k, vals in enumerate(all_theta_per_k):
        arr = np.array(vals)
        fb_rate = total_fallback[k] / max(len(arr), 1)
        print(
            f"  k={k}: n={len(arr):5d} "
            f"mean={arr.mean():.3f} std={arr.std():.3f} "
            f"min={arr.min():.3f} max={arr.max():.3f} "
            f"fallback_rate={fb_rate:.4f}"
        )

    print(f"\n=== Disagreement check ===")
    # Approximate: average pairwise L1 distance between discriminators per step.
    per_step = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        theta = data["theta_hat"]
        K, T = theta.shape
        if K < 2:
            continue
        pair_diffs = []
        for i in range(K):
            for j in range(i + 1, K):
                pair_diffs.append(np.abs(theta[i] - theta[j]).mean())
        per_step.append(np.mean(pair_diffs))
    if per_step:
        mean_dis = float(np.mean(per_step))
        print(f"  Mean pairwise L1 disagreement: {mean_dis:.4f}")
        if mean_dis < 0.01:
            print("  WARNING: very low disagreement — discriminators may be collapsed.")

    manifest_path = report_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            m = json.load(f)
        print(f"\nManifest: model={m.get('model')} K={m.get('K')} "
              f"traces_recorded={len(m.get('traces', []))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Import smoke test**

Run: `python -c "import scripts.inspect_reports"`
Expected: no ImportError.

- [ ] **Step 3: Commit**

```bash
git add scripts/inspect_reports.py
git commit -m "feat: add scripts/inspect_reports.py for Stage 1a sanity checks"
```

---

## Task 7: Manual Stage 1a smoke test (no unit test)

**Files:** None. This is a manual validation step.

- [ ] **Step 1: Set OPENAI_API_KEY**

Ensure the environment variable is exported:

```bash
echo "${OPENAI_API_KEY:0:8}..."
```

Expected: prints the first 8 chars of the key. If empty, ask the user to `export OPENAI_API_KEY=...`.

- [ ] **Step 2: Generate reports on two traces**

```bash
cd /Users/wanghd/Desktop/Research/Dai/vda_experiment
python scripts/generate_reports.py --subset Hand-Crafted --limit 2
```

Expected: completes without exceptions; creates `data/reports/Hand-Crafted/0.npz`, `1.npz`, and `manifest.json`.

- [ ] **Step 3: Inspect the generated reports**

```bash
python scripts/inspect_reports.py --subset Hand-Crafted
```

Expected output should show: NaN-containing traces = 0, shape-mismatch traces = 0, non-trivial per-discriminator spread (std > 0.05), fallback_rate < 0.05 per discriminator, non-zero pairwise disagreement.

- [ ] **Step 4: If smoke test passes, generate all 58 traces**

```bash
python scripts/generate_reports.py --subset Hand-Crafted --resume
python scripts/inspect_reports.py --subset Hand-Crafted
```

Expected: all 58 `.npz` files present, no warnings from the inspector. **Report results to user and confirm before proceeding to Task 8.**

---

## Task 8: Gauss–Seidel allocation solver — part 1: compute_value

**Files:**
- Create: `vda/core.py`
- Create: `tests/test_core.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core.py`:

```python
import numpy as np
import pytest

from vda.core import compute_value


def test_compute_value_worked_example_columns():
    # Section 10 worked example reports
    theta = np.array([
        [0.20, 0.85, 0.10, 0.75],
        [0.30, 0.60, 0.80, 0.70],
        [0.25, 0.90, 0.15, 0.80],
    ])
    d = np.array([0.25, 0.783, 0.35, 0.75])
    v = compute_value(theta, d)
    assert v.shape == (3,)
    # All values should lie in (0, 1].
    assert np.all(v > 0) and np.all(v <= 1)

    # Spot-check v_1 = prod_t [1 - (d_t - theta[0,t])^2]
    expected_v1 = (1 - (0.25 - 0.20) ** 2) * (1 - (0.783 - 0.85) ** 2) \
                  * (1 - (0.35 - 0.10) ** 2) * (1 - (0.75 - 0.75) ** 2)
    assert abs(v[0] - expected_v1) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_core.py -v`
Expected: FAIL — `vda.core` module does not exist.

- [ ] **Step 3: Implement compute_value**

Create `vda/core.py`:

```python
"""Pure-numpy mathematical core for VDA (Algorithms 1-4)."""

from typing import Optional, Tuple, Dict, Any
import numpy as np


def compute_value(theta_hat: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Multiplicative value function (Eq. 1 of the note).

    v_k(d, theta_hat) = prod_t [1 - (d_t - theta_hat[k, t])^2]

    Args:
        theta_hat: shape (K, T)
        d:         shape (T,)
    Returns:
        v: shape (K,), each in (0, 1].
    """
    assert theta_hat.ndim == 2
    assert d.ndim == 1 and d.shape[0] == theta_hat.shape[1]
    diff_sq = (d[None, :] - theta_hat) ** 2
    factors = 1.0 - diff_sq
    return np.prod(factors, axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_core.py::test_compute_value_worked_example_columns -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vda/core.py tests/test_core.py
git commit -m "feat: add compute_value (Eq. 1 multiplicative value)"
```

---

## Task 9: Gauss–Seidel allocation solver — part 2: solve_allocation + worked example

**Files:**
- Modify: `vda/core.py` (add `solve_allocation`)
- Modify: `tests/test_core.py` (add worked-example test)

- [ ] **Step 1: Write the failing worked-example test**

Append to `tests/test_core.py`:

```python
from vda.core import solve_allocation


def test_solve_allocation_worked_example():
    """Section 10 of the note: d* should be approximately (0.248, 0.797, 0.320, 0.755)."""
    theta = np.array([
        [0.20, 0.85, 0.10, 0.75],
        [0.30, 0.60, 0.80, 0.70],
        [0.25, 0.90, 0.15, 0.80],
    ])
    d_star, weights, diag = solve_allocation(theta, eps=1e-6, max_sweeps=100, tol=1e-12)

    expected = np.array([0.248, 0.797, 0.320, 0.755])
    np.testing.assert_allclose(d_star, expected, atol=3e-3)

    # V must be non-decreasing.
    V = diag["V_trajectory"]
    for i in range(1, len(V)):
        assert V[i] >= V[i - 1] - 1e-12

    assert weights.shape == (3, 4)
    assert np.all(weights > 0)


def test_solve_allocation_v_monotonicity_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        K = int(rng.integers(2, 6))
        T = int(rng.integers(3, 21))
        theta = rng.uniform(0.05, 0.95, size=(K, T))
        _, _, diag = solve_allocation(theta, eps=1e-6, max_sweeps=100, tol=1e-12)
        V = diag["V_trajectory"]
        for i in range(1, len(V)):
            assert V[i] >= V[i - 1] - 1e-12


def test_solve_allocation_additive_limit():
    """When all reports are nearly identical, d* ≈ column mean."""
    rng = np.random.default_rng(42)
    base = rng.uniform(0.1, 0.9, size=(1, 6))
    theta = np.tile(base, (3, 1)) + rng.normal(0, 1e-4, size=(3, 6))
    theta = np.clip(theta, 1e-6, 1 - 1e-6)
    d_star, _, _ = solve_allocation(theta, eps=1e-6, max_sweeps=200, tol=1e-14)
    np.testing.assert_allclose(d_star, theta.mean(axis=0), atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core.py -v`
Expected: FAIL — `solve_allocation` not defined.

- [ ] **Step 3: Implement solve_allocation**

Append to `vda/core.py`:

```python
def solve_allocation(
    theta_hat: np.ndarray,
    *,
    eps: float = 1e-6,
    max_sweeps: int = 100,
    tol: float = 1e-10,
    d_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Algorithm 1: Gauss–Seidel solver for the multiplicative allocation.

    Args:
        theta_hat: (K, T) discriminator reports, clipped to [eps, 1-eps].
        eps:       clipping parameter.
        max_sweeps: maximum number of full sweeps over all t.
        tol:       convergence tolerance on ΔV (not ‖Δd‖).
        d_init:    optional warm-start allocation; default is the column mean.

    Returns:
        (d_star, weights, diagnostics) where:
          - d_star: (T,) final allocation
          - weights: (K, T) reputation weights w^t_k
          - diagnostics: {"V_trajectory": list[float], "sweeps": int, "converged": bool}
    """
    assert theta_hat.ndim == 2, "theta_hat must be (K, T)"
    K, T = theta_hat.shape

    if d_init is not None:
        d = np.asarray(d_init, dtype=np.float64).copy()
    else:
        d = theta_hat.mean(axis=0).astype(np.float64)
    d = np.clip(d, eps, 1.0 - eps)

    def total_V(d_vec: np.ndarray) -> float:
        return float(np.sum(compute_value(theta_hat, d_vec)))

    V_trajectory = [total_V(d)]
    converged = False
    sweeps_done = 0

    for sweep in range(1, max_sweeps + 1):
        for t in range(T):
            # w^t_k computed with the CURRENT (in-place) d values
            diff_sq = (d[None, :] - theta_hat) ** 2
            factors = 1.0 - diff_sq
            factors_except_t = np.concatenate([factors[:, :t], factors[:, t + 1 :]], axis=1)
            weights_t = np.prod(factors_except_t, axis=1)  # shape (K,)

            denom = weights_t.sum()
            if denom <= 0:
                continue
            new_dt = float(np.sum(theta_hat[:, t] * weights_t) / denom)
            d[t] = min(max(new_dt, eps), 1.0 - eps)

        V_new = total_V(d)
        V_old = V_trajectory[-1]
        assert V_new >= V_old - 1e-12, (
            f"V non-monotonic at sweep {sweep}: old={V_old}, new={V_new}. "
            "This indicates an implementation bug (Gauss-Seidel must be monotonic)."
        )
        V_trajectory.append(V_new)
        sweeps_done = sweep

        if V_new - V_old < tol:
            converged = True
            break

    # Final weights at every (k, t)
    diff_sq = (d[None, :] - theta_hat) ** 2
    factors = 1.0 - diff_sq
    weights = np.empty((K, T), dtype=np.float64)
    for t in range(T):
        fac_ex_t = np.concatenate([factors[:, :t], factors[:, t + 1 :]], axis=1)
        weights[:, t] = np.prod(fac_ex_t, axis=1)

    diag = {
        "V_trajectory": V_trajectory,
        "sweeps": sweeps_done,
        "converged": converged,
    }
    return d, weights, diag
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core.py -v`
Expected: PASS (all 4 tests in test_core.py).

- [ ] **Step 5: Commit**

```bash
git add vda/core.py tests/test_core.py
git commit -m "feat: add Gauss-Seidel allocation solver (Algorithm 1)"
```

---

## Task 10: Phase 2 stubs in core.py (locked for Stage 2)

**Files:**
- Modify: `vda/core.py` (add stub functions)
- Modify: `tests/test_core.py` (assert stubs raise NotImplementedError)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_core.py`:

```python
from vda.core import compute_vcg_payments, compute_gradient_fd, omd_update


def test_phase2_functions_are_stubs_until_stage2():
    theta = np.array([[0.2, 0.8], [0.3, 0.7]])
    d = np.array([0.25, 0.75])
    with pytest.raises(NotImplementedError):
        compute_vcg_payments(theta, d, solver_kwargs={})
    with pytest.raises(NotImplementedError):
        compute_gradient_fd(theta, solver_kwargs={})
    with pytest.raises(NotImplementedError):
        omd_update(theta, np.zeros_like(theta), eta=0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_core.py::test_phase2_functions_are_stubs_until_stage2 -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Add stubs to vda/core.py**

Append to `vda/core.py`:

```python
def compute_vcg_payments(
    theta_hat: np.ndarray,
    d_star: np.ndarray,
    *,
    solver_kwargs: Dict[str, Any],
) -> np.ndarray:
    """Algorithm 2 (Stage 2): K leave-one-out solves for VCG payments."""
    raise NotImplementedError("compute_vcg_payments is implemented in Stage 2 (OMD plan)")


def compute_gradient_fd(
    theta_hat: np.ndarray,
    *,
    delta: float = 1e-4,
    solver_kwargs: Dict[str, Any] = None,
) -> np.ndarray:
    """Algorithm 3 (Stage 2): finite-difference gradient of Π wrt theta_hat."""
    raise NotImplementedError("compute_gradient_fd is implemented in Stage 2 (OMD plan)")


def omd_update(
    theta_hat: np.ndarray,
    grad: np.ndarray,
    eta: float,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Algorithm 4 (Stage 2): exponentiated-gradient coordinate-wise update."""
    raise NotImplementedError("omd_update is implemented in Stage 2 (OMD plan)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_core.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vda/core.py tests/test_core.py
git commit -m "feat: add Phase 2 stubs (compute_vcg_payments, compute_gradient_fd, omd_update)"
```

---

## Task 11: Pipeline (vda/pipeline.py) — Stage 1 path only

**Files:**
- Create: `vda/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
import numpy as np
import pytest

from vda.pipeline import run_vda_from_reports, blame_attribution
from vda.types import BlameResult


def test_run_vda_from_reports_worked_example():
    """Section 10 worked example: d_final ≈ (0.248, 0.797, 0.320, 0.755),
    agent 2 gets all the blame (steps 2 and 4 are in the blame set)."""
    theta = np.array([
        [0.20, 0.85, 0.10, 0.75],
        [0.30, 0.60, 0.80, 0.70],
        [0.25, 0.90, 0.15, 0.80],
    ])
    agent_names_per_step = ["agent_1", "agent_2", "agent_1", "agent_2"]

    result = run_vda_from_reports(
        theta_hat=theta,
        agent_names_per_step=agent_names_per_step,
        c_t=0.5,
        eps=1e-6,
    )

    assert isinstance(result, BlameResult)
    np.testing.assert_allclose(result.theta_bar, [0.248, 0.797, 0.320, 0.755], atol=3e-3)
    assert result.blame_set == [1, 3]  # t=2 and t=4 (0-indexed: 1, 3)
    assert abs(result.agent_blame["agent_1"]) < 1e-9
    assert abs(result.agent_blame["agent_2"] - (0.797 + 0.755)) < 3e-3
    assert result.predicted_agent == "agent_2"
    assert result.predicted_step == 1  # argmax d_final is step t=2 in 1-indexed terms -> 1 in 0-indexed
    assert result.vcg_payments is None


def test_run_vda_additive_aggregator():
    """With aggregator='additive', d_final is the column mean, not Gauss-Seidel."""
    theta = np.array([
        [0.20, 0.85, 0.10, 0.75],
        [0.30, 0.60, 0.80, 0.70],
        [0.25, 0.90, 0.15, 0.80],
    ])
    agent_names_per_step = ["agent_1", "agent_2", "agent_1", "agent_2"]
    result = run_vda_from_reports(
        theta_hat=theta,
        agent_names_per_step=agent_names_per_step,
        c_t=0.5,
        aggregator="additive",
    )
    np.testing.assert_allclose(result.theta_bar, theta.mean(axis=0), atol=1e-12)


def test_blame_attribution_empty_set_is_deterministic():
    theta_bar = np.array([0.1, 0.2, 0.3])
    agents = ["a", "b", "c"]
    blame_set, agent_blame, predicted_agent, predicted_step = blame_attribution(
        theta_bar, agents, c_t=0.5
    )
    assert blame_set == []
    assert agent_blame == {"a": 0.0, "b": 0.0, "c": 0.0}
    assert predicted_step == 2
    assert predicted_agent in agents  # deterministic, even if all zero
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement vda/pipeline.py**

Create `vda/pipeline.py`:

```python
"""VDA pipeline orchestration (Algorithms 5-6)."""

from typing import List, Tuple, Dict
import numpy as np

from vda.core import solve_allocation
from vda.types import BlameResult


def blame_attribution(
    theta_bar: np.ndarray,
    agent_names_per_step: List[str],
    c_t: float,
) -> Tuple[List[int], Dict[str, float], str, int]:
    """Phase 4 of Algorithm 6.

    Args:
        theta_bar: (T,) final consensus
        agent_names_per_step: length T, agent active at each step
        c_t: fix-cost threshold

    Returns:
        (blame_set, agent_blame, predicted_agent, predicted_step)
    """
    T = len(agent_names_per_step)
    assert theta_bar.shape == (T,)

    blame_set = [t for t in range(T) if theta_bar[t] > c_t]

    # Preserve first-seen agent order so the "deterministic" choice is stable.
    agents_ordered: List[str] = []
    for a in agent_names_per_step:
        if a not in agents_ordered:
            agents_ordered.append(a)

    agent_blame: Dict[str, float] = {a: 0.0 for a in agents_ordered}
    for t in blame_set:
        agent_blame[agent_names_per_step[t]] += float(theta_bar[t])

    # Pick the agent with the largest blame; ties broken by first-seen order.
    predicted_agent = max(agents_ordered, key=lambda a: agent_blame[a])
    predicted_step = int(np.argmax(theta_bar))
    return blame_set, agent_blame, predicted_agent, predicted_step


def run_vda_from_reports(
    theta_hat: np.ndarray,
    agent_names_per_step: List[str],
    *,
    c_t: float = 0.5,
    eps: float = 1e-6,
    max_sweeps: int = 100,
    tol: float = 1e-10,
    aggregator: str = "multiplicative",
) -> BlameResult:
    """Stage 1b entry point: consume a saved theta_hat matrix and return BlameResult.

    aggregator:
        - "multiplicative": Gauss–Seidel solver (Algorithm 1)
        - "additive":       simple column mean (baseline for ablation)
    """
    theta_hat = np.clip(np.asarray(theta_hat, dtype=np.float64), eps, 1.0 - eps)
    K, T = theta_hat.shape
    assert len(agent_names_per_step) == T

    if aggregator == "multiplicative":
        d_final, weights, diag = solve_allocation(
            theta_hat, eps=eps, max_sweeps=max_sweeps, tol=tol,
        )
    elif aggregator == "additive":
        d_final = theta_hat.mean(axis=0)
        weights = np.ones((K, T), dtype=np.float64) / K
        diag = {"V_trajectory": [], "sweeps": 0, "converged": True}
    else:
        raise ValueError(f"Unknown aggregator: {aggregator}")

    blame_set, agent_blame, predicted_agent, predicted_step = blame_attribution(
        d_final, agent_names_per_step, c_t=c_t,
    )

    effective_variance = ((d_final[None, :] - theta_hat) ** 2).mean(axis=1).tolist()
    byzantine_flags = [bool(weights[k].mean() < 0.01) for k in range(K)]

    return BlameResult(
        theta_bar=d_final,
        blame_set=blame_set,
        agent_blame=agent_blame,
        predicted_agent=predicted_agent,
        predicted_step=predicted_step,
        vcg_payments=None,
        solver_diagnostics={
            "aggregator": aggregator,
            "V_trajectory": diag["V_trajectory"],
            "sweeps": diag["sweeps"],
            "converged": diag["converged"],
            "effective_variance": effective_variance,
            "byzantine_flags": byzantine_flags,
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add vda/pipeline.py tests/test_pipeline.py
git commit -m "feat: add vda.pipeline with multiplicative and additive aggregators"
```

---

## Task 12: Metrics (evaluation/metrics.py)

**Files:**
- Create: `evaluation/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
from evaluation.metrics import compute_metrics


def test_compute_metrics_basic():
    # Three traces, 2 agent-correct, 1 step-correct, 1 joint-correct.
    results = [
        {"predicted_agent": "a", "predicted_step": 2, "mistake_agent": "a", "mistake_step": 2},
        {"predicted_agent": "a", "predicted_step": 3, "mistake_agent": "a", "mistake_step": 1},
        {"predicted_agent": "b", "predicted_step": 0, "mistake_agent": "c", "mistake_step": 0},
    ]
    m = compute_metrics(results)
    assert abs(m["acc_agent"] - 2 / 3) < 1e-12
    assert abs(m["acc_step"] - 2 / 3) < 1e-12
    assert abs(m["acc_joint"] - 1 / 3) < 1e-12


def test_compute_metrics_empty():
    m = compute_metrics([])
    assert m == {"acc_agent": 0.0, "acc_step": 0.0, "acc_joint": 0.0, "n": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement evaluation/metrics.py**

Create `evaluation/metrics.py`:

```python
"""Metrics from Section 11.3 of the implementation note."""

from typing import List, Dict


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute Acc_agent, Acc_step, Acc_joint.

    Each entry must have keys: predicted_agent, predicted_step,
    mistake_agent, mistake_step.
    """
    n = len(results)
    if n == 0:
        return {"acc_agent": 0.0, "acc_step": 0.0, "acc_joint": 0.0, "n": 0}

    agent_hits = sum(1 for r in results if r["predicted_agent"] == r["mistake_agent"])
    step_hits = sum(1 for r in results if r["predicted_step"] == r["mistake_step"])
    joint_hits = sum(
        1 for r in results
        if r["predicted_agent"] == r["mistake_agent"]
        and r["predicted_step"] == r["mistake_step"]
    )
    return {
        "acc_agent": agent_hits / n,
        "acc_step": step_hits / n,
        "acc_joint": joint_hits / n,
        "n": n,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add Section 11.3 accuracy metrics"
```

---

## Task 13: Benchmark runner (scripts/run_who_and_when.py)

**Files:**
- Create: `scripts/run_who_and_when.py`

**Note:** This script is end-to-end glue that reads saved reports. No unit test — it's smoke-tested in Task 14.

- [ ] **Step 1: Implement run_who_and_when.py**

Create `scripts/run_who_and_when.py`:

```python
"""Stage 1b+: Run VDA on saved discriminator reports and compute metrics."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import VDAConfig
from datasets.who_and_when import load_who_and_when
from evaluation.metrics import compute_metrics
from vda.pipeline import run_vda_from_reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="Hand-Crafted")
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--aggregator", default="multiplicative",
                        choices=["multiplicative", "additive"])
    parser.add_argument("--output", default=None,
                        help="Defaults to results/<subset>-<aggregator>-<ts>.json")
    args = parser.parse_args()

    config = VDAConfig()
    reports_dir = Path(args.reports_dir) if args.reports_dir else ROOT / "data" / "reports" / args.subset
    if not reports_dir.exists():
        print(f"ERROR: {reports_dir} missing. Run scripts/generate_reports.py first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading trace metadata from HuggingFace subset '{args.subset}'...")
    traces = {t.trace_id: t for t in load_who_and_when(subset=args.subset)}

    report_files = sorted(reports_dir.glob("*.npz"))
    if args.limit is not None:
        report_files = report_files[: args.limit]
    print(f"Found {len(report_files)} saved reports.")

    per_trace: List[Dict] = []
    errors: List[Dict] = []

    for f in tqdm(report_files, desc=args.aggregator):
        trace_id = int(f.stem)
        if trace_id not in traces:
            errors.append({"trace_id": trace_id, "error": "not in dataset"})
            continue
        trace = traces[trace_id]
        data = np.load(f, allow_pickle=True)
        theta = data["theta_hat"]

        agent_names_per_step = [h["name"] for h in trace.history]
        try:
            result = run_vda_from_reports(
                theta_hat=theta,
                agent_names_per_step=agent_names_per_step,
                c_t=config.c_t,
                eps=config.eps,
                max_sweeps=config.L,
                tol=config.tau,
                aggregator=args.aggregator,
            )
        except Exception as e:
            errors.append({"trace_id": trace_id, "error": repr(e)})
            continue

        per_trace.append({
            "trace_id": trace_id,
            "predicted_agent": result.predicted_agent,
            "predicted_step": result.predicted_step,
            "mistake_agent": trace.mistake_agent,
            "mistake_step": trace.mistake_step,
            "theta_bar": result.theta_bar.tolist(),
            "blame_set": result.blame_set,
            "agent_blame": result.agent_blame,
            "diagnostics": {
                k: v for k, v in result.solver_diagnostics.items()
                if k in {"aggregator", "sweeps", "converged",
                         "effective_variance", "byzantine_flags"}
            },
        })

    metrics = compute_metrics(per_trace)
    print(f"\n=== {args.aggregator} on {args.subset} ===")
    print(f"n={metrics['n']}")
    print(f"Acc_agent = {metrics['acc_agent']:.4f}")
    print(f"Acc_step  = {metrics['acc_step']:.4f}")
    print(f"Acc_joint = {metrics['acc_joint']:.4f}")
    if errors:
        print(f"Errors: {len(errors)}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.output) if args.output else ROOT / "results" / f"{args.subset}-{args.aggregator}-{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "subset": args.subset,
                "aggregator": args.aggregator,
                "c_t": config.c_t,
                "eps": config.eps,
            },
            "metrics": metrics,
            "per_trace": per_trace,
            "errors": errors,
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Import smoke test**

Run: `python -c "import scripts.run_who_and_when"`
Expected: no ImportError.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_who_and_when.py
git commit -m "feat: add scripts/run_who_and_when.py benchmark runner"
```

---

## Task 14: Full Stage 1b benchmark run (manual)

**Files:** None — validation step.

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (test_config, test_types, test_prompt, test_discriminator, test_core, test_pipeline, test_metrics).

- [ ] **Step 2: Run multiplicative benchmark**

```bash
cd /Users/wanghd/Desktop/Research/Dai/vda_experiment
python scripts/run_who_and_when.py --subset Hand-Crafted --aggregator multiplicative
```

Expected: prints `n=58`, `Acc_agent`, `Acc_step`, `Acc_joint`. Baseline from the note for best single-model GPT-4o is 53.5% agent / 14.2% step; we should be in a comparable ballpark (our model is weaker but has K=3 and the mechanism).

- [ ] **Step 3: Run additive baseline**

```bash
python scripts/run_who_and_when.py --subset Hand-Crafted --aggregator additive
```

Expected: prints a second set of metrics. Compare to the multiplicative run to see the effect of the mechanism.

- [ ] **Step 4: Report results to user**

Share the two metric printouts. This is the end of Stage 1. **Stop here.** Stage 2 (VCG payments + OMD calibration) is a separate follow-on plan.

---

## Appendix: Quick-reference file responsibility map

| File | Responsibility |
|---|---|
| `config.py` | Hyperparameter dataclass matching Section 7 |
| `datasets/who_and_when.py` | **Reused unchanged** — HuggingFace loader + `trace_to_steps` |
| `vda/types.py` | Dataclasses: `TraceStep`, `Reports`, `BlameResult` |
| `vda/prompt.py` | `build_discriminator_prompt` (Section 6 template) |
| `vda/discriminator.py` | `OpenAIDiscriminator.query` + `extract_probability_a` |
| `vda/core.py` | `compute_value`, `solve_allocation`, Phase 2 stubs |
| `vda/pipeline.py` | `run_vda_from_reports`, `blame_attribution` |
| `evaluation/metrics.py` | `compute_metrics` (Acc_agent/step/joint) |
| `scripts/generate_reports.py` | Stage 1a: KT LLM calls → `.npz` per trace |
| `scripts/inspect_reports.py` | Stage 1a: sanity checks on saved reports |
| `scripts/run_who_and_when.py` | Stage 1b+: offline benchmark from saved reports |
