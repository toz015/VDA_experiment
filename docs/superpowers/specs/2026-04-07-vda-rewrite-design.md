# VDA Rewrite — Design Spec

**Date:** 2026-04-07
**Scope:** Reimplementation of the Reputation-Weighted Multiplicative VCG Mechanism for failure attribution on the Who&When benchmark.
**Source of truth for the algorithm:** `implementation-note-2026-04-07.pdf` (Algorithms 1–6, Section 6 prompt, Section 7 hyperparameters, Section 11 protocol).

## Motivation

The previous implementation of this project is not trusted. We are rebuilding the pipeline from scratch, following the implementation note verbatim. The mathematical core (Gauss–Seidel allocation, VCG payments, OMD calibration) is rewritten; dataset loading and a few supporting files are kept.

## Staged Scope (Scope C)

The work ships in three stages, with an explicit data/algorithm decoupling:

- **Stage 1a — Data generation (Phase 1 only).** Build the LLM plumbing end-to-end: dataset loader → prompt builder → `OpenAIDiscriminator.query` → gather a `(K, T)` matrix `θ̂` for each trace → **save to disk** (one file per trace, plus a manifest). No allocation, no blame. The deliverable is a directory of verified probability matrices on the Hand-Crafted subset. Success means: the data-generation script runs end-to-end on all 58 traces, produces well-formed `θ̂` matrices, and the values pass sanity checks (no NaNs, no collapse to 0/1, distributions look reasonable). Once this is done, every downstream experiment reuses the saved `θ̂` — we don't spend LLM budget again unless the prompt changes.
- **Stage 1b — Algorithm (Phases 1b, 3, 4, R=0).** Consumes the saved `θ̂` matrices. Implements Gauss–Seidel allocation (Algorithm 1), blame attribution (Algorithm 6), and an additive-mean baseline for A/B comparison. No LLM calls; runs offline in seconds. Deliverable: Acc_agent / Acc_step / Acc_joint numbers on Hand-Crafted for both the additive baseline and the multiplicative mechanism.
- **Stage 2 — Full pipeline.** Adds Phase 2 — VCG payment computation (Algorithm 2), finite-difference gradient (Algorithm 3), OMD calibration (Algorithm 4). Still no new LLM calls; the saved `θ̂` from Stage 1a is the input. Delivers the calibration ablation and the VCG-based discriminator ranking.

**Why decouple.** LLM queries are the expensive part (~1,200 calls for Hand-Crafted, ~$ and ~10 min wall clock). The math is cheap and will be iterated many times during debugging and ablations. Saving `θ̂` to disk after Stage 1a means we can reload it instantly, rerun the algorithm, tweak hyperparameters, compare additive vs multiplicative, etc., without re-paying for LLM calls. It also means Stage 1b can be validated on the Section 10 worked example *and* the real saved data without any API key.

The module layout is fixed now so each stage is a purely additive change to the codebase.

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Discriminator extraction** | OpenAI logprobs API, `max_tokens=1`, `top_logprobs=20`, extract P("A") / (P("A") + P("B")) | Single clean call per (k,t); cheapest and lowest-noise option |
| **Discriminator ensemble** | Single model `gpt-4o-mini`, K=3 via temperatures {0.0, 0.7, 1.0} | Most efficient for first experiment; avoids cross-provider complexity |
| **Prompt strategy** | One independent LLM call per (k, t) with history[0..t] as context, ~500-token truncation on preceding steps, current step untruncated | Matches Algorithm 5 literally; produces independent `θ̂ᵏₜ` values suitable for the VCG math |
| **Code placement** | Replace in-place: wipe `vda/` and `scripts/`, keep `datasets/who_and_when.py`, `config.py`, `evaluation/`, `requirements.txt` | User judged the old `vda/` code untrustworthy; `datasets/who_and_when.py` is known-good |
| **Architecture** | Flat modules, numpy-first pure-function core, orchestration in a separate file | Enables unit-testing the math against Section 10 without any LLM mock; keeps phase boundaries explicit |

## Architecture

```
vda_experiment/
├── config.py                   # VDAConfig — updated to match Section 7 defaults
├── datasets/
│   └── who_and_when.py         # Reused as-is (load_who_and_when, trace_to_steps, WhoAndWhenTrace)
├── vda/
│   ├── __init__.py
│   ├── types.py                # TraceStep, Reports, BlameResult dataclasses
│   ├── core.py                 # Pure numpy: Algorithms 1-4
│   ├── prompt.py               # build_discriminator_prompt (Section 6 template)
│   ├── discriminator.py        # OpenAIDiscriminator, build_ensemble
│   └── pipeline.py             # run_vda (Algorithms 5-6)
├── evaluation/
│   ├── metrics.py              # Rewrite: Acc_agent, Acc_step, Acc_joint (Section 11.3)
│   └── runner.py               # Rewrite: progress bar, per-trace error capture, JSON output
├── scripts/
│   ├── generate_reports.py     # Stage 1a: LLM queries → save theta_hat per trace
│   ├── inspect_reports.py      # Stage 1a: sanity-check saved data, print stats/histograms
│   ├── run_single_trace.py     # Smoke test on one real trace
│   ├── run_who_and_when.py     # Stage 1b+: benchmark from saved reports
│   └── run_ablations.py        # Stage 2 deliverable; skeleton only earlier
└── tests/
    ├── test_core.py            # Math unit tests (no LLM)
    ├── test_discriminator.py   # Mocked OpenAI logprob extraction
    └── test_pipeline.py        # End-to-end with stub discriminator
```

**Key property:** `vda/core.py` takes `(K, T)` numpy arrays in and returns numpy arrays out. It has no knowledge of LLMs, traces, or agents. This is the algorithm from the note, nothing else. `vda/pipeline.py` is the only place that glues math to LLMs to blame attribution.

## Data Types (`vda/types.py`)

```python
@dataclass
class TraceStep:
    t: int                # 0-indexed step number
    agent_name: str
    action: str           # history[t]["content"] (untruncated)
    prior_context: str    # concatenated steps 0..t-1, each truncated to ~500 tokens
    task_description: str
    ground_truth: str

@dataclass
class Reports:
    theta_hat: np.ndarray     # shape (K, T), values in [eps, 1-eps]
    model_ids: list[str]      # length K, e.g. ["gpt-4o-mini@T=0.0", ...]

@dataclass
class BlameResult:
    theta_bar: np.ndarray              # shape (T,), final consensus d*
    blame_set: list[int]               # S* = {t : theta_bar[t] > c_t}
    agent_blame: dict[str, float]      # Π_agent_i
    predicted_agent: str               # î*
    predicted_step: int                # t̂*
    vcg_payments: np.ndarray | None    # shape (K,), None in Stage 1
    solver_diagnostics: dict           # V-trajectory, sweep count, effective variance, etc.
```

`TraceStep.prior_context` is pre-built by `datasets/who_and_when.py::trace_to_steps` (existing logic), so the discriminator stays dataset-agnostic.

## Core Math (`vda/core.py`)

Pure numpy functions. All signatures stable across Stage 1 and Stage 2; Stage 1 stubs the Phase 2 bodies.

```python
def solve_allocation(
    theta_hat: np.ndarray,       # (K, T)
    *,
    eps: float = 1e-6,
    max_sweeps: int = 100,
    tol: float = 1e-10,
    d_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Algorithm 1. Returns (d_star, weights, diagnostics).
    diagnostics = {'V_trajectory': [...], 'sweeps': int, 'converged': bool}.
    Raises if V is ever non-monotonic (> 1e-12 tolerance) — indicates a bug."""

def compute_value(theta_hat: np.ndarray, d: np.ndarray) -> np.ndarray:
    """v_k for each k, shape (K,). v_k = prod_t [1 - (d_t - theta_hat[k,t])^2]."""

def compute_vcg_payments(            # Stage 2 — stub in Stage 1
    theta_hat: np.ndarray,
    d_star: np.ndarray,
    *,
    solver_kwargs: dict,
) -> np.ndarray:
    """Algorithm 2. K leave-one-out solves warm-started from d_star.
    Returns Π, shape (K,), non-negative."""

def compute_gradient_fd(             # Stage 2 — stub in Stage 1
    theta_hat: np.ndarray,
    *,
    delta: float = 1e-4,
    solver_kwargs: dict,
) -> np.ndarray:
    """Algorithm 3. Finite-difference gradient ∂Π_k/∂θ̂^k_t for all (k,t)."""

def omd_update(                      # Stage 2 — stub in Stage 1
    theta_hat: np.ndarray,
    grad: np.ndarray,
    eta: float,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Algorithm 4. Coordinate-wise exponentiated gradient."""
```

**Design points:**
- `solve_allocation` uses in-place Gauss–Seidel updates (not Jacobi). Remark 4 in the note is explicit that Jacobi can oscillate.
- The V-monotonicity assertion fires immediately on violation; per Remark 4, a violation always means a bug.
- `solver_kwargs` is passed as a dict so VCG and gradient both re-enter `solve_allocation` with identical hyperparameters.

## Prompt (`vda/prompt.py`)

One pure function that returns the Section 6 template verbatim, filled in from a `TraceStep`. No dependencies.

## Discriminator (`vda/discriminator.py`)

```python
class OpenAIDiscriminator:
    def __init__(self, model: str, temperature: float, api_key: str | None = None):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)
        self.id = f"{model}@T={temperature}"

    def query(self, prompt: str) -> float:
        """Returns P('A'), clipped to [eps, 1-eps].
        Calls chat.completions.create with max_tokens=1, logprobs=True, top_logprobs=20.
        Scans the top-20 for tokens matching 'A' / 'B' after strip+upper,
        converts to linear probs, renormalizes p_A / (p_A + p_B).
        Fallback: if neither token appears in top-20, returns 0.5 and raises a diagnostic flag."""

def build_ensemble(config: VDAConfig) -> list[OpenAIDiscriminator]:
    """Returns K = len(config.temperatures) discriminators, all using config.openai_model."""
```

- Parallelism is deferred. First cut runs KT calls sequentially. If too slow on 58 traces (~1,200 calls), add `asyncio.gather` later.
- Fallback events are counted and surfaced in `BlameResult.solver_diagnostics` per trace.

## Pipeline (`vda/pipeline.py`)

One entry-point function, `run_vda(trace, discriminators, config, *, use_omd=False) -> BlameResult`, implementing Algorithms 5–6. The `use_omd=False` branch is Stage 1; `use_omd=True` gates the Phase 2 stub and is only functional in Stage 2.

Structure:
1. **Phase 1 (KT LLM calls).** Loop over (k, t), build prompt via `build_discriminator_prompt`, call `disc.query`, clip to `[eps, 1-eps]`.
2. **Phase 1b (Gauss–Seidel).** `solve_allocation(theta_raw, ...)`.
3. **Phase 1c (diagnostics).** Effective variance per discriminator, Byzantine flags (mean weight < 0.01).
4. **Phase 2 (OMD).** Stage 1: skipped. Stage 2: calls internal `run_omd_calibration(theta_raw, config)`.
5. **Phase 3 (final aggregation).** Stage 1: `d_final = d_star_0`. Stage 2: re-run `solve_allocation` on calibrated reports.
6. **Phase 4 (blame).** `blame_set = {t : d_final[t] > c_t}`. `agent_blame[a] = Σ_{t ∈ blame_set ∩ T_a} d_final[t]`. `predicted_agent = argmax(agent_blame)`. `predicted_step = argmax(d_final)`.

Edge cases:
- Empty `blame_set`: `agent_blame` is all zeros; `argmax` returns a deterministic choice; diagnostic logged.
- `predicted_step = argmax(d_final)` is always defined, independent of `blame_set`, per the note.

## Config Changes (`config.py`)

Update `VDAConfig` to match Section 7 of the note:

```python
K: int = 3
openai_model: str = "gpt-4o-mini"
temperatures: list[float] = [0.0, 0.7, 1.0]
R: int = 15                    # was 20
eta_0: float = 0.3             # was 0.5
eta_p: float = 0.7
L: int = 100                   # was 50
tau: float = 1e-10             # was 1e-8 — on ΔV, not ‖Δd‖
delta: float = 1e-4
eps: float = 1e-6
c_t: float = 0.5
prompt_max_tokens_per_prior_step: int = 500

def lr(self, r: int) -> float:
    """0-indexed: eta_0 / (r+1)^p, matches Algorithm 5 line 15."""
    return self.eta_0 / ((r + 1) ** self.eta_p)
```

The old 3-provider model list is removed; we're single-provider for the first run.

## Scripts

- **`scripts/generate_reports.py` (Stage 1a).** For each trace in a subset, run the KT discriminator queries and save a `θ̂` file. Output layout:
  ```
  data/reports/<subset>/<trace_id>.npz   # theta_hat: (K,T), model_ids: list, diagnostics: dict
  data/reports/<subset>/manifest.json    # K, T_per_trace, temperatures, model, total_calls, timestamp
  ```
  Flags: `--subset`, `--limit N`, `--trace-ids 0,1,2`, `--resume` (skip traces already saved). Writes incrementally so an interrupted run can be resumed.
- **`scripts/inspect_reports.py` (Stage 1a).** Loads saved `θ̂` files and prints: per-discriminator distribution (histogram or quantile summary), fraction of fallback events, per-trace `(K, T)` shape, mean/std per discriminator, disagreement rate between discriminators. Used to validate Stage 1a before moving to Stage 1b.
- **`scripts/run_single_trace.py`** — Smoke test. Runs `generate_reports` + `run_vda` end-to-end on one trace with a real API key. Used to validate plumbing before launching a full generation run.
- **`scripts/run_who_and_when.py` (Stage 1b+).** Benchmark from saved reports. Loads `data/reports/<subset>/` and runs `run_vda` offline (no LLM calls). Flags: `--subset`, `--limit N`, `--use-omd`, `--aggregator {multiplicative, additive}`. Writes `results/<subset>-<timestamp>.json`.
- **`scripts/run_ablations.py`** — Stage 2 deliverable. Skeleton only earlier.

## Evaluation

- **`evaluation/metrics.py`** — Rewrite. Implements Acc_agent, Acc_step, Acc_joint per Section 11.3. Signature: `compute_metrics(results: list[tuple[BlameResult, WhoAndWhenTrace]]) -> dict`.
- **`evaluation/runner.py`** — Rewrite. Thin loop with tqdm progress bar, per-trace try/except so one crash doesn't kill a 58-trace run, structured JSON output.

## Testing Strategy

Three layers, all runnable with `pytest` and no API keys.

### `tests/test_core.py` — math, no LLM
1. **Worked example reproduction (Section 10).** Hard-code the 3×4 reports from the note, assert `d*` matches `(0.248, 0.797, 0.320, 0.755)` to 3 decimals. **This is the single most important test.**
2. **V-monotonicity stress test.** 20 random (K, T) matrices, K ∈ {2,3,5}, T ∈ {3,10,20}. Assert `V_trajectory` non-decreasing every sweep.
3. **Additive limit.** Near-identical reports → `d*` within 1e-3 of column mean.
4. **Blame attribution on the worked example.** `agent_blame = {1: 0, 2: 1.552}`, `predicted_agent = 2`.
5. **VCG payments on the worked example** (Stage 2 only). `Π ≈ (0.0012, 0.0003, 0.0015)`.
6. **Edge cases.** `eps` boundary, T=1, K=2, all-zero, all-one reports.

### `tests/test_discriminator.py` — mocked OpenAI
1. Logprob extraction on a fixed mocked response.
2. Token-matching robustness: `"A"`, `" A"`, `"a"`, `" a"` all map to the "A" bucket.
3. Fallback: neither A nor B in top-20 → returns 0.5 and raises diagnostic flag.

### `tests/test_pipeline.py` — end-to-end with stub
1. Stub discriminator returns pre-canned Section 10 values. Synthetic `WhoAndWhenTrace` with T=4, M=2. `run_vda(use_omd=False)` → assert full `BlameResult` matches Section 10.
2. `use_omd=True` in Stage 1 raises `NotImplementedError` (locked so we don't silently skip OMD).

### Smoke test
`scripts/run_single_trace.py` hits the real OpenAI API on one trace. Not in CI. Run manually before launching the benchmark.

### Test-driven order
Write `test_core.py::test_worked_example` first, watch it fail, implement `solve_allocation` until it passes. That single test validates the entire mathematical core against an independently-derived expected output from the note.

## Success Criteria

**Stage 1a (data generation):**
- `tests/test_discriminator.py` passes (mocked logprob extraction).
- `scripts/run_single_trace.py` successfully queries the real OpenAI API on one trace and prints a well-formed `θ̂` matrix (no NaNs, values in [eps, 1-eps], correct shape `(K, T)`).
- `scripts/generate_reports.py --subset Hand-Crafted` completes on all 58 traces, writes `.npz` files + manifest, and is resumable.
- `scripts/inspect_reports.py` reports: no NaN values; fallback event rate < 5%; per-discriminator distributions have non-trivial spread (i.e., not all reports collapsed to 0 or 1); disagreement between discriminators is visible (so the reputation mechanism has something to work with).

**Stage 1b (algorithm on saved data):**
- `tests/test_core.py::test_worked_example` passes.
- All core unit tests pass.
- `run_who_and_when.py --subset Hand-Crafted --aggregator multiplicative` completes on all 58 saved-report files without crashes.
- Both additive and multiplicative Acc_agent / Acc_step / Acc_joint numbers are computed and saved to `results/`, enabling a direct comparison.

**Stage 2:**
- VCG payment and gradient functions implemented; worked-example VCG test passes.
- OMD calibration loop converges (`‖θ̂⁽ʳ⁺¹⁾ - θ̂⁽ʳ⁾‖∞ < 1e-5` before R=15).
- Ablation script produces the Section 11.5 comparison table (additive vs multiplicative; K ∈ {1,2,3,5}; R ∈ {0,5,10,15}).

## Out of Scope

- Multi-provider ensembles (Claude, Gemini). Single-provider first.
- Async/parallel LLM calls. Sequential first; add only if needed.
- JAX/PyTorch autodiff for the gradient. Finite differences per Algorithm 3 for now. The note flags autodiff as a future optimization (Remark 6).
- Algorithm-Generated subset (126 traces). Start with Hand-Crafted (58).
