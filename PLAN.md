# VDA Experiment Plan

## 1. Algorithm Overview

We implement the **VCG-based Discriminator Attribution (VDA)** mechanism from
"Reputation-Weighted VCG Mechanism for Multi-Agent Failure Attribution via
Multiplicative Discriminator Scoring."

### Core Idea
Given a failed multi-agent trace of T steps, K heterogeneous LLM discriminators
evaluate each step.  Their reports are aggregated through a **multiplicative**
value function that couples evaluations across steps via reputation weights.
The mechanism:

1. **Phase 1 — Initial Evaluation** (K x T LLM calls): Each discriminator D_k
   scores every step t, producing theta_hat_t^k in (0,1).
2. **Phase 2 — OMD Calibration** (pure CPU): Online Mirror Descent adjusts
   reports using the VCG payment gradient, which encodes reputational feedback.
3. **Phase 3 — Final Aggregation**: Reputation-weighted average produces
   aggregated fix necessity theta_bar_t for each step.
4. **Phase 4 — Agent Blame Attribution**: Blame assigned to each agent based on
   the aggregated scores of their steps.

### Key Equations
- Multiplicative value: v_k(d, theta^k) = prod_t [1 - (d_t - theta_t^k)^2]
- Reputation weight: w_k^t(d) = prod_{t' != t} [1 - (d_{t'} - theta_hat_{t'}^k)^2]
- Allocation (fixed-point): d_t* = sum_k theta_hat_t^k * w_k^t(d*) / sum_k w_k^t(d*)
- VCG payment: Pi_k = sum_{j!=k} v_j(d*, theta_hat^j) - sum_{j!=k} v_j(d*_{-k}, theta_hat^j)
- OMD update (Bernoulli entropy): theta_hat_t^{k,(r+1)} = sigmoid(logit(theta_hat_t^{k,(r)}) + eta_r * g_t^{k,(r)})

---

## 2. Datasets

### 2.1 Who&When (Baseline — Single Error)
- **Source**: HuggingFace `Kevin355/Who_and_When`
- **Size**: 184 failure traces (126 Algorithm-Generated + 58 Hand-Crafted)
- **Structure**: Each trace has `history` (list of agent messages), one
  `mistake_agent`, one `mistake_step`
- **VDA mapping**:
  - T = len(history), typically 5-10
  - i_t = history[t]["name"] (active agent at step t)
  - s_t = history[0:t] (prefix context)
  - a_t = history[t]["content"] (action)
  - W = 0 (all traces are failures)
  - Ground truth: single mistake_step, single mistake_agent
- **Limitation**: Only ONE error per trace. Can only evaluate top-1 accuracy.
- **Metrics**: Agent accuracy, Step accuracy, Joint accuracy, Spearman rho

### 2.2 Math-Shepherd (Primary — Multi Error)
- **Source**: HuggingFace `peiyi9979/Math-Shepherd`
- **Size**: ~445K rows
- **Structure**: Each row has a math problem + step-by-step solution. Every step
  is labeled "+" (correct) or "-" (incorrect). Multiple steps can be "-".
- **VDA mapping**:
  - T = number of solution steps (typically 3-15)
  - M = 1 agent (the model), but we treat each step as an independent action
  - theta_t ground truth: "-" -> 1.0, "+" -> 0.0
  - Multiple theta_t > 0 per trace = multi-error
- **Why it fits**: Directly tests whether VDA can identify ALL erroneous steps,
  not just the first one. Large scale enables statistical power.
- **Metrics**: Per-step precision, recall, F1 at threshold c_t; AUC-ROC of
  theta_bar_t vs ground truth; rank correlation

### 2.3 TravelPlanner (Secondary — Multi Constraint Violations)
- **Source**: HuggingFace `osunlp/TravelPlanner`
- **Size**: 1,225 planning tasks
- **Structure**: Each plan must satisfy multiple constraints. Evaluation checks
  each constraint independently. Plans routinely violate multiple constraints.
- **VDA mapping**:
  - T = number of plan components / constraints checked
  - Each constraint violation = theta_t > 0
  - Typically 5-15 constraints per plan
- **Why it fits**: Different domain (planning vs reasoning), multi-error by
  nature, shows generality of the approach.
- **Metrics**: Same as Math-Shepherd

---

## 3. Code Structure

```
vda_experiment/
  PLAN.md                    # This file
  requirements.txt           # Dependencies
  config.py                  # Hyperparameters and defaults
  vda/
    __init__.py
    core.py                  # Algorithms 1-4: allocation, VCG, gradient, pipeline
    discriminator.py         # LLM discriminator interface + prompt templates
    omd.py                   # Online Mirror Descent with Bernoulli entropy
  datasets/
    __init__.py
    who_and_when.py          # Who&When loader + VDA mapping
    math_shepherd.py         # Math-Shepherd loader + VDA mapping
    travel_planner.py        # TravelPlanner loader + VDA mapping
  evaluation/
    __init__.py
    metrics.py               # All evaluation metrics
    runner.py                # End-to-end experiment runner
  scripts/
    run_who_and_when.py      # Run VDA on Who&When
    run_math_shepherd.py     # Run VDA on Math-Shepherd
    run_travel_planner.py    # Run VDA on TravelPlanner
    run_ablations.py         # Ablation studies
```

---

## 4. Implementation Details (from Section 8)

### Algorithm 1: Fixed-Point Solver for Allocation
- Input: reports theta_hat^k for k=1..K; max iterations L=50; tolerance tau=1e-8
- Initialize: d_t^(0) = (1/K) * sum_k theta_hat_t^k (simple mean)
- Iterate: compute w_k^t, then d_t^(l+1) = sum_k theta_hat_t^k * w_k^t / sum_k w_k^t
- Converge when ||d^(l+1) - d^(l)||_inf < tau
- Typically converges in 5-10 iterations

### Algorithm 2: VCG Payment Computation
- For each k: solve allocation without D_k's reports, compute payment difference
- Requires K+1 calls to SolveAllocation
- Total: O(K^2 * T * L)

### Algorithm 3: Gradient via Finite Differences
- For each (k, t): perturb theta_hat_t^k by +/- delta, re-solve allocation,
  recompute VCG payment, take centered difference
- delta = 1e-4, epsilon (clipping) = 1e-6
- Full gradient: O(K^2 * T^2 * L)

### Algorithm 4: Complete VDA Pipeline
- Phase 1: Query discriminators (K*T LLM calls), clip to [epsilon, 1-epsilon]
- Phase 2: R rounds of OMD. Each round: solve allocation, compute VCG payments,
  compute gradients, update via Bernoulli-entropy OMD. Early stop if
  ||theta^(r+1) - theta^(r)||_inf < 1e-5
- Phase 3: Final allocation -> theta_bar_t
- Phase 4: Agent blame via additive welfare model

### Hyperparameters (Section 8.7)
| Parameter        | Symbol | Default    |
|-----------------|--------|------------|
| Discriminators   | K      | 3          |
| OMD iterations   | R      | 20         |
| LR schedule      | eta_r  | 0.5/r^0.7 |
| Fixed-point iters| L      | 50         |
| Fixed-point tol  | tau    | 1e-8       |
| Finite-diff step | delta  | 1e-4       |
| Clipping         | eps    | 1e-6       |
| Fix threshold    | c_t    | 0.5        |

---

## 5. Evaluation Metrics

### Single-Error Datasets (Who&When)
- **Agent accuracy**: fraction where argmax_i Pi_i^agent = mistake_agent
- **Step accuracy**: fraction where argmax_{t in S*} theta_bar_t = mistake_step
- **Joint accuracy**: both correct
- **Spearman rho**: rank correlation of Pi_i^agent vs binary blame indicator

### Multi-Error Datasets (Math-Shepherd, TravelPlanner)
- **Per-step AUC-ROC**: theta_bar_t as score vs binary ground truth
- **Precision@k**: among top-k steps by theta_bar_t, fraction that are true errors
- **Recall@k**: among true errors, fraction in top-k
- **F1 at threshold c_t**: classify step as error if theta_bar_t > c_t
- **Exact match**: fraction of traces where the error SET is exactly recovered
- **Step-level accuracy**: binary classification accuracy per step

---

## 6. Ablation Studies

1. **Additive vs multiplicative**: Compare v_k = prod[1-(d_t-theta)^2]
   (multiplicative) vs v_k = -sum(d_t-theta)^2 (additive, = simple mean)
2. **Number of discriminators**: K=1 (single LLM), K=2, K=3, K=5
3. **OMD iterations**: R in {0, 5, 10, 20} — R=0 is raw reports, no calibration
4. **With/without sequential dependency resolution** (Algorithm 5)
5. **Discriminator quality scores**: Do higher Pi_k^total correlate with better
   individual accuracy?

---

## 7. Discriminator Choices

For K=3, use maximally diverse models:
- GPT-4o (OpenAI)
- Claude-3.5-Sonnet (Anthropic)
- Gemini-1.5-Pro (Google)

Fallback for cost-sensitive runs: use smaller models (GPT-4o-mini, Claude Haiku,
Gemini Flash) or a mock discriminator for testing.

---

## 8. Execution Order

1. Implement core algorithms (Alg 1-4) with unit tests on Section 9 worked example
2. Implement dataset loaders (Who&When first, then Math-Shepherd)
3. Implement discriminator interface with mock + real LLM backends
4. Run Who&When experiment, compare against 53.5% agent / 14.2% step baselines
5. Run Math-Shepherd experiment, evaluate multi-error metrics
6. Run ablations
7. (Optional) TravelPlanner experiment
