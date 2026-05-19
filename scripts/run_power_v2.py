"""scripts/run_power_v2.py — Power-VCG pipeline with selectable OMD objective.

Identical to scripts/run_power.py except:
  - Adds --omd-objective {payment, entropy, margin}.
    'payment' recovers the original payment-ascent OMD (Tong's).
    'entropy' uses concentration-ascent (descent on allocation entropy).
    'margin'  uses top1-top2 gap ascent.

Outputs:
  - data/reports/<subset>_power_v2/<tag>/per_trace.csv
  - data/reports/<subset>_power_v2/<tag>/summary.json
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vcg.allocation_power import PowerAllocationResult, solve_allocation_power
from vcg.blame import BlameResult, attribute_blame
from vcg.gradient_concentration import compute_inference_gradient_matrix
from vcg.gradient_power import compute_power_gradient_matrix
from vcg.omd import lr_schedule, omd_update
from vcg.payment_power import PaymentResult, compute_power_payments


@dataclass
class PipelineResult:
    theta_hat_raw: np.ndarray
    theta_hat_final: np.ndarray
    initial_alloc: PowerAllocationResult
    final_alloc: PowerAllocationResult
    theta_bar: np.ndarray
    omd_history: List[np.ndarray] = field(default_factory=list)
    payment_history: List[PaymentResult] = field(default_factory=list)
    blame: Optional[BlameResult] = None
    omd_converged: bool = False


def run_pipeline_v2(
    theta_hat_raw,
    *,
    step_agents=None,
    c=0.05,
    p=1.0,
    R=0,
    eta_0=0.3,
    eta_p=0.7,
    omd_tol=1e-5,
    delta=1e-4,
    L=100,
    tau=1e-10,
    eps=1e-6,
    report_eps=None,
    c_t=0.5,
    omd_direction="ascent",
    omd_objective="payment",
):
    if omd_direction not in {"ascent", "descent"}:
        raise ValueError(omd_direction)
    if omd_objective not in {"payment", "entropy", "margin"}:
        raise ValueError(omd_objective)
    if report_eps is None:
        report_eps = eps

    theta = np.clip(np.asarray(theta_hat_raw, dtype=np.float64),
                    report_eps, 1.0 - report_eps)
    K, T = theta.shape

    init_alloc = solve_allocation_power(theta, c=c, p=p, L=L, tau=tau, eps=eps)
    theta_curr = theta.copy()
    omd_history = [theta_curr.copy()]
    payment_history = []
    omd_converged = False

    for r in range(R):
        eta = lr_schedule(r, eta_0=eta_0, p=eta_p)
        # Always record payments for diagnostic comparability.
        alloc_r = solve_allocation_power(
            theta_curr, c=c, p=p, L=L, tau=tau, eps=eps, d_init=init_alloc.d
        )
        pay_r = compute_power_payments(
            theta_curr, alloc_r, c=c, p=p, L=L, tau=tau, eps=eps
        )
        payment_history.append(pay_r)

        if omd_objective == "payment":
            g = compute_power_gradient_matrix(
                theta_curr, c=c, p=p, delta=delta, L=L, tau=tau, eps=eps
            )
        else:
            g = compute_inference_gradient_matrix(
                theta_curr, objective=omd_objective,
                c=c, p=p, delta=delta, L=L, tau=tau, eps=eps,
            )
        if omd_direction == "descent":
            g = -g

        theta_next = omd_update(theta_curr, g, eta=eta, eps=eps)
        omd_history.append(theta_next.copy())
        if np.max(np.abs(theta_next - theta_curr)) < omd_tol:
            theta_curr = theta_next
            omd_converged = True
            break
        theta_curr = theta_next

    final_alloc = solve_allocation_power(
        theta_curr, c=c, p=p, L=L, tau=tau, eps=eps, d_init=init_alloc.d
    )
    blame = attribute_blame(final_alloc.d, step_agents, c_t=c_t) if step_agents else None

    return PipelineResult(
        theta_hat_raw=theta,
        theta_hat_final=theta_curr,
        initial_alloc=init_alloc,
        final_alloc=final_alloc,
        theta_bar=final_alloc.d,
        omd_history=omd_history,
        payment_history=payment_history,
        blame=blame,
        omd_converged=omd_converged,
    )


# ---- HF + agents (copied from run_power.py) ----

PROB_KEYS    = ["theta_hat", "probs", "prob", "p", "scores", "theta", "raw"]
GT_STEP_KEYS = ["mistake_step", "gt_step", "true_step", "label_step"]


def _first_present(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None


def _safe_int(npz, key):
    if not key:
        return None
    try:
        val = np.asarray(npz[key]).item()
    except (ValueError, AttributeError):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


_HF_CACHE = {}


def _load_hf_subset(subset):
    if subset not in _HF_CACHE:
        from datasets import load_dataset
        _HF_CACHE[subset] = load_dataset("Kevin355/Who_and_When", subset)["train"]
    return _HF_CACHE[subset]


def load_step_agents(subset, trace_id):
    cache_path = ROOT / "data" / "step_cache" / f"{subset}_{trace_id}.json"
    with open(cache_path) as f:
        entries = json.load(f)
    if subset == "Algorithm-Generated":
        ds = _load_hf_subset(subset)
        history = ds[trace_id]["history"]
        return [history[e["original_index"]].get("name", e["agent"]) for e in entries]
    return [e["agent"] for e in entries]


def evaluate(subset, report_dir, output_dir, c, p, R, c_t, eta_0, eta_p,
             delta, L, tau, eps, omd_direction, omd_objective,
             report_eps=None, limit=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(report_dir.glob("*.npz"), key=lambda x: int(x.stem))
    if limit > 0:
        files = files[:limit]

    rows = []
    n_step_hit = n_agent_hit = n_joint_hit = skipped = 0
    for f in files:
        trace_id = int(f.stem)
        data = np.load(f, allow_pickle=True)
        theta = np.asarray(data["theta_hat"], dtype=np.float64)
        ms_raw = (data["mistake_step"].item()
                  if hasattr(data["mistake_step"], "item")
                  else data["mistake_step"])
        if ms_raw is None:
            skipped += 1
            continue
        gt_step = int(ms_raw)
        gt_agent = str(data["mistake_agent"])
        try:
            step_agents = load_step_agents(subset, trace_id)
        except FileNotFoundError:
            skipped += 1
            continue
        if len(step_agents) != theta.shape[1]:
            skipped += 1
            continue

        result = run_pipeline_v2(
            theta, step_agents=step_agents,
            c=c, p=p, R=R, eta_0=eta_0, eta_p=eta_p,
            delta=delta, L=L, tau=tau, eps=eps,
            report_eps=report_eps, c_t=c_t,
            omd_direction=omd_direction, omd_objective=omd_objective,
        )
        b = result.blame
        step_hit = int(b.predicted_step == gt_step)
        agent_hit = int(b.predicted_agent == gt_agent) if b.predicted_agent else 0
        joint_hit = step_hit & agent_hit
        n_step_hit += step_hit
        n_agent_hit += agent_hit
        n_joint_hit += joint_hit
        rows.append({
            "trace_id": trace_id, "T": theta.shape[1], "K": theta.shape[0],
            "gt_step": gt_step, "gt_agent": gt_agent,
            "pred_step": b.predicted_step,
            "pred_agent": b.predicted_agent or "",
            "step_hit": step_hit, "agent_hit": agent_hit, "joint_hit": joint_hit,
            "omd_converged": int(result.omd_converged),
            "init_argmax": int(np.argmax(result.initial_alloc.d)),
            "final_argmax": int(np.argmax(result.theta_bar)),
        })

    with open(output_dir / "per_trace.csv", "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    summary = {
        "subset": subset, "n_traces": n, "skipped": skipped,
        "kernel": f"Power(p={p}, c={c})",
        "c": c, "p": p, "R": R, "c_t": c_t,
        "eta_0": eta_0, "eta_p": eta_p, "eps": eps,
        "report_eps": report_eps if report_eps is not None else eps,
        "omd_direction": omd_direction,
        "omd_objective": omd_objective,
        "Acc_step": n_step_hit / n if n else 0.0,
        "Acc_agent": n_agent_hit / n if n else 0.0,
        "Acc_joint": n_joint_hit / n if n else 0.0,
    }
    with open(output_dir / "summary.json", "w") as fout:
        json.dump(summary, fout, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["Hand-Crafted", "Algorithm-Generated"], required=True)
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--tag", default="R0")
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=0.05)
    ap.add_argument("--R", type=int, default=0)
    ap.add_argument("--c-t", type=float, default=0.5)
    ap.add_argument("--eta-0", type=float, default=0.3)
    ap.add_argument("--eta-p", type=float, default=0.7)
    ap.add_argument("--delta", type=float, default=1e-4)
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--tau", type=float, default=1e-10)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--report-eps", type=float, default=None)
    ap.add_argument("--omd-direction", choices=["ascent", "descent"], default="ascent")
    ap.add_argument("--omd-objective", choices=["payment", "entropy", "margin"],
                    default="payment",
                    help="OMD objective: payment (Tong's original), "
                         "entropy (concentration-ascent), or margin (top1-top2 ascent)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    report_dir = (Path(args.report_dir) if args.report_dir
                  else ROOT / "data" / "reports" / f"{args.subset}_hybrid_v2")
    output_dir = (Path(args.output_dir) if args.output_dir
                  else ROOT / "data" / "reports" / f"{args.subset}_power_v2" / args.tag)

    summary = evaluate(
        subset=args.subset, report_dir=report_dir, output_dir=output_dir,
        c=args.c, p=args.p, R=args.R, c_t=args.c_t,
        eta_0=args.eta_0, eta_p=args.eta_p, delta=args.delta,
        L=args.L, tau=args.tau, eps=args.eps,
        omd_direction=args.omd_direction,
        omd_objective=args.omd_objective,
        report_eps=args.report_eps,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()