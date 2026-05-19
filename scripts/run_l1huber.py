"""scripts/run_l1huber.py — L1-Huber VCG pipeline runner on Method-2 reports.

Mirrors scripts/run_vcg.py exactly so that the output summary.json can be
directly compared to the squared-kernel baseline (just diff the Acc_step,
Acc_agent, Acc_joint fields). The ONLY differences are:

  - Uses vcg.pipeline_l1huber.run_pipeline_l1huber instead of run_pipeline.
  - Extra CLI flag --c for the L1-Huber kernel bandwidth (default 0.05).
  - Default output subdir is data/reports/<subset>_l1huber/<tag> instead of
    data/reports/<subset>_vcg/<tag>.

For each .npz in data/reports/<subset>_hybrid_v2/:
  1. Load theta_hat (K, T), mistake_step, mistake_agent.
  2. Load step agents from data/step_cache/<subset>_<id>.json.
  3. Run vcg.run_pipeline_l1huber (R configurable, default 0 == skip OMD).
  4. Record predicted_step, predicted_agent, blame_set size, theta_bar argmax.

Outputs:
  - data/reports/<output_subdir>/per_trace.csv  (one row per trace)
  - data/reports/<output_subdir>/summary.json   (Acc_agent / Acc_step / Acc_joint)
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vcg.pipeline_l1huber import run_pipeline_l1huber  # noqa: E402


def trace_id_from_filename(path: Path) -> int:
    return int(path.stem)


_HF_CACHE: dict = {}


def _load_hf_subset(subset: str):
    if subset not in _HF_CACHE:
        from datasets import load_dataset
        _HF_CACHE[subset] = load_dataset("Kevin355/Who_and_When", subset)["train"]
    return _HF_CACHE[subset]


def load_step_agents(subset: str, trace_id: int) -> List[str]:
    cache_path = ROOT / "data" / "step_cache" / f"{subset}_{trace_id}.json"
    with open(cache_path) as f:
        entries = json.load(f)

    if subset == "Algorithm-Generated":
        ds = _load_hf_subset(subset)
        history = ds[trace_id]["history"]
        return [history[e["original_index"]].get("name", e["agent"]) for e in entries]

    return [e["agent"] for e in entries]


def evaluate(
    subset: str,
    report_dir: Path,
    output_dir: Path,
    c: float,
    R: int,
    c_t: float,
    eta_0: float,
    eta_p: float,
    delta: float,
    L: int,
    tau: float,
    eps: float,
    omd_direction: str,
    report_eps: float = None,
    limit: int = 0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(report_dir.glob("*.npz"), key=trace_id_from_filename)
    if limit > 0:
        files = files[:limit]

    rows = []
    n_step_hit = 0
    n_agent_hit = 0
    n_joint_hit = 0
    skipped = 0

    for f in files:
        trace_id = trace_id_from_filename(f)
        data = np.load(f, allow_pickle=True)
        theta = np.asarray(data["theta_hat"], dtype=np.float64)
        ms_raw = (
            data["mistake_step"].item()
            if hasattr(data["mistake_step"], "item")
            else data["mistake_step"]
        )
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
            print(
                f"trace {trace_id}: agents length {len(step_agents)} != T {theta.shape[1]} — skipping",
                flush=True,
            )
            skipped += 1
            continue

        result = run_pipeline_l1huber(
            theta,
            step_agents=step_agents,
            c=c,
            R=R,
            eta_0=eta_0,
            eta_p=eta_p,
            delta=delta,
            L=L,
            tau=tau,
            eps=eps,
            report_eps=report_eps,
            c_t=c_t,
            omd_direction=omd_direction,
        )
        b = result.blame
        if len(result.omd_history) > 1:
            theta_drift = float(
                np.max(np.abs(result.omd_history[-1] - result.omd_history[0]))
            )
        else:
            theta_drift = 0.0
        init_argmax = int(np.argmax(result.initial_alloc.d))
        final_argmax = int(np.argmax(result.theta_bar))
        omd_changed_pred = int(init_argmax != final_argmax)
        step_hit = int(b.predicted_step == gt_step)
        agent_hit = int(b.predicted_agent == gt_agent) if b.predicted_agent else 0
        joint_hit = step_hit & agent_hit

        n_step_hit += step_hit
        n_agent_hit += agent_hit
        n_joint_hit += joint_hit

        rows.append({
            "trace_id": trace_id,
            "T": theta.shape[1],
            "K": theta.shape[0],
            "gt_step": gt_step,
            "gt_agent": gt_agent,
            "pred_step": b.predicted_step,
            "pred_agent": b.predicted_agent or "",
            "blame_set_size": len(b.blame_set),
            "theta_bar_max": float(np.max(result.theta_bar)),
            "step_hit": step_hit,
            "agent_hit": agent_hit,
            "joint_hit": joint_hit,
            "omd_converged": int(result.omd_converged),
            "init_V_final": float(result.initial_alloc.V_history[-1]),
            "final_V_final": float(result.final_alloc.V_history[-1]),
            "theta_drift_linf": theta_drift,
            "init_argmax": init_argmax,
            "final_argmax": final_argmax,
            "omd_changed_pred": omd_changed_pred,
        })

    per_trace_csv = output_dir / "per_trace.csv"
    with open(per_trace_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    summary = {
        "subset": subset,
        "report_dir": str(report_dir),
        "n_traces": n,
        "skipped": skipped,
        "kernel": "L1-Huber",
        "c": c,
        "R": R,
        "c_t": c_t,
        "eta_0": eta_0,
        "eta_p": eta_p,
        "eps": eps,
        "report_eps": report_eps if report_eps is not None else eps,
        "omd_direction": omd_direction,
        "Acc_step": n_step_hit / n if n else 0.0,
        "Acc_agent": n_agent_hit / n if n else 0.0,
        "Acc_joint": n_joint_hit / n if n else 0.0,
    }
    with open(output_dir / "summary.json", "w") as fout:
        json.dump(summary, fout, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subset", choices=["Hand-Crafted", "Algorithm-Generated"], required=True
    )
    ap.add_argument(
        "--report-dir", default=None,
        help="Override M2 report dir. Default: data/reports/<subset>_hybrid_v2",
    )
    ap.add_argument(
        "--output-dir", default=None,
        help="Where to write per_trace.csv and summary.json. "
             "Default: data/reports/<subset>_l1huber/<tag>",
    )
    ap.add_argument("--tag", default="R0", help="output subdir tag (default R0)")
    ap.add_argument(
        "--c", type=float, default=0.05, help="L1-Huber kernel bandwidth (default 0.05)"
    )
    ap.add_argument("--R", type=int, default=0, help="OMD rounds (0 skips Phase 2)")
    ap.add_argument("--c-t", type=float, default=0.5)
    ap.add_argument("--eta-0", type=float, default=0.3)
    ap.add_argument("--eta-p", type=float, default=0.7)
    ap.add_argument("--delta", type=float, default=1e-4)
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--tau", type=float, default=1e-10)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument(
        "--report-eps", type=float, default=None,
        help="external clip on raw reports (default = --eps). "
             "Use 0.10 to match Tong's baseline.",
    )
    ap.add_argument("--omd-direction", choices=["ascent", "descent"], default="ascent")
    ap.add_argument("--limit", type=int, default=0, help="evaluate first N traces only")
    args = ap.parse_args()

    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else ROOT / "data" / "reports" / f"{args.subset}_hybrid_v2"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "data" / "reports" / f"{args.subset}_l1huber" / args.tag
    )

    summary = evaluate(
        subset=args.subset,
        report_dir=report_dir,
        output_dir=output_dir,
        c=args.c,
        R=args.R,
        c_t=args.c_t,
        eta_0=args.eta_0,
        eta_p=args.eta_p,
        delta=args.delta,
        L=args.L,
        tau=args.tau,
        eps=args.eps,
        report_eps=args.report_eps,
        omd_direction=args.omd_direction,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()