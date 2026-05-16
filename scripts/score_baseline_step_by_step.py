"""Aggregate Acc_step from baseline_who_step_by_step JSON outputs.

Reads data/baselines/who_step_by_step/<subset>/*.json (one file per LLM),
prints a comparison table including:
  - per-LLM Who&When step_by_step Acc_step
  - 'majority Yes' aggregator across the 3 LLMs (Yes from any 1 → first idx that
    saw Yes from at least M of K wins) for M=1, M=2, M=3
  - existing Method 2 numbers (loaded from raw .npz argmax)
  - existing VCG R5_eps10 numbers
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_baseline(subset: str) -> dict:
    """Returns {model_safe_name: [trace_results...]}"""
    out = {}
    base = ROOT / "data" / "baselines" / "who_step_by_step" / subset
    if not base.exists():
        return out
    for p in sorted(base.glob("*.json")):
        out[p.stem] = json.load(open(p))
    return out


def acc_step(rows):
    n = len(rows)
    hits = sum(1 for r in rows if r["predicted_step"] == r["gt_step"])
    return hits, n, hits / n if n else 0.0


def first_yes_majority(per_llm_rows, M_required: int):
    """Across K LLMs' per-step Yes/No, the predicted_step is the smallest idx s
    such that >= M_required of the K LLMs flagged step s as their first-Yes (or
    earlier with no Yes). Equivalent: walk steps in order; at each step count how
    many LLMs have *any* Yes whose first-Yes idx <= s; pick the smallest s where
    that count >= M_required.

    For each LLM, predicted_step is its first-Yes idx (or +inf if no Yes).
    Joint majority predicted_step = the M-th smallest first-Yes across LLMs.
    """
    # Group by trace_id
    by_tid = defaultdict(dict)
    for llm, rows in per_llm_rows.items():
        for r in rows:
            by_tid[r["trace_id"]][llm] = r

    results = []
    for tid, per_llm in by_tid.items():
        if len(per_llm) < M_required:
            continue
        # gt is consistent across LLMs for the same trace
        gt_step = next(iter(per_llm.values()))["gt_step"]
        gt_agent = next(iter(per_llm.values()))["gt_agent"]
        first_yes_idxs = []
        for r in per_llm.values():
            ps = r["predicted_step"]
            first_yes_idxs.append(ps if ps >= 0 else float("inf"))
        first_yes_idxs.sort()
        # M-th smallest (1-indexed)
        mth = first_yes_idxs[M_required - 1]
        pred = int(mth) if mth != float("inf") else -1
        results.append({"trace_id": tid, "gt_step": gt_step, "gt_agent": gt_agent,
                        "predicted_step": pred})
    return results


def raw_ensemble_acc(subset):
    """Re-compute the raw .npz ensemble argmax + per-disc step accuracy."""
    rdir = ROOT / "data" / "reports" / f"{subset}_hybrid_v2"
    files = sorted(rdir.glob("*.npz"), key=lambda p: int(p.stem))
    K = None
    per_disc = None
    ens = 0
    n = 0
    mids = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        ms = z["mistake_step"].item() if hasattr(z["mistake_step"], "item") else z["mistake_step"]
        if ms is None:
            continue
        gt = int(ms)
        theta = np.asarray(z["theta_hat"], dtype=np.float64)
        if K is None:
            K = theta.shape[0]
            per_disc = np.zeros(K, dtype=int)
            mids = [str(m) for m in z["model_ids"]]
        n += 1
        for k in range(K):
            if int(np.argmax(theta[k])) == gt:
                per_disc[k] += 1
        if int(np.argmax(theta.mean(axis=0))) == gt:
            ens += 1
    return mids, per_disc, ens, n


def vcg_best_acc(subset):
    """Read pre-computed VCG R5_eps10 step accuracy from summary.json."""
    p = ROOT / "data" / "reports" / f"{subset}_vcg" / "R5_eps10" / "summary.json"
    if not p.exists():
        return None
    return json.load(open(p))["Acc_step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["Hand-Crafted", "Algorithm-Generated"], required=True)
    args = ap.parse_args()

    print(f"\n=== {args.subset} ===\n")

    # Method 2 raw .npz numbers
    mids, per_disc, ens, n_raw = raw_ensemble_acc(args.subset)
    print(f"Method 2 (our prompt + soft P(A) + argmax over T):")
    if mids:
        for k, m in enumerate(mids):
            print(f"  disc[{k}] {m:<55s}  step={per_disc[k]:>3d}/{n_raw} = {per_disc[k]/n_raw:.3f}")
        print(f"  ensemble-mean argmax{'':<41s}  step={ens:>3d}/{n_raw} = {ens/n_raw:.3f}")

    vcg = vcg_best_acc(args.subset)
    if vcg is not None:
        print(f"  VCG R5_eps10 (best){'':<42s}  step={vcg:.3f}")

    # Baseline rows
    baselines = load_baseline(args.subset)
    print(f"\nWho&When step_by_step (their prompt, first-Yes, single LLM):")
    if not baselines:
        print(f"  no baseline data found at data/baselines/who_step_by_step/{args.subset}/")
        return
    for name, rows in baselines.items():
        h, n, a = acc_step(rows)
        print(f"  {name:<55s}  step={h:>3d}/{n} = {a:.3f}")

    # Majority aggregators across the 3 LLMs
    if len(baselines) >= 2:
        for M in (1, 2, len(baselines)):
            agg = first_yes_majority(baselines, M)
            h = sum(1 for r in agg if r["predicted_step"] == r["gt_step"])
            n = len(agg)
            print(f"  majority(M>={M}/{len(baselines)}) first-Yes{'':<29s}  step={h:>3d}/{n} = {h/n if n else 0:.3f}")


if __name__ == "__main__":
    main()
