"""Compare top-k accuracy between legacy and hybrid reports on overlapping traces.

Usage:
  python scripts/compare_hybrid_vs_legacy.py \
      --legacy data/reports/Hand-Crafted \
      --hybrid data/reports/Hand-Crafted_hybrid
"""

import argparse
from pathlib import Path

import numpy as np


def load_reports(report_dir: Path):
    out = {}
    skipped = []
    for p in sorted(report_dir.glob("*.npz")):
        trace_id = int(p.stem)
        z = np.load(p, allow_pickle=True)
        raw_gt = z["mistake_step"].item() if hasattr(z["mistake_step"], "item") else z["mistake_step"]
        if raw_gt is None:
            skipped.append(trace_id)
            continue
        out[trace_id] = {
            "theta": z["theta_hat"],
            "model_ids": [str(m) for m in z["model_ids"]],
            "gt": int(raw_gt),
            "T": z["theta_hat"].shape[1],
        }
    if skipped:
        print(f"Skipped {len(skipped)} traces in {report_dir} with mistake_step=None: {skipped}")
    return out


def top_k_hit(theta_row: np.ndarray, gt: int, k: int) -> bool:
    if gt < 0 or gt >= len(theta_row):
        return False
    top_k = np.argsort(-theta_row)[:k]
    return gt in top_k.tolist()


def summarize(reports: dict, trace_ids: list, label: str):
    model_ids = reports[trace_ids[0]]["model_ids"]
    K = len(model_ids)
    rows = []
    for mi in range(K):
        row = {"model": model_ids[mi]}
        for k in (1, 3, 5):
            hits = sum(
                top_k_hit(reports[tid]["theta"][mi], reports[tid]["gt"], k)
                for tid in trace_ids
            )
            row[f"top{k}"] = f"{hits}/{len(trace_ids)} ({hits/len(trace_ids):.1%})"
        rows.append(row)

    # Ensemble mean
    ens_row = {"model": "ensemble_mean"}
    for k in (1, 3, 5):
        hits = 0
        for tid in trace_ids:
            theta_mean = reports[tid]["theta"].mean(axis=0)
            if top_k_hit(theta_mean, reports[tid]["gt"], k):
                hits += 1
        ens_row[f"top{k}"] = f"{hits}/{len(trace_ids)} ({hits/len(trace_ids):.1%})"
    rows.append(ens_row)

    print(f"\n=== {label} ===")
    print(f"{'model':<45} {'top1':<15} {'top3':<15} {'top5':<15}")
    for r in rows:
        print(f"{r['model']:<45} {r['top1']:<15} {r['top3']:<15} {r['top5']:<15}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--trace-ids", default=None,
                        help="Comma-separated trace IDs. Defaults to intersection of both dirs.")
    args = parser.parse_args()

    legacy = load_reports(Path(args.legacy))
    hybrid = load_reports(Path(args.hybrid))

    if args.trace_ids:
        tids = sorted(int(x) for x in args.trace_ids.split(","))
    else:
        tids = sorted(set(legacy) & set(hybrid))

    missing_legacy = [t for t in tids if t not in legacy]
    missing_hybrid = [t for t in tids if t not in hybrid]
    if missing_legacy:
        print(f"WARNING: missing in legacy: {missing_legacy}")
    if missing_hybrid:
        print(f"WARNING: missing in hybrid: {missing_hybrid}")
    tids = [t for t in tids if t in legacy and t in hybrid]

    print(f"Comparing on {len(tids)} traces: {tids}")
    print(f"\nTrace-level details (gt indices differ because action_only filter differs):")
    print(f"{'tid':<5} {'legacy T':<10} {'legacy gt':<10} {'hybrid T':<10} {'hybrid gt':<10}")
    for t in tids:
        print(f"{t:<5} {legacy[t]['T']:<10} {legacy[t]['gt']:<10} {hybrid[t]['T']:<10} {hybrid[t]['gt']:<10}")

    summarize(legacy, tids, "LEGACY (action_only=True, raw-content prompt)")
    summarize(hybrid, tids, "HYBRID (all history, structured prior + raw current)")


if __name__ == "__main__":
    main()
