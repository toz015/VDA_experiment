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
