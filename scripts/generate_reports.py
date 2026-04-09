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
