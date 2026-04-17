"""Stage 1a: Generate discriminator probability matrices for every trace in a subset.

For each trace, runs KT LLM calls and saves a .npz file with:
  - theta_hat: (K, T) numpy array, clipped to [eps, 1-eps]
  - model_ids: list of K strings
  - fallback_counts: (K,) int array of per-discriminator fallback events

Usage:
  # Vertex AI ensemble (gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash)
  python scripts/generate_reports.py --subset Hand-Crafted --limit 2

  # Legacy OpenAI mode
  python scripts/generate_reports.py --subset Hand-Crafted --limit 2 --legacy-openai
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
from vda_datasets.who_and_when import load_who_and_when, trace_to_steps, WhoAndWhenTrace
from vda.discriminator import build_ensemble
from vda.prompt import build_discriminator_prompt

# Default Vertex AI ensemble: 3 diverse Gemini models
VERTEX_ENSEMBLE = [
    {"provider": "vertex", "model": "gemini-2.0-flash", "temperature": 0.0},
    {"provider": "vertex", "model": "gemini-1.5-pro", "temperature": 0.0},
    {"provider": "vertex", "model": "gemini-1.5-flash", "temperature": 0.0},
]


def generate_for_trace(trace: WhoAndWhenTrace, discriminators, config: VDAConfig,
                       subset: str = ""):
    """Run KT queries for one trace. Returns theta_hat (K,T_action) and fallback counts (K,)."""
    # Use classified steps if cache is available
    trace.classify_steps(subset=subset)

    steps = trace_to_steps(trace)
    K, T = len(discriminators), trace.T_action
    theta = np.empty((K, T), dtype=np.float64)

    for k, disc in enumerate(discriminators):
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
    parser.add_argument("--legacy-openai", action="store_true",
                        help="Use legacy OpenAI single-model mode instead of Vertex AI")
    parser.add_argument("--project", default="llm-applications-490420",
                        help="GCP project ID for Vertex AI")
    parser.add_argument("--location", default="us-central1",
                        help="GCP region for Vertex AI")
    args = parser.parse_args()

    # Configure ensemble
    if args.legacy_openai:
        config = VDAConfig()
    else:
        discs = []
        for spec in VERTEX_ENSEMBLE:
            discs.append({**spec, "project": args.project, "location": args.location})
        config = VDAConfig(discriminators=discs)

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "data" / "reports" / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading subset '{args.subset}' (action_only=False)...")
    traces = load_who_and_when(subset=args.subset, action_only=False)
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
        "discriminators": [d.id for d in discriminators],
        "K": len(discriminators),
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

        theta, fb = generate_for_trace(trace, discriminators, config, subset=args.subset)

        np.savez(
            out_path,
            theta_hat=theta,
            model_ids=np.array([d.id for d in discriminators]),
            fallback_counts=fb,
            mistake_agent=trace.mistake_agent,
            mistake_step=trace.mistake_step_action,  # index in action-only steps
        )

        manifest["traces"] = [e for e in manifest.get("traces", []) if e["trace_id"] != trace.trace_id]
        manifest["traces"].append({
            "trace_id": trace.trace_id,
            "T": trace.T,
            "T_action": trace.T_action,
            "agents": trace.agents,
            "mistake_agent": trace.mistake_agent,
            "mistake_step": trace.mistake_step_action,  # action-only index
            "fallback_counts": fb.tolist(),
        })
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"Done. Reports in {out_dir}")


if __name__ == "__main__":
    main()
