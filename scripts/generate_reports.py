"""Stage 1a: Generate discriminator probability matrices for every trace in a subset.

For each trace, runs KT LLM calls and saves a .npz file with:
  - theta_hat: (K, T) numpy array, clipped to [eps, 1-eps]
  - model_ids: list of K strings
  - fallback_counts: (K,) int array of per-discriminator fallback events

Usage:
  # Default K=3 cross-provider ensemble (gemini-2.5-flash, gemini-2.5-flash-lite, gpt-4o-mini)
  python scripts/generate_reports.py --subset Hand-Crafted --limit 2

  # Custom ensemble from JSON file
  python scripts/generate_reports.py --subset Hand-Crafted --config configs/ensemble_vertex.json

  # Legacy OpenAI single-model mode
  python scripts/generate_reports.py --subset Hand-Crafted --limit 2 --legacy-openai
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running as `python scripts/generate_reports.py` from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a file into os.environ if not already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv(Path.home() / ".openai.env")
_load_dotenv(ROOT / ".env")

from config import VDAConfig
from vda_datasets.who_and_when import (
    load_who_and_when,
    trace_to_steps,
    trace_to_full_history_steps,
    WhoAndWhenTrace,
)
from vda.discriminator import build_ensemble
from vda.prompt import build_discriminator_prompt, build_json_discriminator_prompt

# Default K=3 ensemble — the only Gemini variants accessible on the UCLA
# llm-applications-490420 project (2.5-flash / 2.5-flash-lite) plus gpt-4o-mini
# for cross-provider diversity.
DEFAULT_ENSEMBLE = [
    {"provider": "vertex", "model": "gemini-2.5-flash", "temperature": 0.0, "thinking_budget": 0},
    {"provider": "vertex", "model": "gemini-2.5-flash-lite", "temperature": 0.0, "thinking_budget": 0},
    {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
]


def generate_for_trace(trace: WhoAndWhenTrace, discriminators, config: VDAConfig,
                       subset: str = "", prompt_mode: str = "logprob",
                       skip_classify: bool = False, method4: bool = False):
    """Run KT queries for one trace. Returns theta_hat (K,T) and fallback counts (K,).

    prompt_mode:
        "logprob" — single-token A/B prompt (for OpenAI / Vertex native logprob backends)
        "json"    — JSON-output prompt (for Vertex MaaS / Vertex JSON backends)
    skip_classify: if True, do not call classify_steps (used when caller already
                   filtered to action roles only — Method 1 legacy).
    method4: if True, score every step in trace.history (T = T_full) using
             action-role priors only. Bypasses classify_steps and trace_to_steps.
    """
    if method4:
        steps = trace_to_full_history_steps(trace)
        K, T = len(discriminators), len(steps)
    else:
        if not skip_classify:
            trace.classify_steps(subset=subset)
        steps = trace_to_steps(trace)
        K, T = len(discriminators), trace.T_action
    theta = np.empty((K, T), dtype=np.float64)

    prompt_fn = build_json_discriminator_prompt if prompt_mode == "json" else build_discriminator_prompt

    # Parallelize the K discriminators per step (different endpoints, independent rate budgets).
    with ThreadPoolExecutor(max_workers=K) as pool:
        for t, step in enumerate(steps):
            prompt = prompt_fn(step)
            futures = {pool.submit(d.query, prompt): k for k, d in enumerate(discriminators)}
            for fut in as_completed(futures):
                k = futures[fut]
                theta[k, t] = fut.result()

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
                        help="Use legacy OpenAI single-model mode instead of the default ensemble")
    parser.add_argument("--config", default=None,
                        help="Path to JSON file with ensemble spec (overrides DEFAULT_ENSEMBLE). "
                             "Top-level may be either a list of discriminator specs "
                             "or an object with a 'discriminators' key.")
    parser.add_argument("--project", default="llm-applications-490420",
                        help="GCP project ID for Vertex AI discriminators that don't specify one")
    parser.add_argument("--location", default="us-central1",
                        help="GCP region for Vertex AI discriminators that don't specify one")
    parser.add_argument("--prompt-mode", default="logprob", choices=["logprob", "json"],
                        help="logprob (single-token A/B for OpenAI/Vertex native) or "
                             "json (for Vertex MaaS / JSON-output backends)")
    parser.add_argument("--action-only", action="store_true",
                        help="Filter to action-role steps (Method 1 / legacy per-message). "
                             "When set, skips LLM step classification.")
    parser.add_argument("--method4", action="store_true",
                        help="Method 4: score every step in trace.history (full T) using "
                             "action-role priors only (legacy raw-truncated format). "
                             "mistake_step is saved in full-history index. "
                             "Mutually exclusive with --action-only.")
    args = parser.parse_args()
    if args.method4 and args.action_only:
        parser.error("--method4 and --action-only are mutually exclusive")

    # Configure ensemble
    if args.legacy_openai:
        config = VDAConfig()
    else:
        if args.config:
            with open(args.config) as f:
                raw = json.load(f)
            specs = raw["discriminators"] if isinstance(raw, dict) else raw
            if not isinstance(specs, list) or not specs:
                raise ValueError(f"--config {args.config} must contain a non-empty list of discriminator specs")
        else:
            specs = DEFAULT_ENSEMBLE
        discs = []
        for spec in specs:
            merged = dict(spec)
            if merged.get("provider") == "vertex":
                merged.setdefault("project", args.project)
                merged.setdefault("location", args.location)
            discs.append(merged)
        config = VDAConfig(discriminators=discs)

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "data" / "reports" / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Method 4 needs the full trace set, so it always loads with action_only=False
    load_action_only = args.action_only and not args.method4
    print(f"Loading subset '{args.subset}' (action_only={load_action_only}, method4={args.method4})...")
    traces = load_who_and_when(subset=args.subset, action_only=load_action_only)
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

        theta, fb = generate_for_trace(
            trace, discriminators, config, subset=args.subset,
            prompt_mode=args.prompt_mode, skip_classify=args.action_only or args.method4,
            method4=args.method4,
        )

        # Method 4 saves mistake_step in full-history index; legacy paths save the
        # action-only-remapped index.
        saved_mistake_step = trace.mistake_step if args.method4 else trace.mistake_step_action

        np.savez(
            out_path,
            theta_hat=theta,
            model_ids=np.array([d.id for d in discriminators]),
            fallback_counts=fb,
            mistake_agent=trace.mistake_agent,
            mistake_step=saved_mistake_step,
        )

        manifest["traces"] = [e for e in manifest.get("traces", []) if e["trace_id"] != trace.trace_id]
        manifest["traces"].append({
            "trace_id": trace.trace_id,
            "T": trace.T,
            "T_action": trace.T_action,
            "agents": trace.agents,
            "mistake_agent": trace.mistake_agent,
            "mistake_step": saved_mistake_step,
            "fallback_counts": fb.tolist(),
        })
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"Done. Reports in {out_dir}")


if __name__ == "__main__":
    main()
