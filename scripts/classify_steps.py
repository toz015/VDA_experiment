"""Classify all steps in Who&When traces into (action_type, state) triples.

Runs LLM-based classification on each history entry and caches results
to data/step_cache/ so they only need to be computed once.

Usage:
    python scripts/classify_steps.py --subset Hand-Crafted
    python scripts/classify_steps.py --subset Algorithm-Generated --limit 5
    python scripts/classify_steps.py --subset Hand-Crafted --model gpt-4o-mini
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vda_datasets.who_and_when import load_who_and_when


def main():
    parser = argparse.ArgumentParser(description="Classify Who&When trace steps")
    parser.add_argument("--subset", default="Hand-Crafted",
                        choices=["Hand-Crafted", "Algorithm-Generated"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N traces")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model for classification (default: gpt-4o-mini)")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache directory (default: data/step_cache/)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else ROOT / "data" / "step_cache"

    print(f"Loading subset '{args.subset}' (action_only=False to get all traces)...")
    traces = load_who_and_when(subset=args.subset, action_only=False)
    print(f"Loaded {len(traces)} traces.")

    if args.limit:
        traces = traces[:args.limit]
        print(f"Processing first {args.limit} traces.")

    total_steps = 0
    for trace in tqdm(traces, desc="Classifying traces"):
        trace.classify_steps(model=args.model, subset=args.subset, cache_dir=cache_dir)
        total_steps += len(trace.classified_steps)

        # Show first trace as a sample
        if trace.trace_id == traces[0].trace_id:
            print(f"\n--- Sample: trace {trace.trace_id} ({len(trace.classified_steps)} steps) ---")
            for cls in trace.classified_steps[:8]:
                print(f"  [{cls['agent'][:30]}] {cls['action_type']} — {cls['state'][:60]}")
            if len(trace.classified_steps) > 8:
                print(f"  ... ({len(trace.classified_steps) - 8} more steps)")
            print()

    print(f"\nDone. Classified {total_steps} steps across {len(traces)} traces.")
    print(f"Cache saved to: {cache_dir}/")


if __name__ == "__main__":
    main()
