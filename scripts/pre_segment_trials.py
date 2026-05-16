"""Pre-build the trial segmentation cache for all traces in a subset.

Uses gpt-4o-mini to segment each trace; results cached to data/trial_cache/.
Run this once before generate_trial_reports.py to save time.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_dotenv(p: Path):
    if not p.exists(): return
    for line in p.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(Path.home() / ".openai.env")
_load_dotenv(ROOT / ".env")

from vda_datasets.who_and_when import load_who_and_when
from vda.trial_segmenter import segment_trace


def main():
    subset = sys.argv[1] if len(sys.argv) > 1 else "Hand-Crafted"
    cache_dir = ROOT / "data" / "step_cache"

    traces = load_who_and_when(subset=subset, action_only=False)
    print(f"Pre-segmenting {len(traces)} traces from {subset}...")

    for i, tr in enumerate(traces):
        clsf = cache_dir / f"{subset}_{tr.trace_id}.json"
        if not clsf.exists():
            print(f"  [{i+1}/{len(traces)}] tid={tr.trace_id}: classifying...")
            tr.classify_steps(subset=subset)
        with open(clsf) as f:
            classified = json.load(f)
        seg = segment_trace(
            history=tr.history,
            classified=classified,
            trace_id=tr.trace_id,
            subset=subset,
        )
        n_trials = len(seg["trial_spans"])
        print(f"  [{i+1}/{len(traces)}] tid={tr.trace_id}: T={tr.T}, n_trials={n_trials}, "
              f"planning={seg['merged_idxs']}")


if __name__ == "__main__":
    main()
