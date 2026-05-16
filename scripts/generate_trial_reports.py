"""Stage A: trial-level discriminator scoring (DoVer-style).

For each trace:
  1. Segment into trials via vda.trial_segmenter (LLM, cached).
  2. For each trial, build a trial-level prompt and query each discriminator.
  3. Save (K, n_trials) theta_hat plus trial spans and gt_trial.

Usage:
  python scripts/generate_trial_reports.py \
      --subset Hand-Crafted \
      --config configs/ensemble_v2_json.json \
      --output-dir data/reports/Hand-Crafted_dover_v2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv(Path.home() / ".openai.env")
_load_dotenv(ROOT / ".env")

from config import VDAConfig
from vda_datasets.who_and_when import load_who_and_when
from vda.discriminator import build_ensemble
from vda.prompt import build_trial_discriminator_prompt
from vda.trial_segmenter import segment_trace, find_trial


def trial_messages_from_trace(history, classified, span):
    s, e = span
    by_idx = {c["original_index"]: c for c in classified}
    out = []
    for i in range(s, e):
        cls = by_idx.get(i, {})
        h = history[i]
        out.append({
            "original_index": i,
            "role": h.get("role", "?"),
            "action_type": cls.get("action_type", ""),
            "state": cls.get("state", ""),
            "content": h.get("content", "") or "",
        })
    return out


def summarize_prior_trials(prior_spans, history, classified):
    """One-line summary per prior trial."""
    if not prior_spans:
        return ""
    by_idx = {c["original_index"]: c for c in classified}
    lines = []
    for i, (s, e) in enumerate(prior_spans):
        first = history[s]
        cls0 = by_idx.get(s, {})
        last_idx = e - 1
        cls_last = by_idx.get(last_idx, {})
        lines.append(
            f"Trial {i}: msgs [{s},{e}) — opens with {first.get('role','?')} "
            f"({cls0.get('action_type','?')}: {cls0.get('state','')[:60]}); "
            f"ends at msg {last_idx} ({cls_last.get('action_type','?')})"
        )
    return "\n".join(lines)


def run_trace(trace, classified, segmentation, discriminators, config):
    spans = [tuple(s) for s in segmentation["trial_spans"]]
    n_trials = len(spans)
    K = len(discriminators)
    theta = np.empty((K, n_trials), dtype=np.float64)

    with ThreadPoolExecutor(max_workers=K) as pool:
        for trial_i, span in enumerate(spans):
            msgs = trial_messages_from_trace(trace.history, classified, span)
            prior_summary = summarize_prior_trials(spans[:trial_i], trace.history, classified)
            prompt = build_trial_discriminator_prompt(
                task_description=trace.question,
                ground_truth=trace.ground_truth,
                trial_index=trial_i,
                n_trials=n_trials,
                trial_messages=msgs,
                prior_trials_summary=prior_summary,
            )
            futures = {pool.submit(d.query, prompt): k for k, d in enumerate(discriminators)}
            for fut in as_completed(futures):
                k = futures[fut]
                theta[k, trial_i] = fut.result()

    theta = np.clip(theta, config.eps, 1.0 - config.eps)
    fb = np.array([d.fallback_count for d in discriminators], dtype=np.int64)

    gt_trial = find_trial(spans, trace.mistake_step)
    return theta, fb, spans, gt_trial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="Hand-Crafted",
                    choices=["Hand-Crafted", "Algorithm-Generated"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--trace-ids", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--segmenter-model", default="gpt-4o-mini")
    args = ap.parse_args()

    with open(args.config) as f:
        raw = json.load(f)
    specs = raw["discriminators"] if isinstance(raw, dict) else raw
    cfg = VDAConfig(discriminators=specs)
    discs = build_ensemble(cfg)
    print(f"Built {len(discs)} discriminators: {[d.id for d in discs]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.subset}...")
    traces = load_who_and_when(subset=args.subset, action_only=False)
    print(f"Loaded {len(traces)} traces.")

    if args.trace_ids:
        wanted = {int(x) for x in args.trace_ids.split(",")}
        traces = [t for t in traces if t.trace_id in wanted]
    elif args.limit:
        traces = traces[: args.limit]

    cache_dir = ROOT / "data" / "step_cache"
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "subset": args.subset,
        "discriminators": [d.id for d in discs],
        "K": len(discs),
        "eps": cfg.eps,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "traces": [],
    }
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    existing = {e["trace_id"] for e in manifest.get("traces", [])}

    for tr in tqdm(traces, desc="trial-traces"):
        out_path = out_dir / f"{tr.trace_id}.npz"
        if args.resume and out_path.exists() and tr.trace_id in existing:
            continue

        # Load classification cache
        clsf = cache_dir / f"{args.subset}_{tr.trace_id}.json"
        if not clsf.exists():
            tr.classify_steps(subset=args.subset)
            with open(clsf) as f:
                classified = json.load(f)
        else:
            with open(clsf) as f:
                classified = json.load(f)

        seg = segment_trace(
            history=tr.history,
            classified=classified,
            trace_id=tr.trace_id,
            subset=args.subset,
            model=args.segmenter_model,
        )

        theta, fb, spans, gt_trial = run_trace(tr, classified, seg, discs, cfg)

        np.savez(
            out_path,
            theta_hat=theta,
            model_ids=np.array([d.id for d in discs]),
            fallback_counts=fb,
            mistake_agent=tr.mistake_agent,
            mistake_step=tr.mistake_step,
            gt_trial=-1 if gt_trial is None else gt_trial,
            trial_spans=np.array(spans, dtype=np.int64),
        )

        manifest["traces"] = [e for e in manifest.get("traces", []) if e["trace_id"] != tr.trace_id]
        manifest["traces"].append({
            "trace_id": tr.trace_id,
            "T": tr.T,
            "n_trials": len(spans),
            "trial_spans": [list(s) for s in spans],
            "gt_trial": gt_trial,
            "mistake_step": tr.mistake_step,
            "fallback_counts": fb.tolist(),
        })
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"Done. Reports in {out_dir}")


if __name__ == "__main__":
    main()
