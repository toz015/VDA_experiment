"""Faithful re-implementation of Who&When `step_by_step` (utils.py:111-184).

Reproduces the paper's per-step iteration with a single LLM and first-Yes-wins
selection rule, but routes through our v2 LLM endpoints (Gemini-3-flash-preview,
Qwen3-next-80b, GPT-OSS-120b) so the ablation against our own Method 2 controls
for LLM identity.

For each trace:
  - iterate `history` in order (NO action-only filtering — matches the paper)
  - send the per-step prompt asking 'is this most recent step an error? (Yes/No)'
  - first response that starts with '1. yes' wins; record (predicted_step, predicted_agent)
  - if no Yes by end of trace: predicted_step = -1

Output (one JSON file per (subset, model)):
  data/baselines/who_step_by_step/<subset>/<model>.json
    [{trace_id, gt_step, gt_agent, predicted_step, predicted_agent,
      n_queries, raw_responses}]

Score with `scripts/score_baseline_step_by_step.py` (separate file).
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Free-form chat backends (reuse auth from vda/discriminator.py).
# ---------------------------------------------------------------------------

class ChatModel:
    id: str
    def chat(self, prompt: str, max_tokens: int = 1024) -> str:
        raise NotImplementedError


class VertexJSONChat(ChatModel):
    """Gemini via google-genai. Free-form text output (NOT JSON mode)."""

    def __init__(self, model: str, project: str, location: str = "global",
                 temperature: float = 0.2, thinking_budget: Optional[int] = 0,
                 timeout_ms: int = 60000, max_retries: int = 8):
        self.id = f"vertex_json/{model}@T={temperature}"
        self.model_name = model
        self.temperature = temperature
        self.max_retries = max_retries

        from google import genai
        from google.genai import types as genai_types
        self._types = genai_types
        self._client = genai.Client(
            vertexai=True, project=project, location=location,
            http_options=genai_types.HttpOptions(timeout=timeout_ms),
        )
        self._tb = thinking_budget

    def chat(self, prompt: str, max_tokens: int = 1024) -> str:
        cfg_kwargs = dict(temperature=self.temperature, max_output_tokens=max_tokens)
        if self._tb is not None:
            cfg_kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_budget=self._tb)
        cfg = self._types.GenerateContentConfig(**cfg_kwargs)

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_name, contents=prompt, config=cfg,
                )
                return resp.text or ""
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                rate = "429" in msg or "resource_exhausted" in msg or ("rate" in msg and "limit" in msg)
                wait = min(60, 8 * (2 ** min(attempt, 3))) if rate else (1.5 ** attempt)
                time.sleep(wait)
        raise last_exc


class VertexMaaSChat(ChatModel):
    """Qwen / GPT-OSS via Vertex MaaS OpenAI-compat endpoint. Free-form text output."""

    def __init__(self, model: str, project: str, location: str = "global",
                 temperature: float = 0.2, max_retries: int = 10):
        self.id = f"vertex_maas/{model}@T={temperature}"
        self.model = model
        self.project = project
        self.location = location
        self.temperature = temperature
        self.max_retries = max_retries

        from google.auth import default as auth_default
        from google.auth.transport.requests import Request
        from openai import OpenAI
        self._auth_default = auth_default
        self._auth_request = Request
        self._OpenAI = OpenAI
        self._creds, _ = auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not self._creds.valid:
            self._creds.refresh(self._auth_request())
        host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
        self._base_url = f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"
        self._client = OpenAI(api_key=self._creds.token, base_url=self._base_url)

    def _rebuild(self):
        if not self._creds.valid:
            self._creds.refresh(self._auth_request())
        self._client = self._OpenAI(api_key=self._creds.token, base_url=self._base_url)

    def chat(self, prompt: str, max_tokens: int = 1024) -> str:
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    max_completion_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                rate = "429" in msg or "resource_exhausted" in msg or ("rate" in msg and "limit" in msg)
                if "auth" in msg or "401" in msg or "expired" in msg:
                    try: self._rebuild()
                    except Exception: pass
                wait = min(60, 8 * (2 ** min(attempt, 3))) if rate else (1.5 ** attempt)
                time.sleep(wait)
        raise last_exc


def build_chat(spec: Dict) -> ChatModel:
    p = spec["provider"]
    if p == "vertex_json":
        return VertexJSONChat(model=spec["model"], project=spec["project"],
                              location=spec.get("location", "global"),
                              temperature=spec.get("temperature", 0.2),
                              thinking_budget=spec.get("thinking_budget", 0))
    if p == "vertex_maas":
        return VertexMaaSChat(model=spec["model"], project=spec["project"],
                              location=spec.get("location", "global"),
                              temperature=spec.get("temperature", 0.2))
    raise ValueError(f"Unsupported provider for chat baseline: {p}")


# ---------------------------------------------------------------------------
# Faithful step_by_step prompt — verbatim from Who&When utils.py:141-150.
# ---------------------------------------------------------------------------

def build_step_prompt(problem: str, ground_truth: str,
                      conversation_history: str, idx: int, agent_name: str) -> str:
    return (
        "You are an AI assistant tasked with evaluating the correctness of each step "
        "in an ongoing multi-agent conversation aimed at solving a real-world problem. "
        f"The problem being addressed is: {problem}. "
        f"The Answer for the problem is: {ground_truth}\n"
        f"Here is the conversation history up to the current step:\n{conversation_history}\n"
        f"The most recent step ({idx}) was by '{agent_name}'.\n"
        "Your task is to determine whether this most recent agent's action (Step "
        f"{idx}) contains an error that could hinder the problem-solving process or "
        "lead to an incorrect solution. "
        "Please respond with 'Yes' or 'No' and provide a clear explanation for your "
        "judgment. "
        "Note: Please avoid being overly critical in your evaluation. Focus on errors "
        "that clearly derail the process."
        "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
    )


def step_by_step_one_trace(chat: ChatModel, trace_row: Dict, is_handcrafted: bool,
                            max_tokens: int = 512) -> Dict:
    """Run faithful step_by_step on one trace. Returns dict with prediction + log."""
    history = trace_row["history"]
    problem = trace_row.get("question", "")
    ground_truth = trace_row.get("groundtruth", trace_row.get("ground_truth", ""))
    index_agent = "role" if is_handcrafted else "name"

    convo = ""
    raw_responses: List[Dict] = []
    predicted_step = -1
    predicted_agent = ""

    for idx, entry in enumerate(history):
        agent_name = entry.get(index_agent, "Unknown Agent")
        content = entry.get("content", "") or ""
        convo += f"Step {idx} - {agent_name}: {content}\n"
        prompt = build_step_prompt(problem, ground_truth, convo, idx, agent_name)
        try:
            answer = chat.chat(prompt, max_tokens=max_tokens) or ""
        except Exception as e:
            raw_responses.append({"step": idx, "error": str(e)})
            break
        raw_responses.append({"step": idx, "agent": agent_name,
                              "answer": answer[:600]})  # truncate to keep file size sane
        norm = answer.lower().strip()
        if norm.startswith("1. yes") or norm.startswith("1.yes") or norm.startswith("1: yes"):
            predicted_step = idx
            predicted_agent = agent_name
            break

    return {
        "trace_id": trace_row["_trace_id"],
        "gt_step": trace_row["_gt_step"],
        "gt_agent": trace_row["_gt_agent"],
        "predicted_step": predicted_step,
        "predicted_agent": predicted_agent,
        "n_queries": len(raw_responses),
        "raw_responses": raw_responses,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_subset_rows(subset: str) -> List[Dict]:
    """Load HF subset, attach (trace_id, gt_step, gt_agent), filter mistake_step != None."""
    from datasets import load_dataset
    ds = load_dataset("Kevin355/Who_and_When", subset)["train"]
    rows = []
    for tid, r in enumerate(ds):
        ms = r.get("mistake_step")
        if ms is None:
            continue
        rows.append({
            **r,
            "_trace_id": tid,
            "_gt_step": int(ms),
            "_gt_agent": r["mistake_agent"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["Hand-Crafted", "Algorithm-Generated"], required=True)
    ap.add_argument("--config", required=True,
                    help="Path to ensemble JSON (we'll iterate over discriminators).")
    ap.add_argument("--model-index", type=int, default=None,
                    help="If set, only run discriminator at this index (0/1/2). "
                         "Default: run all discriminators sequentially.")
    ap.add_argument("--output-root", default=None,
                    help="Default: data/baselines/who_step_by_step/<subset>/")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N traces")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel traces per LLM (each trace is sequential within itself)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--resume", action="store_true",
                    help="skip traces already present in output JSON")
    args = ap.parse_args()

    is_handcrafted = (args.subset == "Hand-Crafted")
    rows = load_subset_rows(args.subset)
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"Loaded {len(rows)} scorable traces from {args.subset}")

    with open(args.config) as f:
        cfg = json.load(f)
    specs = cfg["discriminators"] if isinstance(cfg, dict) else cfg
    if args.model_index is not None:
        specs = [specs[args.model_index]]

    out_root = Path(args.output_root) if args.output_root else \
        ROOT / "data" / "baselines" / "who_step_by_step" / args.subset
    out_root.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        # Use a clean filesystem-safe name based on the model.
        safe = spec["model"].replace("/", "_")
        out_path = out_root / f"{safe}.json"

        existing = []
        done_ids = set()
        if args.resume and out_path.exists():
            existing = json.load(open(out_path))
            done_ids = {r["trace_id"] for r in existing}
            print(f"  resume: {len(done_ids)} traces already done in {out_path.name}")

        chat = build_chat(spec)
        print(f"\n>>> Running {chat.id} on {args.subset} (workers={args.workers})")

        todo = [r for r in rows if r["_trace_id"] not in done_ids]
        results = list(existing)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(step_by_step_one_trace, chat, r, is_handcrafted, args.max_tokens): r
                for r in todo
            }
            for i, fut in enumerate(as_completed(futures), 1):
                r = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"  [{r['_trace_id']}] FAILED: {e}")
                    continue
                results.append(res)
                # Sort + persist after each completion.
                results_sorted = sorted(results, key=lambda x: x["trace_id"])
                with open(out_path, "w") as fout:
                    json.dump(results_sorted, fout, indent=2)
                hits = sum(1 for x in results if x["predicted_step"] == x["gt_step"])
                elapsed = time.time() - t0
                print(f"  [{i}/{len(todo)}] tid={r['_trace_id']:>3d} "
                      f"pred_step={res['predicted_step']:>3d} gt={res['gt_step']:>3d} "
                      f"n_q={res['n_queries']:>3d}  "
                      f"running step-acc: {hits}/{len(results)} = {hits/len(results):.3f}  "
                      f"({elapsed:.0f}s)")

        n = len(results)
        hits = sum(1 for x in results if x["predicted_step"] == x["gt_step"])
        print(f"\n=== {chat.id} on {args.subset} ===")
        print(f"  n={n}  Acc_step = {hits}/{n} = {hits/n:.3f}")
        print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
