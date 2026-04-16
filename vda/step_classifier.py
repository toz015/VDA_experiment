"""LLM-based step classifier for extracting (action_type, state) from raw trace messages.

Mirrors FAMAS's llm_cluster3 pipeline: takes a raw agent message and produces
a structured (action_type, state_summary) tuple.

Results are cached to disk so each trace is only classified once.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

# Default cache location
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "step_cache"

# Action types that represent direct environment actions.
ENVIRONMENT_ACTION_TYPES = frozenset({
    "execute", "run", "search", "click", "navigate", "scroll",
    "type", "open", "install", "download", "upload", "write_script",
    "update_script", "debug", "fill", "input", "calculate",
    "extract", "filter", "read", "transcribe",
})

# Action types that are meta/coordination.
# NOTE: These can still be root-cause mistakes (e.g., bad instructions,
# premature conclusions). Do NOT use this set to filter out steps.
META_ACTION_TYPES = frozenset({
    "plan", "inform", "request", "verify", "confirm", "suggest",
    "summarize", "ask", "agree", "propose", "instruct", "terminate",
    "conclude", "provide", "continue", "clarify", "report",
})

CLASSIFICATION_PROMPT = """\
You are classifying a single step in a multi-agent task execution trace.

Given the agent name and raw message content, extract:
1. action_type: a short verb describing what the agent is doing (e.g., execute, search, plan, inform, click, navigate, install, request, verify, scroll, type, open, read, summarize, suggest, debug, download, write_script, calculate, extract, filter, fill, input, upload, transcribe, propose, instruct, agree, confirm, terminate, conclude, ask, provide, continue, clarify, report, update_script)
2. state: a brief (under 20 words) summary of the action's intent or outcome

Agent: {agent_name}
Message:
{content}

Respond with EXACTLY one line of JSON: {{"action_type": "...", "state": "..."}}"""


def classify_step_llm(
    agent_name: str,
    content: str,
    client=None,
    model: str = "gpt-4o-mini",
) -> Tuple[str, str]:
    """Classify a single step using an LLM. Returns (action_type, state)."""
    if client is None:
        from openai import OpenAI
        client = OpenAI()

    prompt = CLASSIFICATION_PROMPT.format(
        agent_name=agent_name,
        content=content[:2000],  # truncate very long messages
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip()

    try:
        obj = json.loads(raw)
        return obj.get("action_type", "unknown"), obj.get("state", "")
    except json.JSONDecodeError:
        return "unknown", raw[:80]


def _cache_key(trace_id, subset: str) -> str:
    return f"{subset}_{trace_id}"


def classify_trace(
    history: List[dict],
    trace_id,
    subset: str = "",
    cache_dir: Optional[Path] = None,
    client=None,
    model: str = "gpt-4o-mini",
) -> List[dict]:
    """Classify all steps in a trace history. Returns list of
    {"agent": str, "action_type": str, "state": str, "original_index": int}.

    Results are cached to <cache_dir>/<subset>_<trace_id>.json.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"{_cache_key(trace_id, subset)}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        # Validate cache matches history length
        if len(cached) == len(history):
            return cached

    if client is None:
        from openai import OpenAI
        client = OpenAI()

    results = []
    for i, h in enumerate(history):
        agent = h.get("role", h.get("name", "unknown"))
        content = h.get("content", "")
        action_type, state = classify_step_llm(agent, content, client=client, model=model)
        results.append({
            "agent": agent,
            "action_type": action_type,
            "state": state,
            "original_index": i,
        })

    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


def is_environment_action(action_type: str) -> bool:
    """Check if an action_type represents a direct environment action (vs meta/coordination).

    NOTE: This is for informational/analytical purposes only. Do NOT use this
    to filter out steps — meta steps (plan, instruct, verify, conclude) can be
    root-cause mistakes. The Who&When dataset has ground-truth errors on
    Orchestrator planning, bad instructions, and premature conclusions.
    """
    normed = action_type.lower().strip().replace(" ", "_")
    return normed in ENVIRONMENT_ACTION_TYPES
