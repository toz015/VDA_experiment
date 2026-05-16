"""Discriminator implementations for extracting P(root-cause) from LLM logprobs.

Supports multiple backends:
  - OpenAI  (GPT-4o, GPT-4o-mini, etc.) via Chat Completions logprobs
  - Vertex AI  (Gemini native) via response_logprobs OR JSON output
  - Vertex MaaS  (Qwen / GPT-OSS / etc.) via OpenAI-compatible JSON output
"""

import json
import math
import re
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from config import VDAConfig


# ---------------------------------------------------------------------------
# Shared logprob extraction
# ---------------------------------------------------------------------------

def extract_probability_a(top_logprob_entries) -> Tuple[float, bool]:
    """Extract P('A') / (P('A') + P('B')) from a list of top-logprob entries.

    Each entry must expose `.token` and `.logprob`. Tokens are normalized with
    `.strip().upper()` before matching. Returns (probability, fallback_flag);
    fallback_flag is True iff neither 'A' nor 'B' was found (p=0.5 default).
    """
    logprob_a = None
    logprob_b = None
    for e in top_logprob_entries:
        # Normalize: strip whitespace and surrounding parentheses so that
        # "(A", "(A)", "A", "a" all resolve to "A", and similarly for "B".
        key = e.token.strip().lstrip("(").rstrip(")").strip().upper()
        if key == "A" and logprob_a is None:
            logprob_a = e.logprob
        elif key == "B" and logprob_b is None:
            logprob_b = e.logprob

    if logprob_a is None and logprob_b is None:
        return 0.5, True
    if logprob_a is None:
        return 0.0, False
    if logprob_b is None:
        return 1.0, False

    # Softmax-normalize the two tokens.
    m = max(logprob_a, logprob_b)
    e_a = math.exp(logprob_a - m)
    e_b = math.exp(logprob_b - m)
    return e_a / (e_a + e_b), False


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Discriminator(ABC):
    """One discriminator instance = one LLM configuration."""

    def __init__(self, id: str):
        self.id = id
        self.fallback_count = 0

    @abstractmethod
    def query(self, prompt: str) -> float:
        """Query the LLM and return P('A') in [0, 1]."""


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIDiscriminator(Discriminator):
    """Discriminator using OpenAI Chat Completions API (logprobs)."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        client=None,
        api_key: Optional[str] = None,
    ):
        super().__init__(id=f"openai/{model}@T={temperature}")
        self.model = model
        self.temperature = temperature

        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)

    def query(self, prompt: str) -> float:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            messages=[{"role": "user", "content": prompt}],
        )
        top = resp.choices[0].logprobs.content[0].top_logprobs
        p, fallback = extract_probability_a(top)
        if fallback:
            self.fallback_count += 1
        return p


# ---------------------------------------------------------------------------
# Vertex AI (Gemini) backend
# ---------------------------------------------------------------------------

class VertexAIDiscriminator(Discriminator):
    """Discriminator using Google Vertex AI via the google-genai SDK.

    Requires:
      pip install google-cloud-aiplatform  (pulls in google-genai)
      gcloud auth application-default login   (or service account)

    Supports Gemini 2.x models. For Gemini 2.5 thinking models
    (e.g. gemini-2.5-flash), pass `thinking_budget=0` to disable thinking so
    that logprobs at position 0 reflect the A/B answer rather than a thinking
    control token. gemini-2.5-pro does not allow thinking_budget=0 and is
    therefore not usable for logprob-based discrimination.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        project: Optional[str] = None,
        location: str = "us-central1",
        top_logprobs: int = 20,
        thinking_budget: Optional[int] = 0,
        timeout_ms: int = 60000,
        max_retries: int = 2,
    ):
        super().__init__(id=f"vertex/{model}@T={temperature}")
        self.model_name = model
        self.temperature = temperature
        self.top_logprobs = top_logprobs
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries

        from google import genai
        from google.genai import types as genai_types

        self._genai_types = genai_types
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=genai_types.HttpOptions(timeout=timeout_ms),
        )

        cfg_kwargs = dict(
            temperature=temperature,
            max_output_tokens=1,
            response_logprobs=True,
            logprobs=top_logprobs,
        )
        if thinking_budget is not None:
            cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        self._config = genai_types.GenerateContentConfig(**cfg_kwargs)

    def query(self, prompt: str) -> float:
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._config,
                )
                p, fallback = self._extract_from_response(response)
                if fallback:
                    self.fallback_count += 1
                return p
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    time.sleep(1.5 ** attempt)
                    continue
                raise

    @staticmethod
    def _extract_from_response(response) -> Tuple[float, bool]:
        """Extract P(A) from google-genai logprobs response."""
        try:
            logprobs_result = response.candidates[0].logprobs_result
            top_candidates = logprobs_result.top_candidates[0].candidates
            entries = [
                type("Entry", (), {"token": c.token, "logprob": c.log_probability})()
                for c in top_candidates
            ]
            return extract_probability_a(entries)
        except (AttributeError, IndexError, TypeError):
            return 0.5, True


# ---------------------------------------------------------------------------
# JSON-output helpers (for backends without per-token logprobs)
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


_ANSWER_KEYS = ("answer", "final_answer", "final", "Answer", "ANSWER", "verdict", "label", "choice")
_CONF_KEYS = ("confidence", "Confidence", "CONFIDENCE", "prob", "probability", "p")
_BARE_AB_RE = re.compile(r'(?<![A-Za-z])([AB])(?![A-Za-z])')


def _coerce_answer(value) -> str:
    """Map any answer payload to 'A'/'B'/'' (empty = unknown)."""
    if value is None:
        return ""
    s = str(value).strip().upper().lstrip("(").rstrip(")")
    if s in ("A", "B"):
        return s
    if s in ("YES", "Y", "TRUE", "1"):
        return "A"
    if s in ("NO", "N", "FALSE", "0"):
        return "B"
    return ""


def _extract_from_dict(d: dict) -> Tuple[str, Optional[float]]:
    """Pull (answer, confidence) out of a dict using flexible key names."""
    answer = ""
    for k in _ANSWER_KEYS:
        if k in d:
            answer = _coerce_answer(d[k])
            if answer:
                break
    # Some models use {"is_root_cause": "Yes"} per-step
    if not answer and "is_root_cause" in d:
        answer = _coerce_answer(d["is_root_cause"])
    conf = None
    for k in _CONF_KEYS:
        if k in d:
            try:
                conf = float(d[k])
                break
            except (TypeError, ValueError):
                pass
    return answer, conf


def parse_answer_confidence(raw: str) -> Tuple[float, bool]:
    """Parse the discriminator response → (theta, fallback_flag).

    Accepts a wide set of shapes, since different MaaS backends emit different
    JSON structures even with response_format=json_object:

      - {"answer": "A", "confidence": 0.8}            (canonical)
      - {"answer": "A"}                               (no confidence -> assume 1.0)
      - {"final": "A"} / {"Answer": "A"} / {"verdict": "B"}
      - {"answer": "Yes"} / {"answer": "No"}          (Yes -> A, No -> B)
      - [{"answer": "A", "confidence": 0.8}]          (single-element list)
      - ["A"]                                          (bare list of letter)
      - [{"step": 0, "is_root_cause": "Yes"}, ...]    (per-step list — first hit wins)
      - "A" or "B"                                     (raw string)
      - completely malformed text — last-ditch: scan for a single 'A'/'B' token

    theta = confidence if answer=="A" else 1-confidence (default conf = 1.0
    when the model gave a verdict but no confidence value).
    """
    if not raw:
        return 0.5, True
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
        if text.startswith("json"):
            text = text[4:].strip()

    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # First {...} substring (flat — no nested objects).
        m = _JSON_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None

    answer = ""
    conf: Optional[float] = None

    if isinstance(obj, dict):
        answer, conf = _extract_from_dict(obj)
    elif isinstance(obj, list) and obj:
        # Try first element as dict, else as bare string ("A").
        first = obj[0]
        if isinstance(first, dict):
            answer, conf = _extract_from_dict(first)
            # If no answer in first element, scan the rest for is_root_cause=yes
            if not answer:
                for item in obj:
                    if isinstance(item, dict) and _coerce_answer(item.get("is_root_cause")) == "A":
                        answer = "A"
                        break
        else:
            answer = _coerce_answer(first)
    elif isinstance(obj, str):
        answer = _coerce_answer(obj)

    # Last-ditch: pure text response, look for an isolated 'A' or 'B' token.
    if not answer:
        match = _BARE_AB_RE.search(text)
        if match:
            answer = match.group(1).upper()

    if not answer:
        return 0.5, True

    # If no explicit confidence, model gave a categorical verdict — treat as 1.0.
    if conf is None:
        conf = 1.0
    if not (0.0 <= conf <= 1.0):
        conf = max(0.0, min(1.0, conf))

    if answer == "A":
        return conf, False
    if answer == "B":
        return 1.0 - conf, False
    return 0.5, True


# ---------------------------------------------------------------------------
# Vertex MaaS (OpenAI-compatible) backend — Qwen / GPT-OSS / etc.
# ---------------------------------------------------------------------------

class VertexMaaSDiscriminator(Discriminator):
    """Discriminator using Vertex AI MaaS endpoint via OpenAI-compatible API.

    Used for models like:
      - qwen/qwen3-next-80b-a3b-instruct-maas
      - openai/gpt-oss-120b-maas

    These endpoints do NOT provide reliable per-token logprobs, so this
    discriminator uses JSON-output mode and parses {"answer", "confidence"}.

    Auth: bearer token from google.auth.default() (refreshed once per query
    if expired).
    """

    def __init__(
        self,
        model: str,
        project: str,
        location: str = "global",
        temperature: float = 0.2,
        max_completion_tokens: int = 1024,
        max_retries: int = 2,
    ):
        super().__init__(id=f"vertex_maas/{model}@T={temperature}")
        self.model = model
        self.project = project
        self.location = location
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries

        from google.auth import default as auth_default
        from google.auth.transport.requests import Request
        from openai import OpenAI

        self._auth_default = auth_default
        self._auth_request = Request
        self._creds, _ = auth_default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._refresh_token()

        # Vertex MaaS OpenAI-compat URL
        if location == "global":
            host = "aiplatform.googleapis.com"
        else:
            host = f"{location}-aiplatform.googleapis.com"
        base_url = (
            f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"
        )
        self._OpenAI = OpenAI
        self._base_url = base_url
        self._client = OpenAI(api_key=self._creds.token, base_url=base_url)

    def _refresh_token(self):
        if not self._creds.valid:
            self._creds.refresh(self._auth_request())

    def _rebuild_client(self):
        self._refresh_token()
        self._client = self._OpenAI(api_key=self._creds.token, base_url=self._base_url)

    def query(self, prompt: str) -> float:
        last_exc = None
        max_attempts = self.max_retries + 8  # extra retries for 429
        for attempt in range(max_attempts):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content or ""
                p, fallback = parse_answer_confidence(raw)
                if fallback:
                    self.fallback_count += 1
                return p
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                is_rate_limit = ("429" in msg or "resource_exhausted" in msg
                                 or "rate" in msg and "limit" in msg)
                is_auth = "auth" in msg or "401" in msg or "unauth" in msg or "expired" in msg
                if is_auth:
                    try:
                        self._rebuild_client()
                    except Exception:
                        pass
                if attempt < max_attempts - 1:
                    if is_rate_limit:
                        # Long backoff: 8, 16, 32, 60, 60, 60, 60, 60 sec
                        wait = min(60, 8 * (2 ** min(attempt, 3)))
                    else:
                        wait = 1.5 ** attempt
                    time.sleep(wait)
                    continue
                raise


# ---------------------------------------------------------------------------
# Vertex Native (Gemini) JSON-output backend
# ---------------------------------------------------------------------------

class VertexNativeJSONDiscriminator(Discriminator):
    """Gemini via google-genai SDK with JSON-mode output.

    Used as an alternative to logprob-based extraction when ensemble consistency
    requires JSON output across all discriminators (e.g. mixing with MaaS Qwen
    and GPT-OSS that don't expose logprobs).
    """

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        project: Optional[str] = None,
        location: str = "global",
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        thinking_budget: Optional[int] = 0,
        timeout_ms: int = 60000,
        max_retries: int = 2,
    ):
        super().__init__(id=f"vertex_json/{model}@T={temperature}")
        self.model_name = model
        self.temperature = temperature
        self.max_retries = max_retries

        from google import genai
        from google.genai import types as genai_types

        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=genai_types.HttpOptions(timeout=timeout_ms),
        )

        cfg_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        )
        if thinking_budget is not None:
            cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        self._config = genai_types.GenerateContentConfig(**cfg_kwargs)

    def query(self, prompt: str) -> float:
        last_exc = None
        max_attempts = self.max_retries + 8
        for attempt in range(max_attempts):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._config,
                )
                raw = response.text or ""
                p, fallback = parse_answer_confidence(raw)
                if fallback:
                    self.fallback_count += 1
                return p
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                is_rate_limit = ("429" in msg or "resource_exhausted" in msg
                                 or ("rate" in msg and "limit" in msg))
                if attempt < max_attempts - 1:
                    if is_rate_limit:
                        wait = min(60, 8 * (2 ** min(attempt, 3)))
                    else:
                        wait = 1.5 ** attempt
                    time.sleep(wait)
                    continue
                raise


# ---------------------------------------------------------------------------
# Ensemble builder
# ---------------------------------------------------------------------------

def build_ensemble(config: VDAConfig) -> List[Discriminator]:
    """Build K discriminators from config.discriminators list.

    Each entry in config.discriminators is a dict:
        {"provider": "openai"|"vertex", "model": "...", "temperature": 0.0, ...}

    Falls back to the legacy single-model config if discriminators list is empty.
    """
    if config.discriminators:
        discs = []
        for spec in config.discriminators:
            provider = spec["provider"]
            model = spec["model"]
            temp = spec.get("temperature", 0.0)

            if provider == "openai":
                discs.append(OpenAIDiscriminator(
                    model=model,
                    temperature=temp,
                    api_key=spec.get("api_key"),
                ))
            elif provider == "vertex":
                discs.append(VertexAIDiscriminator(
                    model=model,
                    temperature=temp,
                    project=spec.get("project"),
                    location=spec.get("location", "us-central1"),
                    thinking_budget=spec.get("thinking_budget", 0),
                ))
            elif provider == "vertex_json":
                discs.append(VertexNativeJSONDiscriminator(
                    model=model,
                    temperature=temp,
                    project=spec.get("project"),
                    location=spec.get("location", "global"),
                    max_output_tokens=spec.get("max_output_tokens", 1024),
                    thinking_budget=spec.get("thinking_budget", 0),
                ))
            elif provider == "vertex_maas":
                discs.append(VertexMaaSDiscriminator(
                    model=model,
                    temperature=temp,
                    project=spec["project"],
                    location=spec.get("location", "global"),
                    max_completion_tokens=spec.get("max_completion_tokens", 1024),
                ))
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return discs

    # Legacy fallback: single OpenAI model at K temperatures
    return [
        OpenAIDiscriminator(model=config.openai_model, temperature=t)
        for t in config.temperatures
    ]
