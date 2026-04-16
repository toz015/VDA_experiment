"""Discriminator implementations for extracting P(root-cause) from LLM logprobs.

Supports multiple backends:
  - OpenAI  (GPT-4o, GPT-4o-mini, etc.)
  - Vertex AI  (Gemini models via Google Cloud)
"""

import math
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
    """Discriminator using Google Vertex AI (Gemini models with logprobs).

    Requires:
      pip install google-cloud-aiplatform
      gcloud auth application-default login   (or service account)

    Gemini models that support logprobs: gemini-1.5-pro, gemini-1.5-flash,
    gemini-2.0-flash, etc.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        project: Optional[str] = None,
        location: str = "us-central1",
        top_logprobs: int = 20,
    ):
        super().__init__(id=f"vertex/{model}@T={temperature}")
        self.model_name = model
        self.temperature = temperature
        self.top_logprobs = top_logprobs

        from google.cloud import aiplatform
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        if project:
            aiplatform.init(project=project, location=location)

        self._generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=1,
            response_logprobs=True,
            logprobs=top_logprobs,
        )
        self._model = GenerativeModel(model)

    def query(self, prompt: str) -> float:
        response = self._model.generate_content(
            prompt,
            generation_config=self._generation_config,
        )

        # Extract logprobs from Vertex AI response
        p, fallback = self._extract_from_response(response)
        if fallback:
            self.fallback_count += 1
        return p

    @staticmethod
    def _extract_from_response(response) -> Tuple[float, bool]:
        """Extract P(A) from Vertex AI logprobs response."""
        try:
            # Vertex AI logprobs are in response.candidates[0].logprobs_result
            logprobs_result = response.candidates[0].logprobs_result
            top_candidates = logprobs_result.top_candidates[0].candidates

            # Build entries compatible with extract_probability_a
            entries = []
            for c in top_candidates:
                entry = type("Entry", (), {"token": c.token, "logprob": c.log_probability})()
                entries.append(entry)

            return extract_probability_a(entries)
        except (AttributeError, IndexError):
            return 0.5, True


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
                ))
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return discs

    # Legacy fallback: single OpenAI model at K temperatures
    return [
        OpenAIDiscriminator(model=config.openai_model, temperature=t)
        for t in config.temperatures
    ]
