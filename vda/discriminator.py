"""OpenAI discriminator that extracts P('A') from logprobs."""

import math
from typing import Optional, Tuple, List

from config import VDAConfig


def extract_probability_a(top_logprob_entries) -> Tuple[float, bool]:
    """Extract P('A') / (P('A') + P('B')) from a list of top-logprob entries.

    Each entry must expose `.token` and `.logprob`. Tokens are normalized with
    `.strip().upper()` before matching. Returns (probability, fallback_flag);
    fallback_flag is True iff neither 'A' nor 'B' was found (p=0.5 default).
    """
    logprob_a = None
    logprob_b = None
    for e in top_logprob_entries:
        key = e.token.strip().upper()
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


class OpenAIDiscriminator:
    """One discriminator instance = one (model, temperature) pair."""

    def __init__(
        self,
        model: str,
        temperature: float,
        client=None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.id = f"{model}@T={temperature}"
        self.fallback_count = 0

        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)

    def query(self, prompt: str) -> float:
        """Query the LLM and return P('A') ∈ [0, 1]."""
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


def build_ensemble(
    config: VDAConfig,
    api_key: Optional[str] = None,
) -> List[OpenAIDiscriminator]:
    """Build K discriminators from a single model at K different temperatures."""
    return [
        OpenAIDiscriminator(model=config.openai_model, temperature=t, api_key=api_key)
        for t in config.temperatures
    ]
