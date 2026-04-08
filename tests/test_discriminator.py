import math
from unittest.mock import MagicMock
import pytest

from vda.discriminator import OpenAIDiscriminator, extract_probability_a


def _make_logprob_entry(token, logprob):
    m = MagicMock()
    m.token = token
    m.logprob = logprob
    return m


def test_extract_probability_a_basic():
    # p(A) = exp(-0.1), p(B) = exp(-1.0); after renormalization p_A / (p_A + p_B)
    entries = [
        _make_logprob_entry(" A", -0.1),
        _make_logprob_entry(" B", -1.0),
        _make_logprob_entry(" C", -5.0),
    ]
    p, fallback = extract_probability_a(entries)
    p_a = math.exp(-0.1)
    p_b = math.exp(-1.0)
    expected = p_a / (p_a + p_b)
    assert abs(p - expected) < 1e-10
    assert fallback is False


def test_extract_matches_whitespace_and_case():
    for token in ["A", " A", "a", " a"]:
        entries = [_make_logprob_entry(token, -0.2), _make_logprob_entry("B", -0.8)]
        p, fallback = extract_probability_a(entries)
        assert 0.0 < p < 1.0
        assert fallback is False


def test_extract_fallback_when_neither_present():
    entries = [_make_logprob_entry("X", -0.1), _make_logprob_entry("Y", -0.2)]
    p, fallback = extract_probability_a(entries)
    assert p == 0.5
    assert fallback is True


def test_extract_only_a_present():
    # Only "A" token seen → p(A) = 1 before clipping
    entries = [_make_logprob_entry("A", -0.1), _make_logprob_entry("Z", -0.2)]
    p, fallback = extract_probability_a(entries)
    assert p == 1.0
    assert fallback is False


def test_discriminator_query_calls_openai_and_returns_prob(monkeypatch):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.logprobs.content = [
        MagicMock(top_logprobs=[
            _make_logprob_entry(" A", -0.1),
            _make_logprob_entry(" B", -1.0),
        ])
    ]
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    disc = OpenAIDiscriminator(model="gpt-4o-mini", temperature=0.7, client=mock_client)
    p = disc.query("some prompt")

    assert 0.0 < p < 1.0
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 1
    assert call_kwargs["logprobs"] is True
    assert call_kwargs["top_logprobs"] == 20
    assert disc.id == "gpt-4o-mini@T=0.7"
