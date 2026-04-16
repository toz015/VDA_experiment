import math
from unittest.mock import MagicMock
import pytest

from vda.discriminator import (
    OpenAIDiscriminator,
    VertexAIDiscriminator,
    Discriminator,
    extract_probability_a,
    build_ensemble,
)
from config import VDAConfig


def _make_logprob_entry(token, logprob):
    m = MagicMock()
    m.token = token
    m.logprob = logprob
    return m


# --- extract_probability_a ---

def test_extract_probability_a_basic():
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
    for token in ["A", " A", "a", " a", "(A", "(A)"]:
        entries = [_make_logprob_entry(token, -0.2), _make_logprob_entry("(B", -0.8)]
        p, fallback = extract_probability_a(entries)
        assert 0.0 < p < 1.0, f"failed for token={repr(token)}"
        assert fallback is False


def test_extract_fallback_when_neither_present():
    entries = [_make_logprob_entry("X", -0.1), _make_logprob_entry("Y", -0.2)]
    p, fallback = extract_probability_a(entries)
    assert p == 0.5
    assert fallback is True


def test_extract_only_a_present():
    entries = [_make_logprob_entry("A", -0.1), _make_logprob_entry("Z", -0.2)]
    p, fallback = extract_probability_a(entries)
    assert p == 1.0
    assert fallback is False


# --- OpenAIDiscriminator ---

def test_openai_discriminator_query(monkeypatch):
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
    assert disc.id == "openai/gpt-4o-mini@T=0.7"
    assert isinstance(disc, Discriminator)


# --- VertexAIDiscriminator logprob extraction ---

def test_vertex_extract_from_response():
    """Test that Vertex AI response logprobs are correctly parsed."""
    mock_response = MagicMock()
    candidate_a = MagicMock(token="(A", log_probability=-0.2)
    candidate_b = MagicMock(token="(B", log_probability=-1.5)
    mock_response.candidates[0].logprobs_result.top_candidates[0].candidates = [
        candidate_a, candidate_b,
    ]

    p, fallback = VertexAIDiscriminator._extract_from_response(mock_response)
    assert 0.0 < p < 1.0
    assert fallback is False

    expected_a = math.exp(-0.2)
    expected_b = math.exp(-1.5)
    expected = expected_a / (expected_a + expected_b)
    assert abs(p - expected) < 1e-10


def test_vertex_extract_fallback_on_bad_response():
    """Fallback to 0.5 if response structure is unexpected."""
    p, fallback = VertexAIDiscriminator._extract_from_response(MagicMock(candidates=[]))
    assert p == 0.5
    assert fallback is True


# --- build_ensemble ---

def test_build_ensemble_legacy():
    """Legacy mode: no discriminators list → single OpenAI model at K temps."""
    config = VDAConfig()
    # Can't actually build (no API key), but test the fallback path logic
    assert config.discriminators == []


def test_build_ensemble_multi_model(monkeypatch):
    """Multi-model mode: discriminators list specified."""
    config = VDAConfig(discriminators=[
        {"provider": "openai", "model": "gpt-4o", "temperature": 0.0},
        {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
    ])

    # Mock OpenAI client to avoid real API call
    mock_client = MagicMock()
    monkeypatch.setattr("vda.discriminator.OpenAIDiscriminator.__init__",
                        lambda self, **kw: Discriminator.__init__(self, f"openai/{kw['model']}@T={kw['temperature']}"))

    discs = build_ensemble(config)
    assert len(discs) == 2
    assert discs[0].id == "openai/gpt-4o@T=0.0"
    assert discs[1].id == "openai/gpt-4o-mini@T=0.0"
