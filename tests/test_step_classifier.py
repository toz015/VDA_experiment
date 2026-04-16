"""Tests for the step classifier and structured step pipeline."""

import json
from unittest.mock import MagicMock, patch
from vda.step_classifier import (
    classify_step_llm,
    classify_trace,
    is_environment_action,
    ENVIRONMENT_ACTION_TYPES,
    META_ACTION_TYPES,
)


def test_is_environment_action_known_types():
    assert is_environment_action("execute") is True
    assert is_environment_action("search") is True
    assert is_environment_action("click") is True
    assert is_environment_action("plan") is False
    assert is_environment_action("inform") is False
    assert is_environment_action("suggest") is False


def test_is_environment_action_normalizes():
    assert is_environment_action("Execute") is True
    assert is_environment_action(" search ") is True
    assert is_environment_action("write_script") is True


def test_meta_types_not_environment_actions():
    """Meta types are not environment actions, but should NOT be filtered out."""
    for t in META_ACTION_TYPES:
        assert is_environment_action(t) is False


def test_classify_step_llm_parses_json():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content='{"action_type": "search", "state": "search for GPT-4 release date"}')
        )]
    )
    action_type, state = classify_step_llm("WebSurfer", "some content", client=mock_client)
    assert action_type == "search"
    assert "GPT-4" in state


def test_classify_step_llm_fallback_on_bad_json():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content='not valid json')
        )]
    )
    action_type, state = classify_step_llm("Agent", "msg", client=mock_client)
    assert action_type == "unknown"


def test_classify_trace_uses_cache(tmp_path):
    """When cache exists, no LLM calls are made."""
    history = [
        {"role": "Agent_A", "content": "do something"},
        {"role": "Agent_B", "content": "do another thing"},
    ]
    cached_data = [
        {"agent": "Agent_A", "action_type": "execute", "state": "run script", "original_index": 0},
        {"agent": "Agent_B", "action_type": "inform", "state": "report result", "original_index": 1},
    ]
    cache_file = tmp_path / "_42.json"
    with open(cache_file, "w") as f:
        json.dump(cached_data, f)

    # No client needed — should read from cache
    result = classify_trace(history, trace_id=42, subset="", cache_dir=tmp_path)
    assert len(result) == 2
    assert result[0]["action_type"] == "execute"
    assert result[1]["action_type"] == "inform"


def test_classify_trace_invalidates_stale_cache(tmp_path):
    """Cache with wrong length should be re-classified."""
    history = [
        {"role": "Agent_A", "content": "do something"},
        {"role": "Agent_B", "content": "do another thing"},
    ]
    # Stale cache with only 1 entry (history has 2)
    stale = [{"agent": "A", "action_type": "x", "state": "y", "original_index": 0}]
    cache_file = tmp_path / "_99.json"
    with open(cache_file, "w") as f:
        json.dump(stale, f)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content='{"action_type": "run", "state": "test"}')
        )]
    )

    result = classify_trace(history, trace_id=99, subset="", cache_dir=tmp_path, client=mock_client)
    assert len(result) == 2
    # Client should have been called (cache was stale)
    assert mock_client.chat.completions.create.call_count == 2
