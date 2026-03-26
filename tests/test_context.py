"""
Tests for the 4-layer context management system.

Layer 1: Entity Store
Layer 2: Sliding Window
Layer 3: Auto-Summarizer
Layer 4: Context Builder + CallState
"""

import pytest
from unittest.mock import AsyncMock, patch

from server.bot.context import (
    CallEntities,
    CallState,
    get_or_create_call_state,
    get_call_state,
    remove_call_state,
    get_active_call_count,
    get_sliding_window,
    summarize_turns,
    MAX_WINDOW_TURNS,
    SUMMARIZE_THRESHOLD,
)


# =============================================================================
# Layer 1: Entity Store
# =============================================================================


class TestCallEntities:
    def test_default_values(self):
        e = CallEntities()
        assert e.caller_name is None
        assert e.intent is None
        assert e.urgency == "normal"
        assert e.sentiment == "neutral"
        assert e.booking_confirmed is False

    def test_update_single_field(self):
        e = CallEntities()
        changed = e.update(caller_name="John")
        assert e.caller_name == "John"
        assert changed == ["caller_name"]

    def test_update_multiple_fields(self):
        e = CallEntities()
        changed = e.update(
            caller_name="Sarah",
            intent="new_appointment",
            service_type="cleaning",
        )
        assert e.caller_name == "Sarah"
        assert e.intent == "new_appointment"
        assert e.service_type == "cleaning"
        assert len(changed) == 3

    def test_update_ignores_none(self):
        e = CallEntities(caller_name="John")
        changed = e.update(caller_name=None)
        assert e.caller_name == "John"
        assert changed == []

    def test_update_ignores_unknown_fields(self):
        e = CallEntities()
        changed = e.update(nonexistent_field="value")
        assert changed == []

    def test_update_returns_only_changed_fields(self):
        e = CallEntities(caller_name="John")
        changed = e.update(caller_name="John", intent="cancel")
        assert "caller_name" not in changed  # Same value, not changed
        assert "intent" in changed

    def test_to_context_block_empty(self):
        e = CallEntities()
        assert e.to_context_block() == ""

    def test_to_context_block_with_data(self):
        e = CallEntities(
            caller_name="Sarah",
            intent="new_appointment",
            service_type="cleaning",
        )
        block = e.to_context_block()
        assert "KNOWN FACTS" in block
        assert "caller_name: Sarah" in block
        assert "intent: new_appointment" in block
        assert "service_type: cleaning" in block

    def test_to_context_block_hides_defaults(self):
        """Normal urgency and neutral sentiment should not appear."""
        e = CallEntities(caller_name="John", urgency="normal", sentiment="neutral")
        block = e.to_context_block()
        assert "urgency" not in block
        assert "sentiment" not in block

    def test_to_context_block_shows_non_default_urgency(self):
        e = CallEntities(caller_name="John", urgency="emergency")
        block = e.to_context_block()
        assert "urgency: emergency" in block

    def test_to_dict_excludes_none(self):
        e = CallEntities(caller_name="Sarah", intent="cancel")
        d = e.to_dict()
        assert "caller_name" in d
        assert "intent" in d
        assert "preferred_date" not in d


# =============================================================================
# Layer 2: Sliding Window
# =============================================================================


class TestSlidingWindow:
    def test_returns_all_if_under_limit(self):
        turns = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = get_sliding_window(turns)
        assert len(result) == 2

    def test_trims_to_window_size(self):
        # Create 10 turns (20 messages), window is 4 turns (8 messages)
        turns = []
        for i in range(10):
            turns.append({"role": "user", "content": f"User message {i}"})
            turns.append({"role": "assistant", "content": f"Bot message {i}"})

        result = get_sliding_window(turns)
        assert len(result) == MAX_WINDOW_TURNS * 2
        # Should be the LAST turns
        assert result[0]["content"] == f"User message {10 - MAX_WINDOW_TURNS}"

    def test_exact_window_size(self):
        turns = []
        for i in range(MAX_WINDOW_TURNS):
            turns.append({"role": "user", "content": f"msg {i}"})
            turns.append({"role": "assistant", "content": f"reply {i}"})

        result = get_sliding_window(turns)
        assert len(result) == MAX_WINDOW_TURNS * 2

    def test_empty_turns(self):
        result = get_sliding_window([])
        assert result == []


# =============================================================================
# Layer 3: Auto-Summarizer
# =============================================================================


class TestAutoSummarizer:
    @pytest.mark.asyncio
    async def test_summarize_empty_returns_existing(self):
        result = await summarize_turns([], existing_summary="Old summary")
        assert result == "Old summary"

    @pytest.mark.asyncio
    async def test_summarize_empty_no_existing(self):
        result = await summarize_turns([], existing_summary=None)
        assert result == ""

    @pytest.mark.asyncio
    @patch("server.bot.context._get_openai_client")
    async def test_summarize_calls_openai(self, mock_get_client):
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Patient called about a cleaning."
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        turns = [
            {"role": "user", "content": "Hi, I need a cleaning"},
            {"role": "assistant", "content": "I can help with that!"},
        ]
        result = await summarize_turns(turns)
        assert result == "Patient called about a cleaning."
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    @patch("server.bot.context._get_openai_client")
    async def test_summarize_handles_api_error(self, mock_get_client):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client

        result = await summarize_turns(
            [{"role": "user", "content": "test"}],
            existing_summary="Fallback summary",
        )
        assert result == "Fallback summary"


# =============================================================================
# Layer 4: CallState (Context Builder)
# =============================================================================


class TestCallState:
    def test_create_call_state(self):
        state = CallState(call_sid="test-001")
        assert state.call_sid == "test-001"
        assert state.current_state == "greeting"
        assert state.turn_count == 0
        assert state.outcome is None

    def test_add_turn(self):
        state = CallState(call_sid="test-001")
        state.add_turn("user", "Hello")
        state.add_turn("assistant", "Hi there!")
        assert len(state.all_turns) == 2
        assert state.turn_count == 1  # Only user turns counted

    def test_add_tool_call(self):
        state = CallState(call_sid="test-001")
        state.add_tool_call("check_availability", {"date": "2026-03-25"}, "mock_result")
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0]["name"] == "check_availability"

    def test_build_context_block_empty(self):
        state = CallState(call_sid="test-001")
        block = state.build_context_block()
        assert block == ""

    def test_build_context_block_with_entities(self):
        state = CallState(call_sid="test-001")
        state.entities.update(caller_name="John", intent="new_appointment")
        block = state.build_context_block()
        assert "KNOWN FACTS" in block
        assert "John" in block

    def test_build_context_block_with_summary(self):
        state = CallState(call_sid="test-001")
        state.summary = "Patient wants a cleaning appointment."
        block = state.build_context_block()
        assert "CONVERSATION SO FAR" in block
        assert "cleaning" in block

    def test_get_context_messages_within_window(self):
        state = CallState(call_sid="test-001")
        state.add_turn("user", "Hello")
        state.add_turn("assistant", "Hi!")
        msgs = state.get_context_messages()
        assert len(msgs) == 2

    def test_get_context_messages_trimmed(self):
        state = CallState(call_sid="test-001")
        for i in range(10):
            state.add_turn("user", f"msg {i}")
            state.add_turn("assistant", f"reply {i}")
        msgs = state.get_context_messages()
        assert len(msgs) == MAX_WINDOW_TURNS * 2

    def test_get_call_summary(self):
        state = CallState(call_sid="test-001")
        state.entities.update(caller_name="Sarah", intent="cancel")
        state.outcome = "cancelled"
        summary = state.get_call_summary()
        assert "test-001" in summary
        assert "Sarah" in summary
        assert "cancel" in summary
        assert "cancelled" in summary


# =============================================================================
# Active Call Store
# =============================================================================


class TestActiveCallStore:
    def setup_method(self):
        """Clear the store before each test."""
        from server.bot.context import _active_calls
        _active_calls.clear()

    def test_get_or_create_new(self):
        state = get_or_create_call_state("call-001")
        assert state.call_sid == "call-001"
        assert get_active_call_count() == 1

    def test_get_or_create_existing(self):
        state1 = get_or_create_call_state("call-001")
        state1.entities.update(caller_name="John")
        state2 = get_or_create_call_state("call-001")
        assert state2.entities.caller_name == "John"
        assert get_active_call_count() == 1

    def test_get_call_state_exists(self):
        get_or_create_call_state("call-001")
        assert get_call_state("call-001") is not None

    def test_get_call_state_not_exists(self):
        assert get_call_state("nonexistent") is None

    def test_remove_call_state(self):
        get_or_create_call_state("call-001")
        removed = remove_call_state("call-001")
        assert removed is not None
        assert removed.call_sid == "call-001"
        assert get_active_call_count() == 0

    def test_remove_nonexistent(self):
        removed = remove_call_state("nonexistent")
        assert removed is None

    def test_multiple_calls_isolated(self):
        state1 = get_or_create_call_state("call-001")
        state2 = get_or_create_call_state("call-002")
        state1.entities.update(caller_name="Alice")
        state2.entities.update(caller_name="Bob")
        assert state1.entities.caller_name == "Alice"
        assert state2.entities.caller_name == "Bob"
        assert get_active_call_count() == 2
