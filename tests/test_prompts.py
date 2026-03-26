"""
Tests for per-state prompt generation.

Verifies prompts are:
- Short enough for fast LLM response
- Dynamic (change with call state)
- Contain required instructions
"""

import pytest

from server.bot.context import CallState
from server.bot.prompts import (
    get_greeting_prompt,
    get_intent_prompt,
    get_booking_prompt,
    get_availability_prompt,
    get_confirm_prompt,
    get_reschedule_prompt,
    get_cancel_prompt,
    get_question_prompt,
    get_escalate_prompt,
    get_close_prompt,
)


@pytest.fixture
def empty_state():
    return CallState(call_sid="test-prompt-001")


@pytest.fixture
def populated_state():
    state = CallState(call_sid="test-prompt-002")
    state.entities.update(
        caller_name="Sarah",
        intent="new_appointment",
        service_type="cleaning",
        preferred_date="next Tuesday",
        preferred_time="afternoon",
        provider_preference="Dr. Chen",
    )
    return state


class TestGreetingPrompt:
    def test_contains_clinic_name(self, empty_state):
        prompt = get_greeting_prompt(empty_state)
        assert "Bright Smile Dental" in prompt

    def test_contains_hours(self, empty_state):
        prompt = get_greeting_prompt(empty_state)
        assert "Monday" in prompt
        assert "8am" in prompt or "8 AM" in prompt

    def test_contains_voice_style_rules(self, empty_state):
        prompt = get_greeting_prompt(empty_state)
        assert "never" in prompt.lower() or "Never" in prompt

    def test_includes_context_when_populated(self, populated_state):
        prompt = get_greeting_prompt(populated_state)
        assert "KNOWN FACTS" in prompt
        assert "Sarah" in prompt


class TestIntentPrompt:
    def test_mentions_all_routes(self, empty_state):
        prompt = get_intent_prompt(empty_state)
        assert "route_new_appointment" in prompt
        assert "route_reschedule" in prompt
        assert "route_cancel" in prompt
        assert "route_question" in prompt
        assert "route_emergency" in prompt


class TestBookingPrompt:
    def test_lists_required_info(self, empty_state):
        prompt = get_booking_prompt(empty_state)
        assert "name" in prompt.lower()
        assert "service" in prompt.lower()


class TestAvailabilityPrompt:
    def test_includes_patient_info(self, populated_state):
        prompt = get_availability_prompt(populated_state)
        assert "Sarah" in prompt
        assert "cleaning" in prompt
        assert "next Tuesday" in prompt

    def test_empty_state_still_works(self, empty_state):
        prompt = get_availability_prompt(empty_state)
        assert "check_availability" in prompt


class TestConfirmPrompt:
    def test_includes_appointment_details(self, populated_state):
        prompt = get_confirm_prompt(populated_state)
        assert "Sarah" in prompt
        assert "cleaning" in prompt

    def test_warns_about_confirmation(self, populated_state):
        prompt = get_confirm_prompt(populated_state)
        assert "confirm" in prompt.lower()


class TestClosePrompt:
    def test_booked_outcome_message(self, populated_state):
        populated_state.outcome = "booked"
        prompt = get_close_prompt(populated_state)
        assert "next Tuesday" in prompt

    def test_cancelled_outcome_message(self, empty_state):
        empty_state.outcome = "cancelled"
        prompt = get_close_prompt(empty_state)
        assert "cancelled" in prompt


class TestPromptLength:
    """Verify prompts are reasonably short for fast LLM response."""

    MAX_CHARS = 2000  # Keep prompts under 2000 chars for fast TTFT

    def test_intent_prompt_short(self, empty_state):
        prompt = get_intent_prompt(empty_state)
        assert len(prompt) < self.MAX_CHARS, f"Intent prompt too long: {len(prompt)} chars"

    def test_booking_prompt_short(self, empty_state):
        prompt = get_booking_prompt(empty_state)
        assert len(prompt) < self.MAX_CHARS

    def test_cancel_prompt_short(self, empty_state):
        prompt = get_cancel_prompt(empty_state)
        assert len(prompt) < self.MAX_CHARS

    def test_escalate_prompt_short(self, empty_state):
        prompt = get_escalate_prompt(empty_state)
        assert len(prompt) < self.MAX_CHARS

    def test_close_prompt_short(self, empty_state):
        prompt = get_close_prompt(empty_state)
        assert len(prompt) < self.MAX_CHARS
