"""
Integration tests for the CallGenius voice pipeline.

Tests the full flow: state machine transitions + tool calls + context management.
Does NOT require API keys — tests the logic layer only.
"""

import json
import pytest

from server.bot.context import CallState, get_or_create_call_state, remove_call_state
from server.bot.flows import (
    create_greeting_node,
    create_intent_node,
    create_booking_node,
    create_availability_node,
    create_confirm_node,
    create_cancel_node,
    create_question_node,
    create_escalate_node,
    create_close_node,
)
from server.bot.tools import (
    check_availability_handler,
    book_appointment_handler,
    cancel_appointment_handler,
    get_clinic_info_handler,
)
from server.bot.prompts import get_greeting_prompt


class TestFullBookingFlow:
    """Simulate a complete new appointment booking flow."""

    @pytest.fixture
    def call_state(self):
        state = get_or_create_call_state("integration-booking-001")
        yield state
        remove_call_state("integration-booking-001")

    def test_step1_greeting(self, call_state):
        """Patient calls, AI greets."""
        node = create_greeting_node(call_state)
        assert node["name"] == "greeting"
        assert call_state.current_state == "greeting"
        # Verify clinic info in role message
        assert "Bright Smile Dental" in node["role_message"]

    def test_step2_intent_routing(self, call_state):
        """Patient says they want a cleaning — routed to booking."""
        call_state.current_state = "identify_intent"
        node = create_intent_node(call_state)
        # Verify all routing functions available
        func_names = [f.name for f in node["functions"]]
        assert "route_new_appointment" in func_names

    def test_step3_gather_info(self, call_state):
        """Gather patient name and service type."""
        call_state.entities.update(intent="new_appointment")
        call_state.current_state = "booking"
        node = create_booking_node(call_state)
        assert "ready_to_check_availability" in [f.name for f in node["functions"]]

    @pytest.mark.asyncio
    async def test_step4_check_availability(self, call_state):
        """Check calendar for available slots."""
        call_state.entities.update(
            caller_name="Sarah Johnson",
            service_type="cleaning",
            preferred_date="next Tuesday",
        )
        call_state.current_state = "check_availability"

        result = json.loads(await check_availability_handler(
            {"date": "2026-03-31", "service_type": "cleaning"},
            call_state,
        ))
        assert len(result["available_slots"]) > 0
        assert result["available_slots"][0]["service"] == "cleaning"

    @pytest.mark.asyncio
    async def test_step5_confirm_and_book(self, call_state):
        """Patient confirms slot, appointment is booked."""
        call_state.entities.update(
            caller_name="Sarah Johnson",
            service_type="cleaning",
            preferred_date="2026-03-31",
            preferred_time="14:00",
            provider_preference="Dr. Sarah Chen",
            patient_phone="555-0199",
        )

        result = json.loads(await book_appointment_handler(
            {
                "patient_name": "Sarah Johnson",
                "date": "2026-03-31",
                "time": "14:00",
                "service_type": "cleaning",
                "provider": "Dr. Sarah Chen",
                "patient_phone": "555-0199",
            },
            call_state,
        ))
        assert result["success"] is True
        assert result["sms_sent"] is True
        assert call_state.entities.booking_confirmed is True
        assert call_state.entities.appointment_id is not None

    def test_step6_close(self, call_state):
        """Call wraps up after booking."""
        call_state.outcome = "booked"
        call_state.current_state = "close"
        node = create_close_node(call_state)
        assert "end_call" in [f.name for f in node["functions"]]

    @pytest.mark.asyncio
    async def test_full_flow_end_to_end(self, call_state):
        """Complete booking flow as one test."""
        # 1. Greeting
        greeting_node = create_greeting_node(call_state)
        assert greeting_node["name"] == "greeting"

        # 2. Intent
        call_state.current_state = "identify_intent"
        intent_node = create_intent_node(call_state)
        assert intent_node["name"] == "identify_intent"

        # 3. Booking info gathered
        call_state.entities.update(
            intent="new_appointment",
            caller_name="Mike Davis",
            service_type="filling",
        )
        call_state.current_state = "booking"

        # 4. Check availability
        call_state.current_state = "check_availability"
        slots = json.loads(await check_availability_handler(
            {"date": "2026-04-01", "service_type": "filling"},
            call_state,
        ))
        assert len(slots["available_slots"]) > 0
        chosen = slots["available_slots"][0]

        # 5. Confirm and book
        call_state.entities.update(
            preferred_date=chosen["date"],
            preferred_time=chosen["time"],
            provider_preference=chosen["provider"],
        )
        call_state.current_state = "confirm_booking"
        booking = json.loads(await book_appointment_handler(
            {
                "patient_name": "Mike Davis",
                "date": chosen["date"],
                "time": chosen["time"],
                "service_type": "filling",
                "provider": chosen["provider"],
            },
            call_state,
        ))
        assert booking["success"] is True
        call_state.outcome = "booked"

        # 6. Close
        call_state.current_state = "close"
        close_node = create_close_node(call_state)
        assert close_node["name"] == "close"

        # Verify final state
        assert call_state.entities.booking_confirmed is True
        assert call_state.outcome == "booked"
        assert len(call_state.tool_calls) == 2  # check + book


class TestCancelFlow:
    """Simulate appointment cancellation."""

    @pytest.fixture
    def call_state(self):
        state = get_or_create_call_state("integration-cancel-001")
        yield state
        remove_call_state("integration-cancel-001")

    @pytest.mark.asyncio
    async def test_cancel_flow(self, call_state):
        # Intent → Cancel
        call_state.entities.update(intent="cancel", caller_name="John Smith")
        call_state.current_state = "cancel"

        node = create_cancel_node(call_state)
        assert node["name"] == "cancel"

        result = json.loads(await cancel_appointment_handler(
            {"patient_name": "John Smith", "appointment_date": "2026-03-28"},
            call_state,
        ))
        assert result["success"] is True

        call_state.outcome = "cancelled"
        close_node = create_close_node(call_state)
        assert close_node["name"] == "close"


class TestQuestionFlow:
    """Simulate a general question flow."""

    @pytest.fixture
    def call_state(self):
        state = get_or_create_call_state("integration-question-001")
        yield state
        remove_call_state("integration-question-001")

    @pytest.mark.asyncio
    async def test_question_then_booking(self, call_state):
        """Patient asks a question, then decides to book."""
        # Ask about insurance
        call_state.entities.update(intent="question")
        call_state.current_state = "general_question"

        info = json.loads(await get_clinic_info_handler(
            {"question_type": "insurance"}, call_state,
        ))
        assert "Delta Dental" in info["info"]

        # Then decides to book
        call_state.entities.update(intent="new_appointment")
        call_state.current_state = "booking"
        node = create_booking_node(call_state)
        assert node["name"] == "booking"


class TestEscalationFlow:
    """Simulate emergency escalation."""

    @pytest.fixture
    def call_state(self):
        state = get_or_create_call_state("integration-escalate-001")
        yield state
        remove_call_state("integration-escalate-001")

    def test_emergency_escalation(self, call_state):
        call_state.entities.update(intent="emergency", urgency="emergency")
        call_state.current_state = "escalate"

        node = create_escalate_node(call_state)
        assert node["name"] == "escalate"
        assert "transfer_to_human" in [f.name for f in node["functions"]]


class TestContextIsolation:
    """Verify that multiple concurrent calls don't leak data."""

    def test_two_calls_isolated(self):
        state1 = get_or_create_call_state("iso-call-001")
        state2 = get_or_create_call_state("iso-call-002")

        state1.entities.update(caller_name="Alice", service_type="cleaning")
        state2.entities.update(caller_name="Bob", service_type="root_canal")

        assert state1.entities.caller_name == "Alice"
        assert state2.entities.caller_name == "Bob"
        assert state1.entities.service_type == "cleaning"
        assert state2.entities.service_type == "root_canal"

        # Prompts should differ
        prompt1 = get_greeting_prompt(state1)
        prompt2 = get_greeting_prompt(state2)
        assert "Alice" in prompt1
        assert "Bob" in prompt2
        assert "Alice" not in prompt2
        assert "Bob" not in prompt1

        remove_call_state("iso-call-001")
        remove_call_state("iso-call-002")


class TestContextBuilding:
    """Test that context is assembled correctly across all layers."""

    def test_context_with_all_layers(self):
        state = get_or_create_call_state("ctx-test-001")

        # Layer 1: Entities
        state.entities.update(
            caller_name="Sarah",
            intent="new_appointment",
            service_type="cleaning",
        )

        # Layer 2: Add turns
        state.add_turn("user", "Hi, I need a cleaning")
        state.add_turn("assistant", "I'd be happy to help! When works for you?")
        state.add_turn("user", "Next Tuesday afternoon")
        state.add_turn("assistant", "Let me check availability for Tuesday afternoon.")

        # Layer 3: Summary (simulate)
        state.summary = "Patient Sarah wants to book a cleaning appointment."

        # Layer 4: Build context
        block = state.build_context_block()
        assert "KNOWN FACTS" in block
        assert "Sarah" in block
        assert "CONVERSATION SO FAR" in block
        assert "cleaning" in block

        # Sliding window should have all 4 turns (under limit)
        msgs = state.get_context_messages()
        assert len(msgs) == 4

        remove_call_state("ctx-test-001")

    def test_context_trimming_with_many_turns(self):
        state = get_or_create_call_state("ctx-trim-001")

        # Add 12 turns (24 messages) — well over the 4-turn window
        for i in range(12):
            state.add_turn("user", f"User message {i}")
            state.add_turn("assistant", f"Bot reply {i}")

        # Sliding window should only have last 4 turns = 8 messages
        msgs = state.get_context_messages()
        assert len(msgs) == 8
        assert msgs[0]["content"] == "User message 8"  # First message in window

        remove_call_state("ctx-trim-001")
