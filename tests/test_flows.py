"""
Tests for the Pipecat Flows state machine.

Tests that node factory functions return valid NodeConfig dicts
with correct names, prompts, and functions.
"""

import pytest

from server.bot.context import CallState
from server.bot.flows import (
    create_greeting_node,
    create_intent_node,
    create_booking_node,
    create_availability_node,
    create_confirm_node,
    create_reschedule_node,
    create_cancel_node,
    create_question_node,
    create_escalate_node,
    create_close_node,
)


@pytest.fixture
def call_state():
    state = CallState(call_sid="test-flow-001")
    state.entities.update(
        caller_name="Sarah",
        service_type="cleaning",
        preferred_date="next Tuesday",
    )
    return state


class TestGreetingNode:
    def test_node_structure(self, call_state):
        node = create_greeting_node(call_state)
        assert node["name"] == "greeting"
        assert "role_message" in node
        assert "task_messages" in node
        assert "functions" in node

    def test_has_greeting_complete_function(self, call_state):
        node = create_greeting_node(call_state)
        func_names = [f.name for f in node["functions"]]
        assert "greeting_complete" in func_names

    def test_role_message_has_clinic_info(self, call_state):
        node = create_greeting_node(call_state)
        role = node["role_message"]
        assert "Bright Smile Dental" in role
        assert "Monday" in role
        assert "Delta Dental" in role


class TestIntentNode:
    def test_node_structure(self, call_state):
        node = create_intent_node(call_state)
        assert node["name"] == "identify_intent"

    def test_has_all_routing_functions(self, call_state):
        node = create_intent_node(call_state)
        func_names = [f.name for f in node["functions"]]
        assert "route_new_appointment" in func_names
        assert "route_reschedule" in func_names
        assert "route_cancel" in func_names
        assert "route_question" in func_names
        assert "route_emergency" in func_names

    def test_has_five_routes(self, call_state):
        node = create_intent_node(call_state)
        assert len(node["functions"]) == 5


class TestBookingNode:
    def test_node_structure(self, call_state):
        node = create_booking_node(call_state)
        assert node["name"] == "booking"

    def test_has_ready_function(self, call_state):
        node = create_booking_node(call_state)
        func_names = [f.name for f in node["functions"]]
        assert "ready_to_check_availability" in func_names


class TestAvailabilityNode:
    def test_node_structure(self, call_state):
        node = create_availability_node(call_state)
        assert node["name"] == "check_availability"

    def test_has_check_and_select_functions(self, call_state):
        node = create_availability_node(call_state)
        func_names = [f.name for f in node["functions"]]
        assert "check_availability" in func_names
        assert "slot_selected" in func_names

    def test_prompt_includes_patient_info(self, call_state):
        node = create_availability_node(call_state)
        prompt = node["task_messages"][0]["content"]
        assert "Sarah" in prompt
        assert "cleaning" in prompt


class TestConfirmNode:
    def test_node_structure(self, call_state):
        call_state.entities.update(provider_preference="Dr. Chen")
        node = create_confirm_node(call_state)
        assert node["name"] == "confirm_booking"

    def test_has_book_and_change_functions(self, call_state):
        node = create_confirm_node(call_state)
        func_names = [f.name for f in node["functions"]]
        assert "book_appointment" in func_names
        assert "change_time" in func_names


class TestRescheduleNode:
    def test_node_structure(self, call_state):
        node = create_reschedule_node(call_state)
        assert node["name"] == "reschedule"

    def test_has_cancel_function(self, call_state):
        node = create_reschedule_node(call_state)
        func_names = [f.name for f in node["functions"]]
        assert "cancel_old_appointment" in func_names


class TestCancelNode:
    def test_node_structure(self, call_state):
        node = create_cancel_node(call_state)
        assert node["name"] == "cancel"

    def test_has_cancel_function(self, call_state):
        node = create_cancel_node(call_state)
        func_names = [f.name for f in node["functions"]]
        assert "cancel_appointment" in func_names


class TestQuestionNode:
    def test_node_structure(self, call_state):
        node = create_question_node(call_state)
        assert node["name"] == "general_question"

    def test_has_info_done_and_book_functions(self, call_state):
        node = create_question_node(call_state)
        func_names = [f.name for f in node["functions"]]
        assert "get_clinic_info" in func_names
        assert "questions_done" in func_names
        assert "wants_appointment" in func_names


class TestEscalateNode:
    def test_node_structure(self, call_state):
        node = create_escalate_node(call_state)
        assert node["name"] == "escalate"

    def test_has_transfer_function(self, call_state):
        node = create_escalate_node(call_state)
        func_names = [f.name for f in node["functions"]]
        assert "transfer_to_human" in func_names


class TestCloseNode:
    def test_node_structure(self, call_state):
        node = create_close_node(call_state)
        assert node["name"] == "close"

    def test_has_end_and_more_functions(self, call_state):
        node = create_close_node(call_state)
        func_names = [f.name for f in node["functions"]]
        assert "end_call" in func_names
        assert "patient_has_more" in func_names


class TestStateTransitions:
    """Test that state transitions update CallState correctly."""

    def test_greeting_to_intent(self, call_state):
        assert call_state.current_state == "greeting"
        # Simulate transition
        call_state.current_state = "identify_intent"
        node = create_intent_node(call_state)
        assert node["name"] == "identify_intent"

    def test_full_booking_flow_states(self, call_state):
        """Verify state names through a complete booking flow."""
        states = ["greeting", "identify_intent", "booking",
                  "check_availability", "confirm_booking", "close"]
        node_creators = [
            create_greeting_node,
            create_intent_node,
            create_booking_node,
            create_availability_node,
            create_confirm_node,
            create_close_node,
        ]
        for expected_name, creator in zip(states, node_creators):
            call_state.current_state = expected_name
            node = creator(call_state)
            assert node["name"] == expected_name


class TestPromptDynamism:
    """Test that prompts change based on call state."""

    def test_greeting_includes_context_when_present(self, call_state):
        # call_state fixture already has entities set
        node = create_greeting_node(call_state)
        role = node["role_message"]
        assert "KNOWN FACTS" in role
        assert "Sarah" in role

    def test_availability_prompt_changes_with_entities(self, call_state):
        node1 = create_availability_node(call_state)
        prompt1 = node1["task_messages"][0]["content"]

        call_state.entities.update(preferred_time="afternoon")
        node2 = create_availability_node(call_state)
        prompt2 = node2["task_messages"][0]["content"]

        assert "afternoon" not in prompt1
        assert "afternoon" in prompt2
