"""
Tests for tool handler functions (Phase 1: mock data).
"""

import json
import pytest

from server.bot.context import CallState
from server.bot.tools import (
    check_availability_handler,
    book_appointment_handler,
    cancel_appointment_handler,
    get_clinic_info_handler,
    transfer_to_human_handler,
    end_call_handler,
)


@pytest.fixture
def call_state():
    return CallState(call_sid="test-call-001")


class TestCheckAvailability:
    @pytest.mark.asyncio
    async def test_returns_slots(self, call_state):
        result = json.loads(await check_availability_handler(
            {"date": "2026-03-25", "service_type": "cleaning"},
            call_state,
        ))
        assert "available_slots" in result
        assert len(result["available_slots"]) > 0

    @pytest.mark.asyncio
    async def test_slots_have_required_fields(self, call_state):
        result = json.loads(await check_availability_handler(
            {"date": "2026-03-25", "service_type": "filling"},
            call_state,
        ))
        slot = result["available_slots"][0]
        assert "date" in slot
        assert "time" in slot
        assert "provider" in slot
        assert "service" in slot
        assert "duration_minutes" in slot

    @pytest.mark.asyncio
    async def test_filters_by_provider(self, call_state):
        result = json.loads(await check_availability_handler(
            {"date": "2026-03-25", "service_type": "cleaning", "provider": "Dr. Chen"},
            call_state,
        ))
        for slot in result["available_slots"]:
            assert "Chen" in slot["provider"]

    @pytest.mark.asyncio
    async def test_morning_preference(self, call_state):
        result = json.loads(await check_availability_handler(
            {"date": "2026-03-25", "service_type": "cleaning", "time_preference": "morning"},
            call_state,
        ))
        for slot in result["available_slots"]:
            hour = int(slot["time"].split(":")[0])
            assert hour < 12

    @pytest.mark.asyncio
    async def test_logs_tool_call(self, call_state):
        await check_availability_handler(
            {"date": "2026-03-25", "service_type": "cleaning"},
            call_state,
        )
        assert len(call_state.tool_calls) == 1
        assert call_state.tool_calls[0]["name"] == "check_availability"


class TestBookAppointment:
    @pytest.mark.asyncio
    async def test_successful_booking(self, call_state):
        result = json.loads(await book_appointment_handler(
            {
                "patient_name": "John Smith",
                "date": "2026-03-25",
                "time": "14:00",
                "service_type": "cleaning",
                "provider": "Dr. Sarah Chen",
                "patient_phone": "555-0123",
            },
            call_state,
        ))
        assert result["success"] is True
        assert "appointment_id" in result
        assert result["sms_sent"] is True

    @pytest.mark.asyncio
    async def test_booking_updates_entities(self, call_state):
        await book_appointment_handler(
            {
                "patient_name": "Sarah Lee",
                "date": "2026-03-26",
                "time": "10:00",
                "service_type": "filling",
                "provider": "Dr. James Park",
            },
            call_state,
        )
        assert call_state.entities.caller_name == "Sarah Lee"
        assert call_state.entities.booking_confirmed is True
        assert call_state.entities.appointment_id is not None

    @pytest.mark.asyncio
    async def test_booking_without_phone(self, call_state):
        result = json.loads(await book_appointment_handler(
            {
                "patient_name": "John",
                "date": "2026-03-25",
                "time": "14:00",
                "service_type": "cleaning",
                "provider": "Dr. Chen",
            },
            call_state,
        ))
        assert result["sms_sent"] is False

    @pytest.mark.asyncio
    async def test_booking_uses_entity_fallbacks(self, call_state):
        call_state.entities.update(
            caller_name="From Entity",
            preferred_date="2026-04-01",
            preferred_time="09:00",
        )
        result = json.loads(await book_appointment_handler(
            {"service_type": "cleaning", "provider": "Dr. Chen"},
            call_state,
        ))
        assert "From Entity" in result["message"]


class TestCancelAppointment:
    @pytest.mark.asyncio
    async def test_cancel_success(self, call_state):
        result = json.loads(await cancel_appointment_handler(
            {"patient_name": "John Smith", "appointment_date": "2026-03-25"},
            call_state,
        ))
        assert result["success"] is True
        assert "cancelled" in result["message"]


class TestGetClinicInfo:
    @pytest.mark.asyncio
    async def test_hours(self, call_state):
        result = json.loads(await get_clinic_info_handler(
            {"question_type": "hours"}, call_state,
        ))
        assert "Monday" in result["info"]

    @pytest.mark.asyncio
    async def test_location(self, call_state):
        result = json.loads(await get_clinic_info_handler(
            {"question_type": "location"}, call_state,
        ))
        assert "123 Main Street" in result["info"]

    @pytest.mark.asyncio
    async def test_insurance(self, call_state):
        result = json.loads(await get_clinic_info_handler(
            {"question_type": "insurance"}, call_state,
        ))
        assert "Delta Dental" in result["info"]

    @pytest.mark.asyncio
    async def test_services(self, call_state):
        result = json.loads(await get_clinic_info_handler(
            {"question_type": "services"}, call_state,
        ))
        assert "cleaning" in result["info"].lower()

    @pytest.mark.asyncio
    async def test_providers(self, call_state):
        result = json.loads(await get_clinic_info_handler(
            {"question_type": "providers"}, call_state,
        ))
        assert "Dr. Sarah Chen" in result["info"]

    @pytest.mark.asyncio
    async def test_unknown_type(self, call_state):
        result = json.loads(await get_clinic_info_handler(
            {"question_type": "nonexistent"}, call_state,
        ))
        assert "don't have" in result["info"]


class TestTransferToHuman:
    @pytest.mark.asyncio
    async def test_transfer(self, call_state):
        result = json.loads(await transfer_to_human_handler(
            {"reason": "dental emergency"}, call_state,
        ))
        assert result["success"] is True


class TestEndCall:
    @pytest.mark.asyncio
    async def test_end_call(self, call_state):
        result = json.loads(await end_call_handler(
            {"summary": "Booked cleaning for Tuesday"}, call_state,
        ))
        assert result["success"] is True
        assert call_state.outcome == "completed"

    @pytest.mark.asyncio
    async def test_end_call_preserves_existing_outcome(self, call_state):
        call_state.outcome = "booked"
        await end_call_handler({"summary": "Done"}, call_state)
        assert call_state.outcome == "booked"  # Not overwritten
