"""
Function calling tool handlers for the dental receptionist bot.

These are called by the Pipecat Flows state machine.
Phase 1: Mock data. Phase 2: Real Google Calendar, Twilio SMS, DB.
"""

import json
from datetime import datetime, timedelta

from server.bot.context import CallState
from server.utils.logger import logger


# =============================================================================
# Tool Handlers (called from flows.py)
# =============================================================================

async def check_availability_handler(args: dict, call_state: CallState) -> str:
    """Check available appointment slots. Phase 1: mock data."""
    service = args.get("service_type", "cleaning")
    provider = args.get("provider")
    date_str = args.get("date", "")
    time_pref = args.get("time_preference")

    logger.info(f"check_availability: service={service}, date={date_str}, provider={provider}")
    call_state.add_tool_call("check_availability", args, "mock_slots")

    # Mock: generate realistic slots
    base_date = datetime.now() + timedelta(days=1)
    slots = []
    providers = ["Dr. Sarah Chen", "Dr. James Park"]

    for i in range(3):
        day = base_date + timedelta(days=i)
        if day.weekday() == 6:  # Skip Sunday
            day += timedelta(days=1)

        for p in providers:
            if provider and provider.lower() not in p.lower():
                continue

            hours = [9, 11, 14, 15]
            if time_pref:
                if "morning" in time_pref.lower():
                    hours = [9, 10, 11]
                elif "afternoon" in time_pref.lower():
                    hours = [13, 14, 15, 16]

            for hour in hours:
                slots.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "day_name": day.strftime("%A"),
                    "time": f"{hour:02d}:00",
                    "provider": p,
                    "service": service,
                    "duration_minutes": 60 if service == "cleaning" else 45,
                })

    result_slots = slots[:4]
    return json.dumps({
        "available_slots": result_slots,
        "message": f"Found {len(result_slots)} available slots for {service}.",
    })


async def book_appointment_handler(args: dict, call_state: CallState) -> str:
    """Book an appointment. Phase 1: mock booking."""
    patient_name = args.get("patient_name", call_state.entities.caller_name or "Patient")
    date = args.get("date", call_state.entities.preferred_date or "TBD")
    time = args.get("time", call_state.entities.preferred_time or "TBD")
    provider = args.get("provider", call_state.entities.provider_preference or "TBD")
    service = args.get("service_type", call_state.entities.service_type or "appointment")
    phone = args.get("patient_phone", call_state.entities.patient_phone)

    logger.info(
        f"BOOKING: {patient_name} | {service} | {date} at {time} | {provider}"
    )
    call_state.add_tool_call("book_appointment", args, "mock_booked")

    # Update entities
    call_state.entities.update(
        caller_name=patient_name,
        preferred_date=date,
        preferred_time=time,
        provider_preference=provider,
        service_type=service,
        booking_confirmed=True,
        appointment_id="mock-appt-12345",
    )

    result = {
        "success": True,
        "appointment_id": "mock-appt-12345",
        "message": (
            f"Appointment booked for {patient_name}: "
            f"{service} on {date} at {time} with {provider}."
        ),
    }

    if phone:
        result["sms_sent"] = True
        result["sms_message"] = (
            f"Confirmed: {service} at Bright Smile Dental on {date} at {time} "
            f"with {provider}. Reply CANCEL to cancel."
        )
    else:
        result["sms_sent"] = False
        result["sms_note"] = "No phone number provided for SMS confirmation."

    return json.dumps(result)


async def cancel_appointment_handler(args: dict, call_state: CallState) -> str:
    """Cancel an appointment. Phase 1: mock cancellation."""
    patient_name = args.get("patient_name", call_state.entities.caller_name or "Patient")
    date = args.get("appointment_date", "unknown date")

    logger.info(f"CANCEL: {patient_name} on {date}")
    call_state.add_tool_call("cancel_appointment", args, "mock_cancelled")

    return json.dumps({
        "success": True,
        "message": f"Appointment for {patient_name} on {date} has been cancelled.",
    })


async def get_clinic_info_handler(args: dict, call_state: CallState) -> str:
    """Return clinic information. Phase 1: hardcoded."""
    question_type = args.get("question_type", "hours")

    logger.info(f"clinic_info: {question_type}")
    call_state.add_tool_call("get_clinic_info", args, question_type)

    info = {
        "hours": (
            "We're open Monday through Friday, 8 AM to 5 PM, "
            "and Saturday 9 AM to 1 PM. We're closed on Sundays."
        ),
        "location": (
            "We're at 123 Main Street, Suite 100. Second floor, "
            "right above the pharmacy. Free parking in the lot behind the building."
        ),
        "insurance": (
            "We accept Delta Dental, Cigna, Aetna, MetLife, Guardian, "
            "and United Healthcare. For other plans, we recommend "
            "calling your provider about out-of-network benefits."
        ),
        "services": (
            "We offer cleanings, fillings, crowns, root canals, "
            "teeth whitening, and emergency exams. For cosmetic procedures "
            "like veneers or implants, we can provide a referral."
        ),
        "providers": (
            "Dr. Sarah Chen sees patients Monday, Wednesday, and Friday. "
            "Dr. James Park is available Tuesday, Thursday, and Saturday morning."
        ),
    }

    return json.dumps({
        "info": info.get(question_type, "I don't have that information available."),
    })


async def transfer_to_human_handler(args: dict, call_state: CallState) -> str:
    """Transfer call to human staff. Phase 1: mock transfer."""
    reason = args.get("reason", "patient request")

    logger.info(f"TRANSFER: reason={reason}")
    call_state.add_tool_call("transfer_to_human", args, "mock_transfer")

    return json.dumps({
        "success": True,
        "message": "Transferring the call now. The clinic staff will be with you shortly.",
    })


async def end_call_handler(args: dict, call_state: CallState) -> str:
    """End the call and log summary. Phase 1: mock."""
    summary = args.get("summary", "Call completed")

    logger.info(f"CALL END: {summary}")
    logger.info(f"CALL SUMMARY: {call_state.get_call_summary()}")
    call_state.add_tool_call("end_call", args, "ended")

    if not call_state.outcome:
        call_state.outcome = "completed"

    return json.dumps({
        "success": True,
        "message": "Call ended.",
        "summary": summary,
    })
