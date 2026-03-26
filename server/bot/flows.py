"""
Pipecat Flows state machine for CallGenius dental receptionist.

State graph:
  GREETING → IDENTIFY_INTENT → CHECK_AVAILABILITY → CONFIRM_BOOKING → CLOSE
                              → RESCHEDULE        → CONFIRM_BOOKING → CLOSE
                              → CANCEL                              → CLOSE
                              → GENERAL_QUESTION                    → CLOSE
                              → ESCALATE          → TRANSFER

Each node has:
  - Its own focused system prompt (shorter = faster LLM response)
  - Allowed tools (only what's relevant to that state)
  - Transition rules via function call handlers
"""

from pipecat_flows import FlowManager, FlowArgs, FlowResult, NodeConfig, FlowsFunctionSchema

from server.bot.context import CallState, get_call_state
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
from server.utils.logger import logger


# =============================================================================
# Flow Result Types
# =============================================================================

class IntentResult(FlowResult):
    intent: str


class AvailabilityResult(FlowResult):
    slots: list


class BookingResult(FlowResult):
    appointment_id: str
    message: str


# =============================================================================
# Node Factory Functions
# =============================================================================

def create_greeting_node(call_state: CallState) -> NodeConfig:
    """Initial greeting state. AI says hello, asks how it can help."""

    async def on_greeting_complete(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        """Transition to intent identification after greeting."""
        call_state.current_state = "identify_intent"
        logger.info(f"Call {call_state.call_sid}: greeting → identify_intent")
        return None, create_intent_node(call_state)

    return NodeConfig(
        name="greeting",
        role_message=get_greeting_prompt(call_state),
        task_messages=[{
            "role": "system",
            "content": (
                "Greet the caller warmly. Introduce yourself as the front desk. "
                "Ask how you can help today. Keep it to 1-2 short sentences. "
                "Once the caller responds with their need, call greeting_complete."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="greeting_complete",
                handler=on_greeting_complete,
                description="Call this after the patient states their reason for calling.",
                properties={
                    "caller_statement": {
                        "type": "string",
                        "description": "What the caller said they need",
                    },
                },
                required=["caller_statement"],
            ),
        ],
    )


def create_intent_node(call_state: CallState) -> NodeConfig:
    """Identify what the caller needs and route to the right state."""

    async def route_to_booking(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        call_state.entities.update(intent="new_appointment")
        call_state.current_state = "booking"
        logger.info(f"Call {call_state.call_sid}: intent → booking")
        return None, create_booking_node(call_state)

    async def route_to_reschedule(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        call_state.entities.update(intent="reschedule")
        call_state.current_state = "reschedule"
        logger.info(f"Call {call_state.call_sid}: intent → reschedule")
        return None, create_reschedule_node(call_state)

    async def route_to_cancel(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        call_state.entities.update(intent="cancel")
        call_state.current_state = "cancel"
        logger.info(f"Call {call_state.call_sid}: intent → cancel")
        return None, create_cancel_node(call_state)

    async def route_to_question(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        call_state.entities.update(intent="question")
        call_state.current_state = "general_question"
        logger.info(f"Call {call_state.call_sid}: intent → general_question")
        return None, create_question_node(call_state)

    async def route_to_emergency(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        call_state.entities.update(intent="emergency", urgency="emergency")
        call_state.current_state = "escalate"
        logger.info(f"Call {call_state.call_sid}: intent → escalate (emergency)")
        return None, create_escalate_node(call_state)

    return NodeConfig(
        name="identify_intent",
        task_messages=[{
            "role": "system",
            "content": get_intent_prompt(call_state),
        }],
        functions=[
            FlowsFunctionSchema(
                name="route_new_appointment",
                handler=route_to_booking,
                description="Patient wants to book a new appointment.",
                properties={
                    "service_type": {
                        "type": "string",
                        "description": "What service they need (cleaning, filling, crown, etc.)",
                    },
                    "caller_name": {
                        "type": "string",
                        "description": "Caller's name if mentioned",
                    },
                },
                required=[],
            ),
            FlowsFunctionSchema(
                name="route_reschedule",
                handler=route_to_reschedule,
                description="Patient wants to reschedule an existing appointment.",
                properties={
                    "existing_date": {
                        "type": "string",
                        "description": "Their current appointment date if mentioned",
                    },
                },
                required=[],
            ),
            FlowsFunctionSchema(
                name="route_cancel",
                handler=route_to_cancel,
                description="Patient wants to cancel an appointment.",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="route_question",
                handler=route_to_question,
                description="Patient has a question about hours, location, insurance, services, or providers.",
                properties={
                    "question_topic": {
                        "type": "string",
                        "description": "What they're asking about",
                    },
                },
                required=[],
            ),
            FlowsFunctionSchema(
                name="route_emergency",
                handler=route_to_emergency,
                description="Patient has a dental emergency (severe pain, broken tooth, bleeding).",
                properties={
                    "description": {
                        "type": "string",
                        "description": "Nature of the emergency",
                    },
                },
                required=[],
            ),
        ],
    )


def create_booking_node(call_state: CallState) -> NodeConfig:
    """Gather info needed for booking: name, service, date/time preference."""

    async def ready_to_check(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        # Update entities with whatever was collected
        call_state.entities.update(
            caller_name=args.get("patient_name"),
            service_type=args.get("service_type"),
            preferred_date=args.get("preferred_date"),
            preferred_time=args.get("preferred_time"),
            patient_phone=args.get("patient_phone"),
        )
        call_state.current_state = "check_availability"
        logger.info(f"Call {call_state.call_sid}: booking → check_availability")
        return None, create_availability_node(call_state)

    return NodeConfig(
        name="booking",
        task_messages=[{
            "role": "system",
            "content": get_booking_prompt(call_state),
        }],
        functions=[
            FlowsFunctionSchema(
                name="ready_to_check_availability",
                handler=ready_to_check,
                description=(
                    "Call when you have enough info to check the schedule. "
                    "Need at minimum: patient name and service type."
                ),
                properties={
                    "patient_name": {
                        "type": "string",
                        "description": "Patient's full name",
                    },
                    "service_type": {
                        "type": "string",
                        "description": "Service needed (cleaning, filling, crown, etc.)",
                    },
                    "preferred_date": {
                        "type": "string",
                        "description": "Preferred date (e.g. 'next Tuesday', '2026-03-30')",
                    },
                    "preferred_time": {
                        "type": "string",
                        "description": "Preferred time (e.g. 'morning', 'afternoon', '2pm')",
                    },
                    "patient_phone": {
                        "type": "string",
                        "description": "Patient phone number for SMS confirmation",
                    },
                },
                required=["patient_name", "service_type"],
            ),
        ],
    )


def create_availability_node(call_state: CallState) -> NodeConfig:
    """Check calendar and present available slots to the patient."""
    from server.bot.tools import check_availability_handler

    async def on_slot_selected(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        call_state.entities.update(
            preferred_date=args.get("selected_date"),
            preferred_time=args.get("selected_time"),
            provider_preference=args.get("selected_provider"),
        )
        call_state.current_state = "confirm_booking"
        logger.info(f"Call {call_state.call_sid}: availability → confirm_booking")
        return None, create_confirm_node(call_state)

    async def check_avail(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[str, None]:
        """Check calendar — stay on this node with results."""
        result = await check_availability_handler(args, call_state)
        return result, None

    return NodeConfig(
        name="check_availability",
        task_messages=[{
            "role": "system",
            "content": get_availability_prompt(call_state),
        }],
        functions=[
            FlowsFunctionSchema(
                name="check_availability",
                handler=check_avail,
                description="Check available appointment slots for a date and service.",
                properties={
                    "date": {
                        "type": "string",
                        "description": "Date to check (e.g. '2026-03-25' or 'next Tuesday')",
                    },
                    "service_type": {
                        "type": "string",
                        "description": "Service type",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Specific provider if requested",
                    },
                    "time_preference": {
                        "type": "string",
                        "description": "morning, afternoon, or specific time",
                    },
                },
                required=["date", "service_type"],
            ),
            FlowsFunctionSchema(
                name="slot_selected",
                handler=on_slot_selected,
                description="Patient chose a specific time slot. Move to confirmation.",
                properties={
                    "selected_date": {"type": "string", "description": "The date chosen"},
                    "selected_time": {"type": "string", "description": "The time chosen"},
                    "selected_provider": {
                        "type": "string",
                        "description": "The provider for this slot",
                    },
                },
                required=["selected_date", "selected_time", "selected_provider"],
            ),
        ],
    )


def create_confirm_node(call_state: CallState) -> NodeConfig:
    """Read back appointment details and get confirmation before booking."""
    from server.bot.tools import book_appointment_handler

    async def confirm_and_book(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[str, NodeConfig]:
        result = await book_appointment_handler(args, call_state)
        call_state.entities.update(booking_confirmed=True)
        call_state.outcome = "booked"
        call_state.current_state = "close"
        logger.info(f"Call {call_state.call_sid}: confirm → close (booked)")
        return result, create_close_node(call_state)

    async def go_back_to_availability(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        call_state.current_state = "check_availability"
        return None, create_availability_node(call_state)

    return NodeConfig(
        name="confirm_booking",
        task_messages=[{
            "role": "system",
            "content": get_confirm_prompt(call_state),
        }],
        functions=[
            FlowsFunctionSchema(
                name="book_appointment",
                handler=confirm_and_book,
                description=(
                    "Book the appointment after patient confirms. "
                    "Only call this when the patient says yes/confirmed."
                ),
                properties={
                    "patient_name": {"type": "string", "description": "Patient name"},
                    "patient_phone": {"type": "string", "description": "Phone for SMS"},
                    "date": {"type": "string", "description": "Appointment date"},
                    "time": {"type": "string", "description": "Appointment time"},
                    "service_type": {"type": "string", "description": "Service type"},
                    "provider": {"type": "string", "description": "Provider name"},
                },
                required=["patient_name", "date", "time", "service_type", "provider"],
            ),
            FlowsFunctionSchema(
                name="change_time",
                handler=go_back_to_availability,
                description="Patient wants a different time. Go back to checking availability.",
                properties={},
                required=[],
            ),
        ],
    )


def create_reschedule_node(call_state: CallState) -> NodeConfig:
    """Handle rescheduling — cancel old, book new."""
    from server.bot.tools import cancel_appointment_handler

    async def old_cancelled_check_new(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[str, NodeConfig]:
        result = await cancel_appointment_handler(args, call_state)
        call_state.current_state = "check_availability"
        logger.info(f"Call {call_state.call_sid}: reschedule → check_availability")
        return result, create_availability_node(call_state)

    return NodeConfig(
        name="reschedule",
        task_messages=[{
            "role": "system",
            "content": get_reschedule_prompt(call_state),
        }],
        functions=[
            FlowsFunctionSchema(
                name="cancel_old_appointment",
                handler=old_cancelled_check_new,
                description="Cancel the existing appointment to reschedule.",
                properties={
                    "patient_name": {"type": "string", "description": "Patient name"},
                    "appointment_date": {
                        "type": "string",
                        "description": "Date of appointment to cancel",
                    },
                },
                required=["patient_name", "appointment_date"],
            ),
        ],
    )


def create_cancel_node(call_state: CallState) -> NodeConfig:
    """Handle appointment cancellation."""
    from server.bot.tools import cancel_appointment_handler

    async def on_cancelled(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[str, NodeConfig]:
        result = await cancel_appointment_handler(args, call_state)
        call_state.outcome = "cancelled"
        call_state.current_state = "close"
        logger.info(f"Call {call_state.call_sid}: cancel → close")
        return result, create_close_node(call_state)

    return NodeConfig(
        name="cancel",
        task_messages=[{
            "role": "system",
            "content": get_cancel_prompt(call_state),
        }],
        functions=[
            FlowsFunctionSchema(
                name="cancel_appointment",
                handler=on_cancelled,
                description="Cancel the appointment after patient confirms.",
                properties={
                    "patient_name": {"type": "string", "description": "Patient name"},
                    "appointment_date": {
                        "type": "string",
                        "description": "Date of appointment to cancel",
                    },
                },
                required=["patient_name", "appointment_date"],
            ),
        ],
    )


def create_question_node(call_state: CallState) -> NodeConfig:
    """Answer general questions about the clinic."""
    from server.bot.tools import get_clinic_info_handler

    async def answer_question(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[str, None]:
        """Answer and stay on this node (patient might have more questions)."""
        result = await get_clinic_info_handler(args, call_state)
        return result, None

    async def done_with_questions(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        call_state.outcome = "info"
        call_state.current_state = "close"
        return None, create_close_node(call_state)

    async def wants_to_book(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        call_state.entities.update(intent="new_appointment")
        call_state.current_state = "booking"
        return None, create_booking_node(call_state)

    return NodeConfig(
        name="general_question",
        task_messages=[{
            "role": "system",
            "content": get_question_prompt(call_state),
        }],
        functions=[
            FlowsFunctionSchema(
                name="get_clinic_info",
                handler=answer_question,
                description="Look up clinic information to answer a question.",
                properties={
                    "question_type": {
                        "type": "string",
                        "enum": ["hours", "location", "insurance", "services", "providers"],
                        "description": "Type of information needed",
                    },
                },
                required=["question_type"],
            ),
            FlowsFunctionSchema(
                name="questions_done",
                handler=done_with_questions,
                description="Patient has no more questions. Wrap up the call.",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="wants_appointment",
                handler=wants_to_book,
                description="Patient decided they want to book an appointment.",
                properties={},
                required=[],
            ),
        ],
    )


def create_escalate_node(call_state: CallState) -> NodeConfig:
    """Transfer call to human staff."""
    from server.bot.tools import transfer_to_human_handler

    async def do_transfer(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[str, NodeConfig]:
        result = await transfer_to_human_handler(args, call_state)
        call_state.outcome = "transferred"
        call_state.current_state = "close"
        logger.info(f"Call {call_state.call_sid}: escalate → transfer")
        return result, create_close_node(call_state)

    return NodeConfig(
        name="escalate",
        task_messages=[{
            "role": "system",
            "content": get_escalate_prompt(call_state),
        }],
        functions=[
            FlowsFunctionSchema(
                name="transfer_to_human",
                handler=do_transfer,
                description="Transfer the call to clinic staff.",
                properties={
                    "reason": {
                        "type": "string",
                        "description": "Why the call is being transferred",
                    },
                },
                required=["reason"],
            ),
        ],
    )


def create_close_node(call_state: CallState) -> NodeConfig:
    """Wrap up the call — ask if anything else, say goodbye."""
    from server.bot.tools import end_call_handler

    async def do_end_call(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[str, None]:
        result = await end_call_handler(args, call_state)
        return result, None

    async def has_more_questions(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        """Patient has another request — go back to intent identification."""
        call_state.current_state = "identify_intent"
        return None, create_intent_node(call_state)

    return NodeConfig(
        name="close",
        task_messages=[{
            "role": "system",
            "content": get_close_prompt(call_state),
        }],
        functions=[
            FlowsFunctionSchema(
                name="end_call",
                handler=do_end_call,
                description="End the call. Use after patient confirms no more questions.",
                properties={
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was accomplished",
                    },
                },
                required=["summary"],
            ),
            FlowsFunctionSchema(
                name="patient_has_more",
                handler=has_more_questions,
                description="Patient has another question or request.",
                properties={},
                required=[],
            ),
        ],
        post_actions=[{"type": "end_conversation"}],
    )
