"""
4-Layer Context Management System for CallGenius Voice Agent.

Layer 1: Entity Store — structured facts extracted via inline function calling
Layer 2: Sliding Window — last 4 turns verbatim for natural conversation flow
Layer 3: Auto-Summarizer — compresses older turns to keep token costs flat
Layer 4: Context Builder — assembles final prompt from all layers

Design based on patterns from Vapi, Retell AI, and LiveKit Agents research.
Key insight: entity extraction happens inline (same LLM call), NOT via sidecar.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from openai import AsyncOpenAI

from server.config import get_settings
from server.utils.logger import logger

# --- Configuration ---
MAX_WINDOW_TURNS = 4  # Keep last 4 exchanges (8 messages) verbatim
SUMMARIZE_THRESHOLD = 6  # Start summarizing after 6 total turns
MAX_CONTEXT_TOKENS = 2000  # Hard cap for conversation history portion
SUMMARY_MODEL = "gpt-4o-mini"  # Fast, cheap model for compression


# =============================================================================
# Layer 1: Entity Store
# =============================================================================

@dataclass
class CallEntities:
    """Structured entities extracted during a call.

    Updated via the `update_entities` function call — GPT-4o extracts these
    inline as part of its normal response (zero additional latency/cost).
    Persists across all state transitions within a single call.
    """

    caller_name: Optional[str] = None
    intent: Optional[str] = None  # new_appointment, reschedule, cancel, question, emergency
    service_type: Optional[str] = None  # cleaning, filling, crown, root_canal, etc.
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    provider_preference: Optional[str] = None
    insurance: Optional[str] = None
    patient_phone: Optional[str] = None
    existing_appointment: Optional[str] = None
    urgency: str = "normal"  # normal, urgent, emergency
    sentiment: str = "neutral"  # positive, neutral, frustrated, angry
    booking_confirmed: bool = False
    appointment_id: Optional[str] = None

    def update(self, **kwargs) -> list[str]:
        """Update entities from a dict. Returns list of fields that changed."""
        changed = []
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                old = getattr(self, key)
                if old != value:
                    setattr(self, key, value)
                    changed.append(key)
        return changed

    def to_context_block(self) -> str:
        """Render as a compact block for injection into the LLM prompt.

        Only includes non-default, non-None values to save tokens.
        """
        lines = []
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if key == "urgency" and value == "normal":
                continue
            if key == "sentiment" and value == "neutral":
                continue
            if key == "booking_confirmed" and not value:
                continue
            lines.append(f"  {key}: {value}")

        if not lines:
            return ""
        return "KNOWN FACTS ABOUT THIS CALLER:\n" + "\n".join(lines)

    def to_dict(self) -> dict:
        """Export all non-None values as a dict."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# =============================================================================
# Layer 2: Sliding Window
# =============================================================================

def get_sliding_window(turns: list[dict]) -> list[dict]:
    """Return the last N turns as OpenAI message-format dicts.

    Each 'turn' is a user+assistant pair = 2 messages.
    We keep MAX_WINDOW_TURNS pairs = MAX_WINDOW_TURNS * 2 messages.
    """
    max_messages = MAX_WINDOW_TURNS * 2
    if len(turns) <= max_messages:
        return turns
    return turns[-max_messages:]


# =============================================================================
# Layer 3: Auto-Summarizer
# =============================================================================

_openai_client: Optional[AsyncOpenAI] = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        settings = get_settings()
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


async def summarize_turns(
    old_turns: list[dict],
    existing_summary: Optional[str] = None,
) -> str:
    """Compress old conversation turns into a 2-3 sentence summary.

    Uses GPT-4o-mini for speed (~100ms) and low cost.
    Called async — does NOT block the main response pipeline.
    """
    if not old_turns:
        return existing_summary or ""

    turn_text = "\n".join(
        f"{t['role']}: {t['content']}" for t in old_turns if t.get("content")
    )

    prompt = "Summarize this dental clinic phone conversation in 2-3 sentences. "
    prompt += "Focus on: what the patient needs, decisions made, and any pending questions.\n\n"
    if existing_summary:
        prompt += f"Previous summary: {existing_summary}\n\n"
    prompt += f"New conversation:\n{turn_text}"

    try:
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return existing_summary or ""


# =============================================================================
# Layer 4: Context Builder + Call State
# =============================================================================

@dataclass
class CallState:
    """Full state for a single call session.

    Holds all 4 layers and assembles the final prompt on each turn.
    """

    call_sid: str
    clinic_id: Optional[int] = None

    # Layer 1: Entity Store
    entities: CallEntities = field(default_factory=CallEntities)

    # Layer 2+3: Conversation tracking
    all_turns: list[dict] = field(default_factory=list)  # Full history
    summary: Optional[str] = None  # Compressed old turns

    # State machine
    current_state: str = "greeting"

    # Tracking
    tool_calls: list[dict] = field(default_factory=list)
    outcome: Optional[str] = None  # booked, rescheduled, cancelled, transferred, info
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Phase 2: Database + multi-tenant fields
    db_session: Optional[object] = field(default=None, repr=False)
    clinic_config: dict = field(default_factory=dict)
    caller_phone: Optional[str] = None
    called_number: Optional[str] = None
    recording_url: Optional[str] = None
    clinic_timezone: str = "America/Los_Angeles"

    # Async summarization
    _summarize_task: Optional[asyncio.Task] = field(default=None, repr=False)

    def add_turn(self, role: str, content: str):
        """Record a conversation turn and trigger summarization if needed."""
        self.all_turns.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        # Trigger async summarization if we've crossed the threshold
        self._maybe_start_summarization()

    def add_tool_call(self, name: str, arguments: dict, result: str):
        """Log a tool call for analytics."""
        self.tool_calls.append({
            "name": name,
            "arguments": arguments,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

    @property
    def turn_count(self) -> int:
        """Number of user turns (not messages)."""
        return sum(1 for t in self.all_turns if t["role"] == "user")

    def get_context_messages(self) -> list[dict]:
        """Get conversation messages for the LLM context.

        Returns only the sliding window portion — the system prompt
        and entity block are added by build_prompt().
        """
        return get_sliding_window(self.all_turns)

    def build_context_block(self) -> str:
        """Build the dynamic context section injected after the system prompt.

        Combines: entity store + summary + (sliding window is separate in messages).
        """
        parts = []

        # Layer 1: Entity store
        entity_block = self.entities.to_context_block()
        if entity_block:
            parts.append(entity_block)

        # Layer 3: Summary of older turns
        if self.summary:
            parts.append(f"CONVERSATION SO FAR:\n  {self.summary}")

        return "\n\n".join(parts)

    def get_call_summary(self) -> str:
        """Generate a human-readable call summary for logging/dashboard."""
        parts = [f"Call {self.call_sid}"]
        if self.entities.caller_name:
            parts.append(f"Patient: {self.entities.caller_name}")
        if self.entities.intent:
            parts.append(f"Intent: {self.entities.intent}")
        if self.outcome:
            parts.append(f"Outcome: {self.outcome}")
        parts.append(f"Turns: {self.turn_count}")
        parts.append(f"State: {self.current_state}")
        return " | ".join(parts)

    def _maybe_start_summarization(self):
        """Kick off async summarization if turn count exceeds threshold."""
        if self.turn_count <= SUMMARIZE_THRESHOLD:
            return

        # Don't start if one is already running
        if self._summarize_task and not self._summarize_task.done():
            return

        # Get turns outside the sliding window
        max_messages = MAX_WINDOW_TURNS * 2
        if len(self.all_turns) <= max_messages:
            return

        old_turns = self.all_turns[:-max_messages]

        async def _do_summarize():
            self.summary = await summarize_turns(old_turns, self.summary)
            logger.info(f"Call {self.call_sid}: summarized {len(old_turns)} old messages")

        try:
            self._summarize_task = asyncio.create_task(_do_summarize())
        except RuntimeError:
            # No event loop — skip summarization
            pass


# =============================================================================
# Active Call Store (in-memory, will be Redis in Phase 2)
# =============================================================================

_active_calls: dict[str, CallState] = {}


def get_or_create_call_state(call_sid: str, clinic_id: int = None) -> CallState:
    """Get existing call state or create a new one."""
    if call_sid not in _active_calls:
        _active_calls[call_sid] = CallState(call_sid=call_sid, clinic_id=clinic_id)
        logger.info(f"Created call state for {call_sid}")
    return _active_calls[call_sid]


def get_call_state(call_sid: str) -> Optional[CallState]:
    """Get call state if it exists."""
    return _active_calls.get(call_sid)


def remove_call_state(call_sid: str) -> Optional[CallState]:
    """Remove and return the call state when call ends."""
    state = _active_calls.pop(call_sid, None)
    if state:
        logger.info(f"Removed call state: {state.get_call_summary()}")
    return state


def get_active_call_count() -> int:
    """Return number of active calls (for monitoring)."""
    return len(_active_calls)
