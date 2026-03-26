"""
CallGenius Voice Bot — Pipecat pipeline with Flows state machine.

Pipeline: Twilio Audio → Silero VAD → Deepgram STT → GPT-4o (via Flows) → Cartesia TTS → Twilio Audio

Key optimizations (from Vapi/Retell research):
- Streaming end-to-end: STT→LLM→TTS all stream, no stage waits for full completion
- Barge-in: VAD stops TTS within ~200ms when caller speaks
- Allow interruptions: enabled in PipelineParams
- Filler phrases: planned for Phase 2 (requires custom FrameProcessor)
"""

from fastapi import WebSocket

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat_flows import FlowManager

from server.config import get_settings
from server.bot.context import get_or_create_call_state, remove_call_state
from server.bot.flows import create_greeting_node
from server.utils.logger import logger

settings = get_settings()


async def run_bot(websocket: WebSocket, call_context: dict):
    """
    Initialize and run the Pipecat voice pipeline for an inbound call.

    Args:
        websocket: The FastAPI WebSocket connection from Twilio
        call_context: Dict with stream_sid, call_sid, caller_phone, called_number
    """
    call_sid = call_context.get("call_sid", "unknown")
    logger.info(f"Starting bot for call {call_sid}")

    # --- Create call state (4-layer context manager) ---
    call_state = get_or_create_call_state(call_sid)

    # --- Transport: Twilio WebSocket audio in/out ---
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=SileroVADAnalyzer.InputParams(
                    threshold=0.5,
                    min_volume=0.6,
                )
            ),
            vad_audio_passthrough=True,
            serializer=None,
            input_sample_rate=8000,
            output_sample_rate=8000,
        ),
    )

    # --- STT: Deepgram Nova-2 (streaming, ~150-300ms) ---
    stt = DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        params=DeepgramSTTService.InputParams(
            model="nova-2",
            language="en",
            encoding="mulaw",
            sample_rate=8000,
        ),
    )

    # --- LLM: GPT-4o (streaming, ~500ms TTFT) ---
    llm = OpenAILLMService(
        api_key=settings.openai_api_key,
        model="gpt-4o",
        params=OpenAILLMService.InputParams(
            temperature=0.4,
        ),
    )

    # --- TTS: Cartesia Sonic (streaming, ~40-90ms TTFB) ---
    tts = CartesiaTTSService(
        api_key=settings.cartesia_api_key,
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
        params=CartesiaTTSService.InputParams(
            encoding="pcm_mulaw",
            sample_rate=8000,
            language="en",
        ),
    )

    # --- LLM Context (managed by FlowManager) ---
    context = OpenAILLMContext()
    context_aggregator = llm.create_context_aggregator(context)

    # --- Pipeline ---
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # --- FlowManager: State machine orchestrator ---
    # FlowManager hooks into llm + context_aggregator to manage state transitions.
    # It does NOT appear in the pipeline array — it intercepts function calls.
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    # --- Event handlers ---
    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info(f"Call {call_sid}: client connected, starting greeting flow")
        initial_node = create_greeting_node(call_state)
        await flow_manager.initialize(initial_node)

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info(f"Call {call_sid}: client disconnected")
        # Log final call summary
        state = remove_call_state(call_sid)
        if state:
            logger.info(f"Call {call_sid} final: {state.get_call_summary()}")
        await task.cancel()

    # --- Run ---
    runner = PipelineRunner()
    await runner.run(task)
    logger.info(f"Bot finished for call {call_sid}")
