import json
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from server.config import get_settings
from server.bot.bot import run_bot
from server.utils.logger import logger

settings = get_settings()

app = FastAPI(title="CallGenius AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "service": "CallGenius AI Voice Receptionist"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.api_route("/twilio/incoming", methods=["GET", "POST"])
async def twilio_incoming(request: Request):
    """Twilio webhook for incoming calls. Returns TwiML that connects to our WebSocket."""
    host = request.headers.get("host", f"localhost:{settings.port}")
    # Use wss:// in production, ws:// in dev
    protocol = "wss" if not settings.debug else "ws"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{protocol}://{host}/ws">
            <Parameter name="caller_phone" value="{{{{From}}}}"/>
            <Parameter name="called_number" value="{{{{To}}}}"/>
        </Stream>
    </Connect>
</Response>"""

    return HTMLResponse(content=twiml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    """WebSocket endpoint that Twilio streams audio to. Runs the Pipecat voice pipeline."""
    await websocket.accept()
    logger.info("Twilio WebSocket connected")

    start_data = None
    try:
        # Wait for the initial 'start' message from Twilio with call metadata
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            if data.get("event") == "start":
                start_data = data.get("start", {})
                logger.info(
                    f"Call started | stream_sid={start_data.get('streamSid')} "
                    f"call_sid={start_data.get('callSid')}"
                )
                break
            elif data.get("event") == "connected":
                logger.info("Twilio stream connected")
                continue

        if start_data:
            # Extract call metadata
            custom_params = start_data.get("customParameters", {})
            call_context = {
                "stream_sid": start_data.get("streamSid"),
                "call_sid": start_data.get("callSid"),
                "caller_phone": custom_params.get("caller_phone", "unknown"),
                "called_number": custom_params.get("called_number", "unknown"),
            }
            logger.info(f"Call context: {call_context}")

            # Run the Pipecat voice pipeline
            await run_bot(websocket, call_context)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("Twilio WebSocket disconnected")


if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
