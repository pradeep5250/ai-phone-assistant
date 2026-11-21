"""
main.py — AI Phone Call Assistant Entry Point

Starts the FastAPI server with:
- Exotel phone webhook routes
- A local test endpoint for text-based testing
- Health check endpoint
- Audio file serving

Run:
    python main.py                   
    or                                                                                                                                                                                                                                     
    uvicorn main:app --reload
"""

import sys
import os
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables first (.env.local wins)
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent / ".env.local", override=True)

# Import project modules
from database import connect_db
from ai_assistant import AIAssistant
from phone import router as phone_router
from realtime_server import router as realtime_router, STATIC_DIR
from fastapi.staticfiles import StaticFiles

# ── Lifespan ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    """Connect to MongoDB when the server starts."""
    try:
        connect_db()
    except Exception as e:
        print(f"[!] MongoDB not available: {e}")
        print("    The server will start, but database features won't work.")
        print("    Make sure MongoDB is running at the URI specified in .env")
    yield

# ── FastAPI App ─────────────────────────────────────────────────────

app = FastAPI(
    title="AI Phone Call Assistant",
    description="A simple AI phone assistant using OpenAI, Exotel, and MongoDB",
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/realtime":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Mount static files for Realtime Voice Studio UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Include Exotel phone webhook routes & Realtime UI routes
app.include_router(phone_router)
app.include_router(realtime_router)


# ── Root & Utility Routes ──────────────────────────────────────────

@app.get("/")
async def root():
    """Redirect root to the interactive Realtime Voice Studio UI."""
    return RedirectResponse(url="/realtime")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Handle browser favicon request cleanly without 404."""
    return Response(status_code=204)


@app.get("/agent", response_class=RedirectResponse)
@app.get("/elevenlabs", response_class=RedirectResponse)
@app.get("/elevenlabs/agent", response_class=RedirectResponse)
async def redirect_agent_page():
    """Redirect to Realtime Studio (ElevenLabs agent removed, voice enabled)."""
    return RedirectResponse(url="/realtime")


@app.get("/api/voice/config")
@app.get("/api/agent/config")
async def get_voice_config():
    """Return active Sarvam AI / voice & main assistant configuration from .env."""
    provider = os.getenv("TTS_PROVIDER", "sarvam")
    sarvam_speaker = os.getenv("SARVAM_TTS_SPEAKER", "kavitha")
    sarvam_model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
    sarvam_lang = os.getenv("SARVAM_TTS_LANGUAGE_CODE", "ta-IN")
    sarvam_pace = float(os.getenv("SARVAM_TTS_PACE", "1.25"))

    voice_id = sarvam_speaker if provider in ("sarvam", "sarvamai") else os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0")
    model_id = sarvam_model if provider in ("sarvam", "sarvamai") else os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")

    realtime_model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
    reasoning_model = os.getenv("OPENAI_REASONING_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.6-sol"))
    transcription_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", os.getenv("OPENAI_REALTIME_TRANSCRIPTION_MODEL", "gpt-live-transcribe"))

    return {
        "status": "ok",
        "provider": provider,
        "voice_id": voice_id,
        "model": model_id,
        "main_agent_voice_id": voice_id,
        "main_agent_model": model_id,
        "speaker": sarvam_speaker,
        "language_code": sarvam_lang,
        "pace": sarvam_pace,
        "realtime_model": realtime_model,
        "reasoning_model": reasoning_model,
        "transcription_model": transcription_model,
        "stt_model": transcription_model,
    }


# ── In-Memory Test Sessions ────────────────────────────────────────

# For local text-based testing (no phone needed)
test_sessions: dict[str, AIAssistant] = {}


# ── Health Check ───────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the server is running."""
    return {
        "status": "ok",
        "service": "AI Phone Call Assistant",
        "version": "1.0.0",
    }


# ── Test Endpoint (Text-Based Chat) ────────────────────────────────

@app.post("/test/chat")
async def test_chat(request: Request):
    """
    Test endpoint for text-based AI conversation.
    No phone or Exotel needed — just send JSON messages.

    Usage:
        curl -X POST http://localhost:8000/test/chat \\
          -H "Content-Type: application/json" \\
          -d '{"message": "Hello", "caller_number": "+919999999999"}'

    Send multiple messages with the same caller_number to maintain
    conversation history.
    """
    body = await request.json()
    message = body.get("message", "")
    caller_number = body.get("caller_number", "+919999999999")

    if not message:
        return JSONResponse(
            status_code=400,
            content={"error": "Message is required"},
        )

    # Get or create a test session for this caller
    if caller_number not in test_sessions:
        test_sessions[caller_number] = AIAssistant(call_id=f"test_{caller_number}")
        print(f"\n[+] New test session for {caller_number}")

    assistant = test_sessions[caller_number]

    # Get AI response
    ai_response = assistant.chat(message)

    # Detect language and intent
    language = assistant.detect_language(message)
    intent = assistant.detect_intent(message)

    # Save to database (if connected)
    try:
        from database import save_conversation, db
        if db is not None:
            save_conversation({
                "call_sid": f"test_{caller_number}",
                "caller_number": caller_number,
                "messages": assistant.get_history(),
            })
    except Exception as e:
        print(f"[!] Could not save to database: {e}")

    # Generate speech audio using main assistant's ElevenLabs voice ID
    audio_filename = None
    try:
        audio_path = assistant.save_speech(ai_response)
        audio_filename = os.path.basename(audio_path)
    except Exception as e:
        print(f"[!] Test chat voice synthesis note: {e}")

    result = {
        "response": ai_response,
        "language": language,
        "intent": intent,
        "history": assistant.get_history(),
        "voice_id": assistant.voice_id,
        "voice_model": assistant.voice_model,
        "tts_provider": assistant.tts_provider,
    }
    if audio_filename:
        result["audio_url"] = f"/audio/{audio_filename}"

    return result


@app.post("/test/reset")
async def test_reset(request: Request):
    """Reset a test session to start a new conversation."""
    body = await request.json()
    caller_number = body.get("caller_number", "+919999999999")

    if caller_number in test_sessions:
        del test_sessions[caller_number]
        return {"status": "reset", "caller_number": caller_number}

    return {"status": "no session found", "caller_number": caller_number}


# ── Audio File Serving ─────────────────────────────────────────────
# Note: Audio files are also served via phone.py router at /exotel/audio/
# This is a secondary route for direct access.

@app.get("/audio/{filename}")
async def serve_audio_main(filename: str):
    """Serve TTS audio files (used by Exotel's <Play> verb)."""
    from voice import AUDIO_DIR
    from fastapi.responses import FileResponse

    file_path = AUDIO_DIR / filename
    if file_path.exists():
        media_type = "audio/wav" if filename.lower().endswith(".wav") else "audio/mpeg"
        return FileResponse(str(file_path), media_type=media_type)
    return JSONResponse(status_code=404, content={"error": "Audio not found"})


# ── ElevenLabs Speech Engine Upstream WebSocket ────────────────────

speech_engine_sessions: dict[str, AIAssistant] = {}

@app.websocket("/ws")
async def speech_engine_ws_route(websocket: WebSocket):
    """
    Upstream WebSocket endpoint for ElevenLabs Speech Engine (seng_...).
    ElevenLabs connects here to bridge ASR (Scribe) + MintX LLM + TTS (Yazhini).
    """
    from elevenlabs.speech_engine.session import SpeechEngineSession
    from elevenlabs.speech_engine.types import ConversationMessage

    await websocket.accept()
    print("\n[+] ElevenLabs Speech Engine connected at /ws!")

    session = SpeechEngineSession(websocket, debug=True)
    conv_id = "default"

    async def on_init(conversation_id: str):
        nonlocal conv_id
        conv_id = conversation_id
        print(f"[ElevenLabs Init] Conversation ID: {conversation_id}")
        if conv_id not in speech_engine_sessions:
            speech_engine_sessions[conv_id] = AIAssistant(call_id=f"el_{conv_id}")

    async def on_transcript(transcript: list[ConversationMessage]):
        if not transcript:
            return
        user_text = transcript[-1].content.strip()
        print(f"\n🗣️ [ElevenLabs Speech Engine User]: {user_text}")
        if not user_text:
            return
        if conv_id not in speech_engine_sessions:
            speech_engine_sessions[conv_id] = AIAssistant(call_id=f"el_{conv_id}")
        bot = speech_engine_sessions[conv_id]
        reply = bot.chat(user_text)
        print(f"🤖 [MintX Agent Response]: {reply}")
        await session.send_response(reply)

    async def on_close():
        print(f"[-] ElevenLabs Speech Engine session closed: {conv_id}")
        if conv_id in speech_engine_sessions:
            del speech_engine_sessions[conv_id]

    session.on("init", on_init)
    session.on("user_transcript", on_transcript)
    session.on("close", on_close)

    try:
        await session.run()
    except WebSocketDisconnect:
        print(f"[-] ElevenLabs WebSocket disconnected: {conv_id}")
    except Exception as e:
        print(f"[!] ElevenLabs Speech Engine error: {e}")


def main():
    """Main application entry point for Mintx Solutions AI Phone Assistant."""
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    display_host = "localhost" if host == "0.0.0.0" else host

    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
    voice = os.getenv("OPENAI_VOICE", "sarvam")
    stt_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", os.getenv("OPENAI_REALTIME_TRANSCRIPTION_MODEL", "gpt-transcribe"))
    reasoning_model = os.getenv("OPENAI_REASONING_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.6-sol"))

    # Check active instruction persona
    active_inst_name = "Default (Mintx Solutions)"
    try:
        from instructions import get_active_instruction
        active_inst = get_active_instruction()
        if active_inst and active_inst.get("name"):
            active_inst_name = f"{active_inst.get('name')} [{active_inst.get('category', 'Custom')}]"
    except Exception:
        pass

    sarvam_speaker = os.getenv("SARVAM_TTS_SPEAKER", "kavitha")
    sarvam_model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
    tts_provider = os.getenv("TTS_PROVIDER", "sarvam")

    print("\n" + "=" * 65)
    print("  🎙️  MINTX SOLUTIONS AI — MULTI-MODEL PHONE ASSISTANT  🎙️")
    print("=" * 65)
    print(f"  🧠 Reasoning / Requirements: {reasoning_model}")
    print(f"  🎙️ Realtime Conversation:   {model}")
    print(f"  🗣️ Speech Transcription:     {stt_model}")
    print(f"  🔊 Voice Output Engine:      Sarvam AI ({sarvam_model} • {sarvam_speaker})")
    print(f"  🎭 Main Persona Voice:       Yazhini ({sarvam_speaker})")
    print(f"  📋 Active System Persona:    {active_inst_name}")

    # Verify OpenAI API Key and Credit Quota
    openai_key = os.getenv("OPENAI_API_KEY", "")
    key_status_str = "Not Configured"
    is_quota_exhausted = False
    if openai_key:
        try:
            from openai import OpenAI
            _client = OpenAI(api_key=openai_key)
            _client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
            key_status_str = "Active (Credits Available ✅)"
        except Exception as e:
            err_text = str(e)
            if "insufficient_quota" in err_text or "credit_balance_exhausted" in err_text or "429" in err_text:
                key_status_str = "EXHAUSTED (0 Credits Remaining ❌)"
                is_quota_exhausted = True
            else:
                key_status_str = f"Error ({err_text[:35]})"
    print(f"  🔑 OpenAI Key Status:    {key_status_str}")

    if is_quota_exhausted:
        print("\n  ⚠️  ATTENTION: OPENAI ACCOUNT HAS NO REMAINING CREDITS!")
        print("  👉 The Realtime Model will not be able to respond until credits are added.")
        print("  👉 Add credits at: https://platform.openai.com/settings/organization/billing/")
        print("  👉 Or update OPENAI_API_KEY in .env or via Web UI Settings.")

    print(f"  🌐 Studio UI:        http://{display_host}:{port}/realtime")
    print(f"  👉 Direct Link:      http://localhost:{port}")
    print("=" * 60 + "\n")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
