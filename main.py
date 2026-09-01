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

import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load  environment variables first
load_dotenv()

# Import project modules
from database import connect_db
from ai_assistant import AIAssistant
from phone import router as phone_router

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

# Include Exotel webhook routes
app.include_router(phone_router)

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

    return {
        "response": ai_response,
        "language": language,
        "intent": intent,
        "history": assistant.get_history(),
    }


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
        return FileResponse(str(file_path), media_type="audio/mpeg")
    return JSONResponse(status_code=404, content={"error": "Audio not found"})


# ── Run Server ─────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))

    print("\n" + "=" * 50)
    print("  AI Phone Call Assistant")
    print("=" * 50)
    print(f"  Server:  http://{host}:{port}")
    print(f"  Health:  http://{host}:{port}/health")
    print(f"  Test:    http://{host}:{port}/test/chat")
    print(f"  Docs:    http://{host}:{port}/docs")
    print("=" * 50 + "\n")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
    )
