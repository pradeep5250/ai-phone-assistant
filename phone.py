"""
phone.py — Exotel Telephony Integration

Handles incoming phone calls via Exotel webhooks.
Uses ExoML (Exotel's XML dialect) for call flow control.

Call Flow:
    1. Caller dials +91 number
    2. Exotel sends webhook → /exotel/webhook
    3. Server returns ExoML to greet and record caller speech
    4. Exotel sends recording → /exotel/gather
    5. Server: download audio → STT → AI → TTS → respond via ExoML
    6. Loop until call ends
    7. Exotel sends status → /exotel/status
    8. Server saves call summary to MongoDB
"""

import os
import uuid
import httpx
from fastapi import APIRouter, Request, Form, Response
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from ai_assistant import AIAssistant
from voice import speech_to_text, text_to_speech_save, cleanup_audio_file
from database import (
    save_call, update_call, save_conversation,
    save_customer, save_appointment,
)

load_dotenv()

# ── Configuration ───────────────────────────────────────────────────

PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000")
HUMAN_AGENT_NUMBER = os.getenv("HUMAN_AGENT_NUMBER", "")
EXOTEL_SID = os.getenv("EXOTEL_SID", "")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY", "")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN", "")

# ── Active Calls ────────────────────────────────────────────────────

# In-memory store of active AI assistants (keyed by call SID)
# For a college project, this is fine. For production, use Redis.
active_calls: dict[str, AIAssistant] = {}

# ── FastAPI Router ──────────────────────────────────────────────────

router = APIRouter(prefix="/exotel", tags=["Exotel Webhooks"])


# ── Helper: Build ExoML Response ────────────────────────────────────

def exoml_response(xml_body: str) -> Response:
    """Wrap XML content in a proper ExoML response."""
    return Response(
        content=f'<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n{xml_body}\n</Response>',
        media_type="application/xml",
    )


# ── Webhook: Incoming Call ──────────────────────────────────────────

@router.post("/webhook")
async def handle_incoming_call(
    CallSid: str = Form(""),
    From: str = Form(""),
    To: str = Form(""),
    CurrentTime: str = Form(""),
):
    """
    Called by Exotel when a new call comes in.

    Creates an AI assistant for this call and returns ExoML to:
    1. Play a greeting
    2. Record the caller's speech
    """
    call_sid = CallSid or f"local_{uuid.uuid4().hex[:8]}"
    caller_number = From or "+91XXXXXXXXXX"

    print(f"\n[CALL] Incoming call: {caller_number} (SID: {call_sid})")

    # Create a new AI assistant for this call
    assistant = AIAssistant(call_id=call_sid)
    active_calls[call_sid] = assistant

    # Save initial call record
    save_call({
        "call_sid": call_sid,
        "caller_number": caller_number,
        "status": "in-progress",
        "duration": 0,
        "language": "en",
        "summary": "",
    })

    # Save/update customer
    save_customer({"phone_number": caller_number})

    # Generate greeting audio
    greeting = "Hi! I'm an AI assistant. How can I help you today?"
    greeting_audio_path = text_to_speech_save(greeting)
    greeting_filename = os.path.basename(greeting_audio_path)

    # Add greeting to conversation history
    assistant.messages.append({"role": "assistant", "content": greeting})

    # Return ExoML: play greeting, then record caller's response
    gather_url = f"{PUBLIC_URL}/exotel/gather?call_sid={call_sid}&caller={caller_number}"
    audio_url = f"{PUBLIC_URL}/audio/{greeting_filename}"

    return exoml_response(f"""
  <Play>{audio_url}</Play>
  <Record action="{gather_url}" maxLength="30" timeout="3" />
""")


# ── Webhook: Gather Caller Speech ───────────────────────────────────

@router.post("/gather")
async def handle_gather(
    request: Request,
    call_sid: str = "",
    caller: str = "",
    RecordingUrl: str = Form(""),
    CallSid: str = Form(""),
):
    """
    Called by Exotel after recording the caller's speech.

    Flow:
    1. Download the recorded audio from Exotel
    2. Convert speech to text (Whisper)
    3. Send text to AI assistant
    4. Convert AI response to speech (TTS)
    5. Return ExoML to play response and record again
    """
    # Get parameters from query string if not in form
    params = request.query_params
    call_sid = call_sid or params.get("call_sid", CallSid or "unknown")
    caller = caller or params.get("caller", "+91XXXXXXXXXX")

    print(f"\n[MIC] Processing speech for call: {call_sid}")

    # Get the AI assistant for this call
    assistant = active_calls.get(call_sid)
    if not assistant:
        print(f"[!] No assistant found for call {call_sid}, creating new one")
        assistant = AIAssistant(call_id=call_sid)
        active_calls[call_sid] = assistant

    # ── Step 1: Download recorded audio from Exotel ──
    caller_text = ""
    if RecordingUrl:
        try:
            async with httpx.AsyncClient() as client:
                audio_response = await client.get(
                    RecordingUrl,
                    auth=(EXOTEL_API_KEY, EXOTEL_API_TOKEN),
                    follow_redirects=True,
                    timeout=15.0,
                )
                audio_bytes = audio_response.content

            # ── Step 2: Speech to text ──
            caller_text = speech_to_text(audio_bytes)
        except Exception as e:
            print(f"[ERR] Error processing audio: {e}")
            caller_text = ""

    # Handle empty/no speech
    if not caller_text or caller_text.strip() == "":
        print("[...] No speech detected")
        # Ask the caller to speak again
        retry_audio = text_to_speech_save("I didn't catch that. Could you please say that again?")
        retry_filename = os.path.basename(retry_audio)
        gather_url = f"{PUBLIC_URL}/exotel/gather?call_sid={call_sid}&caller={caller}"
        audio_url = f"{PUBLIC_URL}/audio/{retry_filename}"
        return exoml_response(f"""
  <Play>{audio_url}</Play>
  <Record action="{gather_url}" maxLength="30" timeout="3" />
""")

    # ── Step 3: Get AI response ──
    print(f'[USER] Caller: "{caller_text}"')
    ai_response = assistant.chat(caller_text)
    print(f'[AI] AI: "{ai_response}"')

    # Detect intent
    intent = assistant.detect_intent(caller_text)

    # ── Handle human transfer ──
    if intent == "transfer":
        # Save conversation before transfer
        save_conversation({
            "call_sid": call_sid,
            "caller_number": caller,
            "messages": assistant.get_history(),
        })

        if HUMAN_AGENT_NUMBER:
            transfer_audio = text_to_speech_save(ai_response)
            transfer_filename = os.path.basename(transfer_audio)
            audio_url = f"{PUBLIC_URL}/audio/{transfer_filename}"
            return exoml_response(f"""
  <Play>{audio_url}</Play>
  <Dial>{HUMAN_AGENT_NUMBER}</Dial>
""")
        else:
            no_agent = text_to_speech_save(
                "Sorry, no human agent is available right now. Please try again later."
            )
            no_agent_filename = os.path.basename(no_agent)
            audio_url = f"{PUBLIC_URL}/audio/{no_agent_filename}"
            return exoml_response(f"""
  <Play>{audio_url}</Play>
  <Hangup />
""")

    # ── Step 4: Convert AI response to speech ──
    response_audio_path = text_to_speech_save(ai_response)
    response_filename = os.path.basename(response_audio_path)

    # Save conversation progress
    update_conversation_data(call_sid, caller, assistant)

    # ── Step 5: Play response and record next turn ──
    gather_url = f"{PUBLIC_URL}/exotel/gather?call_sid={call_sid}&caller={caller}"
    audio_url = f"{PUBLIC_URL}/audio/{response_filename}"

    return exoml_response(f"""
  <Play>{audio_url}</Play>
  <Record action="{gather_url}" maxLength="30" timeout="3" />
""")


# ── Webhook: Call Status ────────────────────────────────────────────

@router.post("/status")
async def handle_call_status(
    CallSid: str = Form(""),
    Status: str = Form(""),
    Duration: str = Form("0"),
    From: str = Form(""),
):
    """
    Called by Exotel when the call ends or status changes.
    Saves the final call summary and conversation to MongoDB.
    """
    call_sid = CallSid or "unknown"
    status = Status or "completed"
    duration = int(Duration) if Duration else 0
    caller_number = From or "+91XXXXXXXXXX"

    print(f"\n[STATUS] Call status: {call_sid} -> {status} ({duration}s)")

    # Get the assistant and generate summary
    assistant = active_calls.get(call_sid)
    summary = ""
    language = "en"

    if assistant:
        # Generate AI summary of the call
        try:
            summary = assistant.get_summary()
        except Exception as e:
            print(f"[!] Could not generate summary: {e}")
            summary = "Call completed."

        # Detect predominant language
        history = assistant.get_history()
        if history:
            user_messages = [m["message"] for m in history if m["role"] == "user"]
            if user_messages:
                language = assistant.detect_language(" ".join(user_messages))

        # Save final conversation
        save_conversation({
            "call_sid": call_sid,
            "caller_number": caller_number,
            "messages": assistant.get_history(),
        })

        # Clean up
        del active_calls[call_sid]

    # Update call record
    update_call(call_sid, {
        "status": status,
        "duration": duration,
        "language": language,
        "summary": summary,
    })

    print(f"[SUMMARY] Summary: {summary}")
    return {"status": "ok"}


# ── Serve Audio Files ───────────────────────────────────────────────

@router.get("/audio/{filename}")
async def serve_audio(filename: str):
    """
    Serve generated TTS audio files.
    Exotel's <Play> verb needs a public URL to the audio.
    """
    from voice import AUDIO_DIR
    file_path = AUDIO_DIR / filename
    if file_path.exists():
        return FileResponse(str(file_path), media_type="audio/mpeg")
    return Response(content="Audio not found", status_code=404)


# ── Helper Functions ────────────────────────────────────────────────

def update_conversation_data(call_sid: str, caller: str, assistant: AIAssistant):
    """Save conversation progress to MongoDB."""
    try:
        from database import update_conversation
        update_conversation(call_sid, assistant.get_history())
    except Exception as e:
        print(f"[!] Could not save conversation: {e}")


async def make_outbound_call(to_number: str):
    """
    Make an outbound call using Exotel API.
    (For future use — e.g., appointment reminders)
    """
    if not all([EXOTEL_SID, EXOTEL_API_KEY, EXOTEL_API_TOKEN]):
        print("[ERR] Exotel credentials not configured")
        return None

    subdomain = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")
    exotel_number = os.getenv("EXOTEL_PHONE_NUMBER", "")
    url = f"https://{subdomain}/v1/Accounts/{EXOTEL_SID}/Calls/connect.json"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            auth=(EXOTEL_API_KEY, EXOTEL_API_TOKEN),
            data={
                "From": exotel_number,
                "To": to_number,
                "CallerId": exotel_number,
                "Url": f"{PUBLIC_URL}/exotel/webhook",
                "StatusCallback": f"{PUBLIC_URL}/exotel/status",
            },
        )
        print(f"[OUT] Outbound call to {to_number}: {response.status_code}")
        return response.json()
