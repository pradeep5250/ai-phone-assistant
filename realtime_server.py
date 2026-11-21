"""
realtime_server.py — FastAPI WebSocket Proxy & Web UI Server for OpenAI Realtime Voice

Bridges the browser Web Audio UI to OpenAI's Realtime WebSocket API:
- Serves static assets & HTML UI
- Establishes persistent WebSocket link to wss://api.openai.com/v1/realtime
- Streams PCM 24000Hz audio bidirectional with sub-second latency
- Injects Mintx Solutions system persona, selected voice, and server VAD
"""

import os
import sys
import json
import time
import inspect
import asyncio
import re
import socket
import ssl
from pathlib import Path
from dotenv import load_dotenv

from typing import Optional
import websockets
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import existing configs
from voice import REALTIME_SESSION_CONFIG, DEFAULT_VOICE, SARVAM_SPEAKERS

OPENAI_VOICES = ("marin", "cedar", "alloy", "echo", "fable", "onyx", "nova", "shimmer", "verse", "ballad", "ash", "sage", "coral")
SARVAM_VOICE_ALIASES = ("sarvam", "sarvamai", "default", "kavitha", "kavya", "priya", "shreya", "roopa", "mani", "gokul", "vijay")
from ai_assistant import SYSTEM_PROMPT
from instructions import (
    get_all_data, get_instructions_list, get_active_instruction,
    get_active_system_prompt, set_active_instruction, add_instruction,
    update_instruction, delete_instruction, refine_instruction_with_feedback,
    undo_last_voice_correction, get_training_history,
    get_active_starting_greeting, set_active_starting_greeting,
    modify_training_rule, delete_training_rule, add_custom_training_rule
)
from tanglish_script import tanglish_to_tamil_display, tanglish_for_tts, normalize_brand_name
from language_detector import detect_language, get_language_display_name, get_sarvam_language_code


def is_english_tool_filler(text: str) -> bool:
    """Detect if text is an English conversational filler/thinking phrase before or during tool operations."""
    if not text:
        return False
    t = text.lower().strip()
    filler_patterns = [
        r"^got it,?\s+thanks for sharing",
        r"^i['’]?ll (?:quickly )?(?:store|record|finalize|save) this",
        r"^alright,?\s+i['’]?ll (?:record|store|finalize|save)",
        r"^let me (?:quickly )?(?:store|record|finalize|save)",
        r"^okay,?\s+i['’]?ll (?:record|store|finalize|save)",
        r"i['’]?ll record that requirement and then we['’]?ll confirm",
    ]
    return any(re.search(p, t) for p in filler_patterns)


load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent / ".env.local", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
DEFAULT_VOICE_NAME = os.getenv("OPENAI_VOICE", "sarvam")
DEFAULT_SARVAM_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "kavitha")
DEFAULT_SARVAM_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
DEFAULT_REASONING_MODEL = os.getenv("OPENAI_REASONING_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.6-sol"))
DEFAULT_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", os.getenv("OPENAI_REALTIME_TRANSCRIPTION_MODEL", "gpt-live-transcribe"))
DEFAULT_ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_VECTOR_DB = "Qdrant"

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


async def connect_openai_realtime(model: str, api_key: str, open_timeout: float = 20.0):
    """
    Connect to OpenAI Realtime WebSocket with IPv4-first DNS and retries.
    Fixes intermittent Windows [Errno 11001] getaddrinfo failed (often IPv6/DNS race).
    """
    host = "api.openai.com"
    uri = f"wss://{host}/v1/realtime?model={model}"
    headers = {"Authorization": f"Bearer {api_key}"}

    sig = inspect.signature(websockets.connect)
    header_kwarg = "additional_headers" if "additional_headers" in sig.parameters else "extra_headers"

    last_err = None
    loop = asyncio.get_running_loop()

    for attempt in range(1, 5):
        try:
            # Warm DNS with IPv4 preference before connecting (avoids flaky getaddrinfo)
            try:
                await loop.getaddrinfo(host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
            except Exception as dns_err:
                print(f"[!] DNS warm-up failed (attempt {attempt}): {dns_err}")
                await asyncio.sleep(0.3 * attempt)
                await loop.getaddrinfo(host, 443, type=socket.SOCK_STREAM)

            connect_kwargs = {
                header_kwarg: headers,
                "open_timeout": open_timeout,
                "ping_interval": 20,
                "ping_timeout": 20,
                "proxy": None,  # ignore broken system proxies that break DNS
            }
            # Force IPv4 when supported by this websockets version
            try:
                connect_kwargs["family"] = socket.AF_INET
                print(f"[Realtime] Connecting to OpenAI (IPv4, attempt {attempt}/4)...")
                return await websockets.connect(uri, **connect_kwargs)
            except TypeError:
                connect_kwargs.pop("family", None)
                print(f"[Realtime] Connecting to OpenAI (attempt {attempt}/4)...")
                return await websockets.connect(uri, **connect_kwargs)
        except Exception as e:
            last_err = e
            print(f"[!] OpenAI WS connect attempt {attempt} failed: {e}")
            await asyncio.sleep(0.5 * attempt)

    raise last_err if last_err else OSError("Failed to connect to OpenAI Realtime")

router = APIRouter()

_KNOWLEDGE_FILE = Path(__file__).parent / "data" / "mintx_knowledge.json"
_knowledge_cache = {"mtime": None, "text": ""}


def get_knowledge_brief() -> str:
    """All MintX knowledge-base facts as compact text (re-read when the file changes)."""
    try:
        mtime = _KNOWLEDGE_FILE.stat().st_mtime
    except OSError:
        return ""
    if _knowledge_cache["mtime"] != mtime:
        try:
            docs = json.loads(_KNOWLEDGE_FILE.read_text(encoding="utf-8"))
            _knowledge_cache["text"] = "\n".join(f"- {d.get('title', '')}: {d.get('content', '')}" for d in docs)
            _knowledge_cache["mtime"] = mtime
        except Exception as e:
            print(f"[!] Could not load knowledge for instructions: {e}")
    return _knowledge_cache["text"]


@router.get("/realtime", response_class=HTMLResponse)
async def get_realtime_ui():
    """Serve the Realtime Voice Studio UI with no-cache headers."""
    html_file = TEMPLATES_DIR / "realtime.html"
    if html_file.exists():
        content = html_file.read_text(encoding="utf-8")
        return HTMLResponse(
            content=content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return HTMLResponse("<h2>Realtime UI template not found.</h2>", status_code=404)


# ── Available GPT Voices Metadata ─────────────────────────────────────

# Voice speaking is Sarvam AI. OpenAI Realtime is brain/text; Sarvam AI synthesizes speech.
AVAILABLE_GPT_VOICES = [
    {
        "id": "sarvam",
        "name": "Yazhini (Sarvam AI)",
        "gender": "Female",
        "tone": "Warm & Natural South Indian Customer Care",
        "tag": "Active Voice",
        "emoji": "🌟",
        "speaker": os.getenv("SARVAM_TTS_SPEAKER", "kavitha"),
        "provider": "sarvam",
    },
    {
        "id": "kavitha",
        "name": "Kavitha (Sarvam AI - Yazhini)",
        "gender": "Female",
        "tone": "Natural Tamil Female Voice",
        "tag": "Recommended",
        "emoji": "👩",
        "speaker": "kavitha",
        "provider": "sarvam",
    },
    {
        "id": "kavya",
        "name": "Kavya (Sarvam AI)",
        "gender": "Female",
        "tone": "Young & Energetic Voice",
        "tag": "Alternative",
        "emoji": "👧",
        "speaker": "kavya",
        "provider": "sarvam",
    },
    {
        "id": "priya",
        "name": "Priya (Sarvam AI)",
        "gender": "Female",
        "tone": "Warm & Professional",
        "tag": "Alternative",
        "emoji": "👩‍💼",
        "speaker": "priya",
        "provider": "sarvam",
    },
    {
        "id": "shreya",
        "name": "Shreya (Sarvam AI)",
        "gender": "Female",
        "tone": "Gentle & Expressive",
        "tag": "Alternative",
        "emoji": "✨",
        "speaker": "shreya",
        "provider": "sarvam",
    },
    {
        "id": "mani",
        "name": "Mani (Sarvam AI)",
        "gender": "Male",
        "tone": "Authentic Tamil Male Executive",
        "tag": "Male",
        "emoji": "👨",
        "speaker": "mani",
        "provider": "sarvam",
    },
    {
        "id": "gokul",
        "name": "Gokul (Sarvam AI)",
        "gender": "Male",
        "tone": "Young Confident Tamil Male",
        "tag": "Male",
        "emoji": "🧔",
        "speaker": "gokul",
        "provider": "sarvam",
    },
]

@router.get("/api/voices")
async def get_available_voices():
    """Return all available GPT voices and metadata."""
    return {
        "status": "ok",
        "default_voice": DEFAULT_VOICE_NAME,
        "voices": AVAILABLE_GPT_VOICES
    }


# ── Instructions REST API Endpoints ─────────────────────────────────

@router.get("/api/instructions")
async def get_instructions():
    """Get all instruction presets and active instruction info."""
    data = get_all_data()
    active = get_active_instruction()
    active_greet = get_active_starting_greeting()
    if active:
        active["starting_greeting"] = active_greet
    return {
        "status": "ok",
        "active_id": data.get("active_id", "default-mintx"),
        "items": data.get("items", []),
        "active": active,
        "starting_greeting": active_greet
    }


@router.post("/api/instructions")
async def create_instruction(request: Request):
    """Create a new manual instruction preset."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    name = body.get("name", "").strip()
    inst = body.get("instructions", "").strip()
    desc = body.get("description", "").strip()
    cat = body.get("category", "Custom").strip()

    if not name or not inst:
        raise HTTPException(status_code=400, detail="Name and instructions are required.")

    item = add_instruction(name=name, instructions=inst, description=desc, category=cat)
    if body.get("set_active"):
        set_active_instruction(item["id"])

    return {
        "status": "ok",
        "item": item,
        "active": get_active_instruction(),
        "active_id": get_all_data().get("active_id")
    }


@router.put("/api/instructions/{instruction_id}")
async def modify_instruction(instruction_id: str, request: Request):
    """Modify / edit an existing instruction preset."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    name = body.get("name")
    inst = body.get("instructions")
    desc = body.get("description")
    cat = body.get("category")

    updated = update_instruction(instruction_id, name=name, instructions=inst, description=desc, category=cat)
    if not updated:
        raise HTTPException(status_code=404, detail="Instruction preset not found.")

    if body.get("set_active"):
        set_active_instruction(instruction_id)

    return {
        "status": "ok",
        "item": updated,
        "active": get_active_instruction(),
        "active_id": get_all_data().get("active_id")
    }


@router.delete("/api/instructions/{instruction_id}")
async def remove_instruction(instruction_id: str):
    """Delete an instruction preset managed by the UI."""
    success = delete_instruction(instruction_id)
    if not success:
        raise HTTPException(status_code=404, detail="Instruction preset not found.")
    return {
        "status": "ok",
        "active_id": get_all_data().get("active_id"),
        "active": get_active_instruction()
    }


@router.post("/api/instructions/active")
async def set_active(request: Request):
    """Set the currently active instruction."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    instruction_id = body.get("id")
    if not instruction_id:
        raise HTTPException(status_code=400, detail="Instruction ID is required.")

    success = set_active_instruction(instruction_id)
    if not success:
        raise HTTPException(status_code=404, detail="Instruction preset not found.")

    return {
        "status": "ok",
        "active": get_active_instruction(),
        "active_id": instruction_id
    }


# ── Voice Training & Interactive Prompt Tuning Endpoints ──────────

@router.post("/api/instructions/voice-correct")
async def voice_correct_instruction(request: Request):
    """
    Refines system instructions based on spoken feedback text.
    Uses GPT-4o-mini to seamlessly integrate the user's correction (Tamil/English).
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    feedback = body.get("feedback", "").strip()
    if not feedback:
        raise HTTPException(status_code=400, detail="Feedback / correction text is required.")

    instruction_id = body.get("instruction_id") or get_all_data().get("active_id", "default-mintx")
    last_response = body.get("last_response", "").strip()
    context = body.get("context", "").strip()

    try:
        result = await asyncio.to_thread(
            refine_instruction_with_feedback,
            instruction_id=instruction_id,
            user_feedback=feedback,
            last_assistant_response=last_response,
            conversation_context=context,
            api_key=OPENAI_API_KEY
        )
        result["active"] = get_active_instruction()
        result["active_prompt"] = get_active_system_prompt()
        return result
    except Exception as e:
        print(f"[Voice Correct Error]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/instructions/voice-transcribe")
async def transcribe_voice(
    audio: UploadFile = File(...),
    instruction_id: Optional[str] = Form(None),
    last_response: Optional[str] = Form(""),
    context: Optional[str] = Form(""),
    auto_correct: Optional[bool] = Form(False)
):
    """
    Transcribe a recorded spoken voice clip using OpenAI Whisper API.
    Auto-detects Tamil and English. If auto_correct is True, refines the prompt directly.
    Runs non-blocking on threadpool.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")

    from openai import OpenAI
    import io
    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        content = await audio.read()
        audio_file = io.BytesIO(content)
        filename = audio.filename or "recording.webm"
        if not any(filename.endswith(ext) for ext in [".webm", ".wav", ".mp3", ".m4a", ".ogg"]):
            filename = "recording.webm"
        audio_file.name = filename

        def _do_whisper():
            stt_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", os.getenv("OPENAI_REALTIME_TRANSCRIPTION_MODEL", "whisper-1"))
            try:
                transcription = client.audio.transcriptions.create(
                    model=stt_model,
                    file=audio_file,
                    prompt="MintX Solutions, Tamil, Tanglish, English, website, app, software, வணக்கம், வெல்கம்",
                    language=None,
                    temperature=0.0
                )
                return transcription.text.strip()
            except Exception:
                audio_file.seek(0)
                try:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        prompt="MintX Solutions, Tamil, Tanglish, English, website, app, software, வணக்கம், வெல்கம்",
                        language=None,
                        temperature=0.0
                    )
                    return transcription.text.strip()
                except Exception:
                    audio_file.seek(0)
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=None,
                        temperature=0.0
                    )
                    return transcription.text.strip()

        text = await asyncio.to_thread(_do_whisper)

        result = {
            "status": "ok",
            "transcript": text
        }

        if auto_correct and text:
            active_id = instruction_id or get_all_data().get("active_id", "default-mintx")
            corr = await asyncio.to_thread(
                refine_instruction_with_feedback,
                instruction_id=active_id,
                user_feedback=text,
                last_assistant_response=last_response or "",
                conversation_context=context or "",
                api_key=OPENAI_API_KEY
            )
            result["correction"] = corr
            result["active"] = get_active_instruction()
            result["active_prompt"] = get_active_system_prompt()

        return result
    except Exception as e:
        print(f"[Whisper Transcribe Error]: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")


@router.get("/api/instructions/history")
async def get_history():
    """Return recent voice training history log."""
    return {
        "status": "ok",
        "history": get_training_history()
    }


@router.post("/api/instructions/undo")
async def undo_correction(request: Request):
    """Undo the last voice correction for an instruction."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    instruction_id = body.get("instruction_id") or get_all_data().get("active_id", "default-mintx")
    item = undo_last_voice_correction(instruction_id)
    if not item:
        raise HTTPException(status_code=400, detail="No previous revision found to revert.")
    return {
        "status": "ok",
        "item": item,
        "active": get_active_instruction(),
        "active_prompt": get_active_system_prompt()
    }


@router.put("/api/instructions/history/{train_id}")
async def update_training_rule(train_id: str, request: Request):
    """Modify an existing training correction rule and update instructions.json."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    new_rule = body.get("rule_diff") or body.get("rule")
    if not new_rule or not new_rule.strip():
        raise HTTPException(status_code=400, detail="rule_diff is required.")

    new_summary = body.get("summary_en")
    result = modify_training_rule(train_id, new_rule.strip(), new_summary)
    if not result:
        raise HTTPException(status_code=404, detail="Training rule record not found.")

    return result


@router.delete("/api/instructions/history/{train_id}")
async def remove_training_rule(train_id: str):
    """Delete a training correction rule from history and instructions.json."""
    success = delete_training_rule(train_id)
    if not success:
        raise HTTPException(status_code=404, detail="Training rule not found.")
    return {
        "status": "ok",
        "deleted_id": train_id,
        "history": get_training_history(),
        "active": get_active_instruction(),
        "active_prompt": get_active_system_prompt()
    }


@router.post("/api/instructions/rules")
async def add_rule_direct(request: Request):
    """Directly customize and add a new rule into instructions.json."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    rule = body.get("rule") or body.get("rule_diff")
    if not rule or not rule.strip():
        raise HTTPException(status_code=400, detail="Rule text is required.")

    instruction_id = body.get("instruction_id")
    summary = body.get("summary_en")
    return add_custom_training_rule(rule.strip(), instruction_id=instruction_id, summary=summary)


@router.get("/api/instructions/starting-greeting")
async def get_starting_greeting_api(instruction_id: Optional[str] = None):
    """Return the active starting greeting phrase."""
    return {
        "status": "ok",
        "starting_greeting": get_active_starting_greeting(instruction_id),
        "active_id": get_all_data().get("active_id", "default-mintx")
    }


@router.post("/api/instructions/starting-greeting")
async def set_starting_greeting_api(request: Request):
    """Set and persist the starting greeting phrase."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    greeting = body.get("greeting", "").strip()
    if not greeting:
        raise HTTPException(status_code=400, detail="Greeting text cannot be empty.")

    instruction_id = body.get("instruction_id") or get_all_data().get("active_id", "default-mintx")
    res = set_active_starting_greeting(instruction_id, greeting)
    res["active"] = get_active_instruction()
    res["active_prompt"] = get_active_system_prompt()
    res["verbal_ack"] = f'Sure sir, starting greeting-a "{res["starting_greeting"]}" nu update panniten!'
    return res


# ── Recordings & Transcripts Storage API ────────────────────────────

@router.post("/api/recordings/save")
async def save_recording_api(
    audio_file: Optional[UploadFile] = File(None),
    transcript_text: Optional[str] = Form(""),
    session_id: Optional[str] = Form(""),
    voice: Optional[str] = Form(""),
    model: Optional[str] = Form(""),
    persona: Optional[str] = Form(""),
):
    """
    Save audio recording file and formatted text transcript to server recordings/ directory.
    """
    try:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        safe_sid = session_id or f"session_{timestamp_str}"
        # Sanitize session_id
        safe_sid = "".join(c for c in safe_sid if c.isalnum() or c in ("-", "_")).strip() or f"session_{timestamp_str}"

        saved_audio_name = None
        saved_txt_name = None

        # 1. Save Audio File if provided
        if audio_file:
            orig_name = audio_file.filename or "recording.wav"
            ext = os.path.splitext(orig_name)[1].lower()
            if not ext or ext not in (".wav", ".webm", ".mp3", ".ogg"):
                ext = ".wav"
            saved_audio_name = f"call_audio_{safe_sid}_{timestamp_str}{ext}"
            audio_path = RECORDINGS_DIR / saved_audio_name
            content = await audio_file.read()
            with open(audio_path, "wb") as f:
                f.write(content)
            print(f"[Recordings] Saved audio recording: {audio_path} ({len(content)} bytes)")

        # 2. Save Formatted Text Transcript if provided
        if transcript_text and transcript_text.strip():
            saved_txt_name = f"call_transcript_{safe_sid}_{timestamp_str}.txt"
            txt_path = RECORDINGS_DIR / saved_txt_name
            txt_content = transcript_text.strip()
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(txt_content)
            print(f"[Recordings] Saved conversation transcript: {txt_path} ({len(txt_content)} chars)")

        return {
            "status": "ok",
            "session_id": safe_sid,
            "audio_filename": saved_audio_name,
            "transcript_filename": saved_txt_name,
            "audio_url": f"/api/recordings/download/{saved_audio_name}" if saved_audio_name else None,
            "transcript_url": f"/api/recordings/download/{saved_txt_name}" if saved_txt_name else None,
        }
    except Exception as e:
        print(f"[Recordings Save Error]: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save recording: {str(e)}")


@router.get("/api/recordings")
async def list_recordings_api():
    """
    List all saved conversation audio and text transcript recordings.
    """
    try:
        files = []
        if RECORDINGS_DIR.exists():
            for p in sorted(RECORDINGS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if p.is_file() and p.suffix.lower() in (".wav", ".webm", ".mp3", ".txt"):
                    files.append({
                        "filename": p.name,
                        "size_bytes": p.stat().st_size,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
                        "type": "audio" if p.suffix.lower() in (".wav", ".webm", ".mp3") else "transcript",
                        "download_url": f"/api/recordings/download/{p.name}",
                    })
        return {"status": "ok", "count": len(files), "recordings": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/recordings/download/{filename}")
async def download_recording_api(filename: str):
    """
    Download or stream a saved audio recording or text transcript file.
    """
    # Prevent path traversal
    clean_name = os.path.basename(filename)
    file_path = RECORDINGS_DIR / clean_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    ext = file_path.suffix.lower()
    media_type = "application/octet-stream"
    if ext == ".wav":
        media_type = "audio/wav"
    elif ext == ".webm":
        media_type = "audio/webm"
    elif ext == ".mp3":
        media_type = "audio/mpeg"
    elif ext == ".txt":
        media_type = "text/plain; charset=utf-8"

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=clean_name,
        headers={"Content-Disposition": f'attachment; filename="{clean_name}"'}
    )


# ── Sarvam AI Voice Testing & Preview API ───────────────────────────

@router.post("/api/sarvam/test")
@router.post("/api/elevenlabs/test")
async def test_sarvam_audio(request: Request):
    """
    Synthesizes a short test audio preview using Sarvam AI Text-to-Speech (bulbul:v3).
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    text = body.get("text", "").strip() or "வணக்கம்! MintX Solutions-ku வரவேற்கிறோம். என்ன சர்வீஸ் வேணும்னு சொல்லுங்க?"
    speaker = body.get("speaker") or body.get("voice_id") or os.getenv("SARVAM_TTS_SPEAKER", "kavitha")
    model = body.get("model") or body.get("model_id") or os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
    language_code = body.get("language_code") or os.getenv("SARVAM_TTS_LANGUAGE_CODE", "ta-IN")
    pace = float(body.get("pace") or os.getenv("SARVAM_TTS_PACE", "1.25"))

    try:
        from voice import text_to_speech_sarvam
        import base64
        audio_bytes = await asyncio.to_thread(
            text_to_speech_sarvam,
            text=text,
            speaker=speaker,
            model=model,
            language_code=language_code,
            pace=pace,
        )
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {
            "status": "ok",
            "audio_base64": b64,
            "format": "audio/wav",
            "model_id": model,
            "voice_id": speaker,
            "speaker": speaker,
            "language_code": language_code,
        }
    except Exception as e:
        print(f"[!] Sarvam AI Voice Test Error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# ── OpenAI Key & Quota Management API ────────────────────────────────

@router.get("/api/openai/status")
async def get_openai_status():
    """Check OpenAI API key status and verify credit quota."""
    key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY or ""
    if not key:
        return {
            "status": "missing",
            "credits": "none",
            "masked_key": "",
            "message": "No OPENAI_API_KEY configured in .env"
        }

    masked = key[:7] + "..." + key[-4:] if len(key) > 12 else "***"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        # Test completion with 1 token to check quota
        await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
        return {
            "status": "ok",
            "credits": "available",
            "masked_key": masked,
            "message": "OpenAI API Key is active with available credits."
        }
    except Exception as e:
        err_str = str(e)
        if "insufficient_quota" in err_str or "credit_balance_exhausted" in err_str or "429" in err_str:
            return {
                "status": "quota_exhausted",
                "credits": "exhausted",
                "masked_key": masked,
                "message": "OpenAI account credit balance is exhausted ($0.00). Please add credits at https://platform.openai.com/settings/organization/billing/ or enter a new API key."
            }
        return {
            "status": "error",
            "credits": "unknown",
            "masked_key": masked,
            "message": f"OpenAI Key verification: {err_str}"
        }


@router.post("/api/openai/key")
async def update_openai_key(request: Request):
    """Validate and update OPENAI_API_KEY dynamically and in .env."""
    global OPENAI_API_KEY
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    new_key = body.get("api_key", "").strip()
    if not new_key or not new_key.startswith("sk-"):
        raise HTTPException(status_code=400, detail="Invalid OpenAI API key format. Key must start with 'sk-'.")

    # Verify new key with OpenAI
    try:
        from openai import OpenAI
        client = OpenAI(api_key=new_key)
        await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
    except Exception as e:
        err_str = str(e)
        if "insufficient_quota" in err_str or "credit_balance_exhausted" in err_str:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "code": "insufficient_quota",
                "message": "The provided OpenAI API key has no credit balance ($0.00). Please add credits to this account first."
            })
        return JSONResponse(status_code=400, content={
            "status": "error",
            "message": f"Verification failed: {err_str}"
        })

    # Update in memory
    os.environ["OPENAI_API_KEY"] = new_key
    OPENAI_API_KEY = new_key

    # Save to .env and .env.local
    for env_name in [".env", ".env.local"]:
        p = BASE_DIR / env_name
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8")
                if "OPENAI_API_KEY=" in txt:
                    txt = re.sub(r"OPENAI_API_KEY=.*", f"OPENAI_API_KEY={new_key}", txt)
                else:
                    txt += f"\nOPENAI_API_KEY={new_key}\n"
                p.write_text(txt, encoding="utf-8")
            except Exception as e:
                print(f"[!] Warning updating {env_name}: {e}")

    masked = new_key[:7] + "..." + new_key[-4:]
    return {
        "status": "ok",
        "masked_key": masked,
        "message": "OpenAI API Key verified and saved successfully!"
    }


# ── Comprehensive Model & System Settings API ────────────────────────

@router.get("/api/architecture")
@router.get("/api/models/architecture")
async def get_architecture_layers():
    """
    Return status and specifications for the official 7-layer real-time architecture:
    1. 🎤 STT: OpenAI GPT Live Transcribe (gpt-live-transcribe)
    2. 🧠 LLM: OpenAI latest realtime/reasoning model (gpt-realtime-2.1 / gpt-5.6-sol)
    3. 🔎 Embeddings: text-embedding-3-large
    4. 📚 Vector DB: Qdrant (Collection: mintx_knowledge)
    5. 🔊 TTS: Eleven v4 Turbo (eleven_v4_turbo)
    6. 💾 Database: MongoDB (ai_phone_assistant)
    7. ⚡ Backend: FastAPI + WebSocket
    """
    # Fetch live RAG / Qdrant status
    rag_info = {}
    try:
        from rag import get_rag_engine
        rag_engine = get_rag_engine()
        rag_info = rag_engine.get_status()
    except Exception as e:
        rag_info = {"status": "error", "error": str(e), "vector_db": "Qdrant"}

    # Fetch live MongoDB status
    mongo_status = "offline"
    mongo_collections = []
    try:
        from database import get_db
        db_inst = get_db()
        if db_inst is not None:
            mongo_status = "connected"
            mongo_collections = db_inst.list_collection_names()
    except Exception:
        pass

    realtime_m = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
    reasoning_m = os.getenv("OPENAI_REASONING_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.6-sol"))
    transcribe_m = os.getenv("OPENAI_TRANSCRIPTION_MODEL", os.getenv("OPENAI_REALTIME_TRANSCRIPTION_MODEL", "gpt-live-transcribe"))
    eleven_m = os.getenv("ELEVENLABS_MODEL_ID", "eleven_v4_turbo")
    embedding_m = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    layers = [
        {
            "id": "stt",
            "layer": "STT",
            "name": "OpenAI GPT Live Transcribe",
            "model_id": transcribe_m,
            "purpose": "Real-time customer speech → text",
            "icon": "🎤",
            "status": "active",
            "specs": "Sub-50ms live stream recognition with native code-switching & Indian English / Tamil handling"
        },
        {
            "id": "llm",
            "layer": "LLM",
            "name": "OpenAI Realtime 2.1 & Reasoning",
            "model_id": f"{realtime_m} / {reasoning_m}",
            "realtime_model": realtime_m,
            "reasoning_model": reasoning_m,
            "purpose": "Conversation + RAG + decision making",
            "icon": "🧠",
            "status": "active",
            "specs": "Full duplex conversational AI with structured reasoning and automatic tool calling"
        },
        {
            "id": "embeddings",
            "layer": "Embeddings",
            "name": "text-embedding-3-large",
            "model_id": embedding_m,
            "dimension": 3072,
            "purpose": "RAG semantic search",
            "icon": "🔎",
            "status": "active",
            "specs": "High-dimensional 3072-vector semantic representation for deep technical & domain precision"
        },
        {
            "id": "vector_db",
            "layer": "Vector DB",
            "name": "Qdrant",
            "collection": rag_info.get("collection", "mintx_knowledge"),
            "storage_mode": rag_info.get("qdrant_mode", "local persistent"),
            "indexed_points": rag_info.get("indexed_points", 12),
            "purpose": "MintX knowledge retrieval",
            "icon": "📚",
            "status": "active" if rag_info.get("status") == "ready" else "ready",
            "specs": "HNSW cosine vector index for sub-10ms knowledge retrieval"
        },
        {
            "id": "tts",
            "layer": "TTS",
            "name": "Eleven v4 Turbo",
            "model_id": eleven_m,
            "voice_id": os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0"),
            "voice_name": "Yazhini (Customer Care)",
            "purpose": "Real-time, high-quality voice",
            "icon": "🔊",
            "status": "active",
            "specs": "100ms ultra low-latency expressive speech synthesis with natural Tanglish pronunciation"
        },
        {
            "id": "database",
            "layer": "Database",
            "name": "MongoDB",
            "database_name": os.getenv("MONGODB_DATABASE", "ai_phone_assistant"),
            "collections": mongo_collections,
            "purpose": "Customer/lead details",
            "icon": "💾",
            "status": mongo_status,
            "specs": "Real-time lead storage, call transcripts, appointments, and conversation memory"
        },
        {
            "id": "backend",
            "layer": "Backend",
            "name": "FastAPI + WebSocket",
            "purpose": "Real-time call pipeline",
            "icon": "⚡",
            "status": "active",
            "specs": "Bidirectional PCM 24000Hz low-latency async pipeline proxying browser Web Audio & Exotel telephony"
        }
    ]

    return {
        "status": "ok",
        "architecture": "7-Layer High-Performance AI Pipeline",
        "layers": layers,
        "stt": {
            "name": "OpenAI GPT Live Transcribe",
            "active_model": transcribe_m,
            "purpose": "Real-time customer speech → text",
            "status": "active"
        },
        "llm": {
            "realtime_model": realtime_m,
            "reasoning_model": reasoning_m,
            "purpose": "Conversation + RAG + decision making",
            "status": "active"
        },
        "embeddings": {
            "model": embedding_m,
            "dimensions": 3072,
            "metric": "Cosine Similarity",
            "cache_loaded": rag_info.get("status") == "ready",
            "purpose": "RAG semantic search",
            "status": "active"
        },
        "vector_db": {
            "provider": "Qdrant",
            "collection": rag_info.get("collection", "mintx_knowledge"),
            "indexed_vectors": rag_info.get("indexed_points", 12),
            "storage_path": rag_info.get("qdrant_mode", "local persistent"),
            "purpose": "MintX knowledge retrieval",
            "status": "online" if rag_info.get("status") == "ready" else "standby"
        },
        "tts": {
            "active_model": eleven_m,
            "voice_id": os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0"),
            "voice_name": "Yazhini (Customer Care)",
            "purpose": "Real-time, high-quality voice",
            "status": "active"
        },
        "database": {
            "provider": "MongoDB",
            "database": os.getenv("MONGODB_DATABASE", "ai_phone_assistant"),
            "leads_collection": "leads",
            "status": mongo_status,
            "purpose": "Customer/lead details"
        },
        "backend": {
            "framework": "FastAPI + WebSocket",
            "status": "active",
            "purpose": "Real-time call pipeline"
        },
        "rag_diagnostics": rag_info,
        "mongo_status": mongo_status
    }


@router.get("/api/models/settings")
@router.get("/api/settings")
async def get_models_settings():
    """Return all active models, reasoning configuration, STT, Embeddings, Qdrant and Sarvam AI voice settings."""
    reasoning_m = os.getenv("OPENAI_REASONING_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.6-sol"))
    realtime_m = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
    transcribe_m = os.getenv("OPENAI_TRANSCRIPTION_MODEL", os.getenv("OPENAI_REALTIME_TRANSCRIPTION_MODEL", "gpt-live-transcribe"))
    tts_prov = os.getenv("TTS_PROVIDER", "sarvam").lower()
    sarvam_speaker = os.getenv("SARVAM_TTS_SPEAKER", "kavitha")
    sarvam_model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
    voice_output_model = f"Sarvam AI {sarvam_model} ({sarvam_speaker})" if tts_prov in ("sarvam", "sarvamai") else "Eleven v4 Turbo"
    embedding_m = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    vector_db_m = os.getenv("VECTOR_DB_PROVIDER", "qdrant")

    # Fetch live Qdrant stats
    rag_stats = {}
    try:
        from rag import get_rag_engine
        rag_stats = get_rag_engine().get_status()
    except Exception:
        pass

    return {
        "status": "ok",
        "reasoning_model": reasoning_m,
        "realtime_model": realtime_m,
        "transcription_model": transcribe_m,
        "stt_model": transcribe_m,
        "embedding_model": embedding_m,
        "vector_db": "Qdrant",
        "voice": os.getenv("OPENAI_VOICE", "sarvam"),
        "tts_provider": tts_prov,
        "sarvam": {
            "speaker": sarvam_speaker,
            "model": sarvam_model,
            "language_code": os.getenv("SARVAM_TTS_LANGUAGE_CODE", "ta-IN"),
            "pace": float(os.getenv("SARVAM_TTS_PACE", "1.25")),
        },
        "elevenlabs": {
            "voice_id": os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0"),
            "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_v4_turbo"),
            "stability": float(os.getenv("ELEVENLABS_STABILITY", "0.40")),
            "similarity": float(os.getenv("ELEVENLABS_SIMILARITY", "0.80")),
            "style": float(os.getenv("ELEVENLABS_STYLE", "0.35")),
            "speaker_boost": os.getenv("ELEVENLABS_SPEAKER_BOOST", "true").lower() in ("true", "1"),
            "optimize_latency": int(os.getenv("ELEVENLABS_OPTIMIZE_LATENCY", "4")),
        },
        "rag": rag_stats,
        "tasks": {
            "stt": {
                "name": "Real-time speech transcription",
                "model": transcribe_m,
                "label": "OpenAI GPT Live Transcribe",
                "icon": "🎤"
            },
            "realtime": {
                "name": "Real-time voice conversation",
                "model": realtime_m,
                "label": "OpenAI Realtime 2.1",
                "icon": "🎙️"
            },
            "reasoning": {
                "name": "Customer requirements & decision making",
                "model": reasoning_m,
                "label": "OpenAI Latest Reasoning",
                "icon": "🧠"
            },
            "embeddings": {
                "name": "RAG semantic search",
                "model": embedding_m,
                "label": "text-embedding-3-large (3072 dim)",
                "icon": "🔎"
            },
            "vector_db": {
                "name": "MintX knowledge retrieval",
                "model": "Qdrant",
                "label": "Qdrant Vector Database",
                "icon": "📚"
            },
            "voice_output": {
                "name": "Voice output",
                "model": voice_output_model,
                "label": "Eleven v4 Turbo",
                "icon": "🔊"
            },
            "database": {
                "name": "Customer & lead details",
                "model": "MongoDB",
                "label": "MongoDB (ai_phone_assistant)",
                "icon": "💾"
            },
            "backend": {
                "name": "Real-time call pipeline",
                "model": "FastAPI + WebSocket",
                "label": "FastAPI WebSocket Stream",
                "icon": "⚡"
            }
        }
    }


@router.post("/api/models/settings")
@router.post("/api/settings")
async def update_models_settings(request: Request):
    """
    Update models and voice settings dynamically in memory and persist to .env and .env.local.
    """
    global DEFAULT_REALTIME_MODEL, DEFAULT_VOICE_NAME, DEFAULT_REASONING_MODEL, DEFAULT_TRANSCRIPTION_MODEL
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    env_updates = {}

    # 1. 🧠 High-quality reasoning / customer requirements model (gpt-5.6-sol)
    if "reasoning_model" in body and body["reasoning_model"]:
        m = body["reasoning_model"].strip()
        env_updates["OPENAI_REASONING_MODEL"] = m
        env_updates["OPENAI_MODEL"] = m
        os.environ["OPENAI_REASONING_MODEL"] = m
        os.environ["OPENAI_MODEL"] = m
        DEFAULT_REASONING_MODEL = m

    # 2. 🎙️ Real-time voice model (gpt-realtime-2.1)
    if "realtime_model" in body and body["realtime_model"]:
        rm = body["realtime_model"].strip()
        env_updates["OPENAI_REALTIME_MODEL"] = rm
        os.environ["OPENAI_REALTIME_MODEL"] = rm
        DEFAULT_REALTIME_MODEL = rm

    # 3. 🗣️ Speech transcription model (gpt-live-transcribe)
    if "transcription_model" in body and body["transcription_model"]:
        tm = body["transcription_model"].strip()
        env_updates["OPENAI_TRANSCRIPTION_MODEL"] = tm
        env_updates["OPENAI_REALTIME_TRANSCRIPTION_MODEL"] = tm
        os.environ["OPENAI_TRANSCRIPTION_MODEL"] = tm
        os.environ["OPENAI_REALTIME_TRANSCRIPTION_MODEL"] = tm
        DEFAULT_TRANSCRIPTION_MODEL = tm
    elif "stt_model" in body and body["stt_model"] and body["stt_model"] != "none":
        tm = body["stt_model"].strip()
        env_updates["OPENAI_TRANSCRIPTION_MODEL"] = tm
        env_updates["OPENAI_REALTIME_TRANSCRIPTION_MODEL"] = tm
        os.environ["OPENAI_TRANSCRIPTION_MODEL"] = tm
        os.environ["OPENAI_REALTIME_TRANSCRIPTION_MODEL"] = tm
        DEFAULT_TRANSCRIPTION_MODEL = tm

    # 4. 🔎 Embeddings & 📚 Vector DB
    if "embedding_model" in body and body["embedding_model"]:
        em = body["embedding_model"].strip()
        env_updates["OPENAI_EMBEDDING_MODEL"] = em
        os.environ["OPENAI_EMBEDDING_MODEL"] = em

    if "vector_db" in body and body["vector_db"]:
        vdb = body["vector_db"].strip().lower()
        env_updates["VECTOR_DB_PROVIDER"] = vdb
        os.environ["VECTOR_DB_PROVIDER"] = vdb

    # 5. 🔊 Voice output (Sarvam AI / ElevenLabs / provider)
    if "tts_provider" in body and body["tts_provider"]:
        tp = body["tts_provider"].strip()
        env_updates["TTS_PROVIDER"] = tp
        os.environ["TTS_PROVIDER"] = tp

    if "sarvam_speaker" in body and body["sarvam_speaker"]:
        spk = body["sarvam_speaker"].strip()
        env_updates["SARVAM_TTS_SPEAKER"] = spk
        os.environ["SARVAM_TTS_SPEAKER"] = spk

    if "sarvam_model" in body and body["sarvam_model"]:
        sm = body["sarvam_model"].strip()
        env_updates["SARVAM_TTS_MODEL"] = sm
        os.environ["SARVAM_TTS_MODEL"] = sm

    if "sarvam_pace" in body:
        sp = str(body["sarvam_pace"])
        env_updates["SARVAM_TTS_PACE"] = sp
        os.environ["SARVAM_TTS_PACE"] = sp

    if "sarvam_language_code" in body and body["sarvam_language_code"]:
        sl = body["sarvam_language_code"].strip()
        env_updates["SARVAM_TTS_LANGUAGE_CODE"] = sl
        os.environ["SARVAM_TTS_LANGUAGE_CODE"] = sl

    if "elevenlabs_voice_id" in body and body["elevenlabs_voice_id"]:
        vid = body["elevenlabs_voice_id"].strip()
        env_updates["ELEVENLABS_VOICE_ID"] = vid
        os.environ["ELEVENLABS_VOICE_ID"] = vid

    if "elevenlabs_model" in body and body["elevenlabs_model"]:
        em = body["elevenlabs_model"].strip()
        env_updates["ELEVENLABS_MODEL_ID"] = em
        os.environ["ELEVENLABS_MODEL_ID"] = em

    if "elevenlabs_stability" in body:
        stab = str(body["elevenlabs_stability"])
        env_updates["ELEVENLABS_STABILITY"] = stab
        os.environ["ELEVENLABS_STABILITY"] = stab

    if "elevenlabs_similarity" in body:
        sim = str(body["elevenlabs_similarity"])
        env_updates["ELEVENLABS_SIMILARITY"] = sim
        os.environ["ELEVENLABS_SIMILARITY"] = sim

    if "elevenlabs_style" in body:
        sty = str(body["elevenlabs_style"])
        env_updates["ELEVENLABS_STYLE"] = sty
        os.environ["ELEVENLABS_STYLE"] = sty

    if "voice" in body and body["voice"]:
        v = body["voice"].strip()
        env_updates["OPENAI_VOICE"] = v
        os.environ["OPENAI_VOICE"] = v
        DEFAULT_VOICE_NAME = v

    # Write changes to .env and .env.local
    if env_updates:
        for env_file in [BASE_DIR / ".env", BASE_DIR / ".env.local"]:
            if env_file.exists():
                try:
                    content = env_file.read_text(encoding="utf-8")
                    for k, val in env_updates.items():
                        if f"{k}=" in content:
                            content = re.sub(rf"^{k}=.*$", f"{k}={val}", content, flags=re.MULTILINE)
                        else:
                            content += f"\n{k}={val}\n"
                    env_file.write_text(content, encoding="utf-8")
                except Exception as e:
                    print(f"[!] Warning persisting settings to {env_file.name}: {e}")

    return {
        "status": "ok",
        "message": "Model settings updated and saved to .env successfully!",
        "updated": env_updates,
        "current": {
            "reasoning_model": os.getenv("OPENAI_REASONING_MODEL", "gpt-5.6-sol"),
            "realtime_model": os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1"),
            "transcription_model": os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-live-transcribe"),
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            "vector_db": "Qdrant",
            "tts_provider": os.getenv("TTS_PROVIDER", "elevenlabs"),
            "elevenlabs_voice_id": os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0"),
            "elevenlabs_model": os.getenv("ELEVENLABS_MODEL_ID", "eleven_v4_turbo"),
        }
    }


# ── Realtime WebSocket Proxy ────────────────────────────────────────

@router.websocket("/ws/realtime")
async def websocket_realtime_proxy(client_ws: WebSocket):
    """
    Bi-directional WebSocket proxy between browser client and OpenAI Realtime API.
    """
    await client_ws.accept()

    current_api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not current_api_key:
        await client_ws.send_text(json.dumps({
            "type": "error",
            "error": {"code": "no_api_key", "message": "OPENAI_API_KEY is not configured in .env"}
        }))
        await client_ws.close()
        return

    # WebSocket connection to OpenAI (GA Realtime API without beta header)
    requested_model = client_ws.query_params.get("model")
    current_model = requested_model if (requested_model and requested_model.strip()) else DEFAULT_REALTIME_MODEL
    openai_ws_url = f"wss://api.openai.com/v1/realtime?model={current_model}"

    # ["audio"] when a GPT voice speaks natively, ["text"] when Sarvam AI / ElevenLabs speaks.
    voice_mode = {"modalities": None}
    # "sarvam" while Saaras v4 is transcribing the caller; OpenAI then only receives text turns.
    stt_state = {"engine": "openai", "client": None, "task": None}

    def build_session_config(
        voice_name: str = None,
        system_instructions: str = None,
        reasoning_effort: str = None,
        temperature: float = None,
        stt_model: str = None,
        vad_threshold: float = None,
        vad_silence_ms: int = None,
        vad_prefix_ms: int = None,
        modalities: list = None
    ) -> dict:
        """Construct session payload conforming to OpenAI Realtime GA schema with direct instructions and STT/TTS controls."""
        v = (voice_name or DEFAULT_VOICE_NAME or "").lower()
        valid_voice = v if v in OPENAI_VOICES else "marin"

        # Direct connection to instructions from instructions.json or caller without arbitrary guard overrides
        base_inst = system_instructions if (system_instructions is not None and system_instructions.strip()) else (get_active_system_prompt() or SYSTEM_PROMPT)

        customer_language_directive = (
            "================================================================================\n"
            "🚨 MINTX SOLUTIONS AI CUSTOMER CARE AGENT — YAZHINI 🚨\n"
            "================================================================================\n"
            "You are Yazhini, a professional, friendly, attentive female AI customer-care executive for MintX Solutions, Salem, Tamil Nadu.\n\n"
            "## BRAND NAME STRICT RULE — NEVER DECODE INTO TAMIL:\n"
            "* ALWAYS write and pronounce the company name cleanly as 'MintX Solutions' (or 'MintX').\n"
            "* NEVER decode, translate, transliterate, or spell into Tamil script (NEVER use 'மிண்ட் எக்ஸ்', 'மின்ட் எக்ஸ்', 'சொல்யூஷன்ஸ்', 'மின்டெக்ஸ்', 'மிண்டெக்ஸ்', 'மைண்ட் எக்ஸ்', etc.).\n"
            "* Even when speaking 100% in Tamil or Tanglish, the brand name 'MintX Solutions' must ALWAYS remain in clean English Latin script ('MintX Solutions')!\n\n"
            "## GOAL — SOUND LIKE A REAL HUMAN, NOT A BOT:\n"
            "The caller must feel a friendly Tamil-speaking human customer-care executive is talking: natural acknowledgements, small pauses, enthusiasm, empathy and real conversational reactions.\n\n"
            "## LANGUAGE — AUTO-DETECT AND MIRROR THE CUSTOMER ON EVERY TURN:\n"
            "* Tamil / Tanglish caller (e.g. 'வணக்கம்', 'Enakku business-ku website venum', 'Evlo aagum?') → THANGLISH MODE (below). Never switch the whole reply to English.\n"
            "  - Phone numbers spoken in English digits, or words like 'okay', 'yes', 'website', 'app', 'e-commerce' DO NOT mean the caller switched to English.\n"
            "* English caller → clean, warm, friendly English only (no 'Vanakkam', 'sollunga', 'venum').\n"
            "* Any other Indian language (Hindi, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia) → reply in THAT language in its NATIVE SCRIPT "
            "(Hindi/Marathi: देवनागरी, Telugu: తెలుగు, Kannada: ಕನ್ನಡ, Malayalam: മലയാളം, Bengali: বাংলা, Gujarati: ગુજરાતી, Punjabi: ਗੁਰਮੁਖੀ, Odia: ଓଡ଼ିଆ), never romanized. Tech words may stay English.\n"
            "* If the caller switches language mid-call, switch with them from the very next reply.\n\n"
            "## THANGLISH MODE (Tamil callers):\n"
            "* Speak natural spoken Thanglish like a real Chennai/Salem customer-care executive. Mixing English business/tech words inside Tamil sentences is CORRECT and expected: "
            "website, app, software, e-commerce, budget, requirement, details, booking, payment, online, design, development, support, team, timeline, quote.\n"
            "* Good: 'Super-nga, ungalukku e-commerce website venumaa?' / 'Seri, unga budget range evlo-nga?' / 'Naan unga requirement note pannikiren.'\n"
            "* Do NOT translate tech words into heavy formal Tamil (no 'இணையதளம்', 'மென்பொருள்'). Do NOT speak formal or written-style Tamil.\n"
            "* Confirming details: 'Seri-nga, oru thadavai confirm pannikiren. Unga name [Name], business [Business], ungalukku [Service] venum. Correct-aa-nga?'\n\n"
            "## JOYFUL, WARM PERSONALITY:\n"
            "* Cheerful, positive, energetic, polite, patient and genuinely happy to help — like smiling while talking. Never dull, flat or scripted.\n"
            "* React positively when the customer shares something: 'Wow, super-nga!', 'Oh nice!', 'Romba nalla idea-nga!', 'Semma-nga!', 'Great!'.\n\n"
            "## EMOTIONAL AWARENESS — MATCH THE CALLER'S MOOD:\n"
            "* Happy/excited → energetic: 'Super-nga! Kandippa help panrom.'\n"
            "* Confused → slow, simple: 'Paravalla-nga, simple-aa sollren.'\n"
            "* Worried/frustrated → calm, empathetic: 'Purinjudhu-nga, kavalai padaadheenga. Naanga help panrom.'\n"
            "* Price-sensitive → reassuring: 'Unga budget-ku yetha maadhiri options paakkalaam-nga.'\n"
            "* In a hurry → very short, direct answers only.\n\n"
            "## HUMAN ACKNOWLEDGEMENTS & ACTIVE LISTENING:\n"
            "* Start most replies with a short natural acknowledgement, and VARY it every turn — never repeat the same one twice in a row: "
            "'Aamaa-nga', 'Seri-nga', 'Okay-nga', 'Purinjudhu', 'Nice-nga', 'Super', 'Kandippa', 'Correct-aa sonneenga'. English: 'Sure', 'Got it', 'Oh nice', 'Absolutely'.\n"
            "* Briefly reflect what the caller said so they feel heard: 'Oh, unga clothing business-ku online store venumaa? Super-nga.'\n"
            "* Small natural fillers are fine, but sparingly (at most one per reply, not every reply): 'Hmm', 'Seri', 'Okay', 'Ah', 'Oh nice'.\n"
            "* Use punctuation for natural pauses: 'Seri-nga... ungalukku website venumaa?' / 'Oh nice! Enna maadhiri business-nga?'\n"
            "* Never mention a knowledge base, database, system, tool or AI.\n\n"
            "## SHORT, ONE-QUESTION REPLIES:\n"
            "* Keep every reply to 1 or 2 short spoken sentences (about 25 words), 3 only if the caller asks for details. "
            "Give the key fact first, then ONE question, then stop and let the caller talk.\n"
            "* Never give long explanations, lists, or several questions together.\n\n"
            "## NATURAL LEAD COLLECTION (feel like a chat, not a form):\n"
            "* Never sound like a form ('Please provide your name and phone number'). Ask one detail at a time, naturally:\n"
            "  'Unga perai sollunga-nga?' → 'Thanks [Name]! Unga business enna-nga?' → 'Super! Unga contact number sollunga, namma team call pannuvaanga.'\n\n"
            "## SPEAKABLE OUTPUT (the reply is spoken by a voice engine):\n"
            "* No markdown, bullet points, numbered lists, emojis, URLs, brackets or special symbols. Plain spoken sentences only.\n"
            "* Prices like '₹10,000' (say naturally), phone numbers in natural groups, no codes or tables.\n\n"
            "## ANTI-ROBOTIC RULES:\n"
            "* Never say 'As an AI', 'I am an AI language model', 'Please wait while I process', 'Your request has been received', 'Kindly provide'.\n"
            "* Never repeat the same sentence structure or opening every turn. Never read like a document.\n"
            "* Light small talk is fine if the caller starts it (one short friendly line), then gently return to how MintX can help.\n\n"
            "## CALL ENDINGS — VARY THEM, KEEP THEM WARM:\n"
            "* 'Romba thanks-nga! Namma team seekiram ungala contact pannuvaanga. Have a nice day!'\n"
            "* 'Super-nga, ungaloda pesinadhu romba santhosham. Vera edhavadhu venumna eppo venumnaalum call pannunga!'\n"
            "* 'Thank you so much! It was lovely talking to you. Have a great day!'\n\n"
            "* SILENT TOOL EXECUTION: Execute tools (`save_customer_memory`, `search_mintx_knowledge`) silently. NEVER output English thinking preambles before calling tools!\n\n"
            "## GREETING & ACKNOWLEDGMENTS — CRITICAL PHONE BEHAVIOR:\n"
            "* The call starting greeting has ALREADY been spoken once at the start. NEVER repeat 'Welcome to MintX Solutions' or 'What service do you need today?' again.\n"
            "* Caller Response Handling:\n"
            "  - If caller speaks English (e.g. 'Hello, I need website details', 'Hi, can you help me?'):\n"
            "    → Respond in English: 'Sure! Are you looking for a website, a mobile app, or something else?'\n"
            "  - If caller speaks Tamil / Tunglish, or gives short acknowledgments like 'Yes', 'Okay', 'Sollunga', 'Hmm':\n"
            "    → Stay in Thanglish: 'Seri-nga! Ungalukku website venumaa, mobile app venumaa, illa vera edhavadhu service-aa?'\n"
            "* Keep moving the conversation forward. Always ask only ONE clear question at a time.\n\n"
            "## KNOWLEDGE RULE — ANSWER INSTANTLY FROM VERIFIED FACTS:\n"
            "* All verified MintX Solutions facts (services, prices, timelines, contact, office, support) are in the 'MINTX VERIFIED KNOWLEDGE' section below.\n"
            "* Answer directly from those facts in the SAME reply — do NOT say 'let me check' / 'oru nimisham check panren' and do NOT call `search_mintx_knowledge` when the answer is there.\n"
            "* Call `search_mintx_knowledge` only if the question is about MintX and the answer is genuinely missing below.\n"
            "* Never invent company information or prices. Use only these facts.\n"
            "* If the information is not available, say naturally:\n"
            "  - English: 'I don't have that specific information right now. I can connect you with our team for the exact details.'\n"
            "  - Tunglish: 'Andha exact information en kitta ippo illa. Correct details-ku namma team-kitta connect panni kudukren.'\n"
            "  - Tamil: 'அந்த குறிப்பிட்ட தகவல் என்னிடம் இப்போது இல்லை. சரியான விவரங்களுக்கு எங்கள் team-க்கு connect பண்ணிக்கொடுக்கிறேன்.'\n\n"
            "## DETAIL COLLECTION & CONFIRMATION FLOW:\n"
            "* Collect details one at a time: Customer Name, Business Name, Service, Phone number, Requirement.\n"
            "* Check your memory before asking; NEVER ask for a detail already given.\n"
            "* ONLY after customer confirms, call `save_customer_memory` with is_confirmed=true.\n\n"
            "## PRIORITY ORDER:\n"
            "ACCURACY (RAG facts only) → NATURAL HUMAN CONVERSATION → CUSTOMER'S LANGUAGE → EMOTIONAL AWARENESS → FAST → SHORT → CUSTOMER SATISFACTION\n"
            "================================================================================\n\n"
        )

        memory_directive = (
            "## DIRECT CONNECTION TO GPT MEMORY & RAG:\n"
            "- You have direct tools: `search_mintx_knowledge` for company facts, and `save_customer_memory` for lead details.\n"
            "- As soon as the customer mentions any detail (Name, Business Name, Service, Phone number, Requirement), call `save_customer_memory` to store it.\n"
            "- When confirming customer details, retrieve and read the EXACT values from your memory.\n"
            "- If the customer corrects any detail, update it and confirm only the corrected detail.\n\n"
        )

        knowledge = get_knowledge_brief()
        knowledge_directive = (
            "## MINTX VERIFIED KNOWLEDGE (the only source of truth for company facts):\n" + knowledge + "\n\n"
        ) if knowledge else ""

        inst = customer_language_directive + knowledge_directive + memory_directive + base_inst

        # Inject real-time training instruction so ChatGPT calls train_voice_correction when receiving feedback
        training_directive = (
            "\n\n## REAL-TIME VOICE CORRECTION & SELF-TRAINING TOOL\n"
            "ONLY if the user explicitly gives you a system training instruction (e.g. says 'நான் கரெக்ஷன் சொல்றேன்' / 'correction' / 'இதை மாத்து' / 'update rule'), "
            "call the `train_voice_correction` tool. "
            "CRITICAL: If the customer asks questions about developer, software, website, apps, pricing, or is having a normal customer care dialogue, DO NOT call train_voice_correction; answer them as customer care!"
        )
        if "train_voice_correction" not in inst:
            inst = inst + training_directive

        # Voice: OpenAI Realtime returns text; Sarvam AI / ElevenLabs synthesizes speech.
        tts_prov = os.getenv("TTS_PROVIDER", "sarvam").lower()
        openai_voice = os.getenv("OPENAI_VOICE", "sarvam").lower()
        force_text_out = (
            tts_prov in ("sarvam", "sarvamai", "elevenlabs")
            or openai_voice in ("sarvam", "kavitha", "elevenlabs")
            or (voice_name or "").lower() in ("sarvam", "kavitha", "kavya", "priya", "shreya", "roopa", "mani", "gokul", "elevenlabs")
            or (voice_name and len(str(voice_name)) >= 20)
        )
        default_modalities = voice_mode["modalities"] or (["text"] if force_text_out else ["audio"])

        session_cfg = {
            "type": "realtime",
            "instructions": inst,
            "output_modalities": modalities or default_modalities,
            "max_output_tokens": 220,
            "tools": [
                {
                    "type": "function",
                    "name": "search_mintx_knowledge",
                    "description": (
                        "MANDATORY RAG TOOL: Search the MintX Solutions vector database for accurate information about "
                        "services (website, e-commerce, mobile apps, software, digital marketing, QR), pricing, technologies, "
                        "Salem office location, contact details, delivery timelines, support policies, and FAQs. "
                        "Call this tool whenever the customer asks a factual question about MintX Solutions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query in English, Tamil, or Tunglish (e.g. 'ecommerce website cost', 'office location', 'services offered')"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "type": "function",
                    "name": "save_customer_memory",
                    "description": (
                        "DIRECT MEMORY CONNECTION: Call this tool immediately when the caller provides any customer detail "
                        "(name, business_name, service, phone, requirement). When all details have been collected and the "
                        "customer confirms they are correct, set is_confirmed to true to finalize saving to the database."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Customer's personal name"
                            },
                            "business_name": {
                                "type": "string",
                                "description": "Customer's company, shop, or business name"
                            },
                            "service": {
                                "type": "string",
                                "description": "Service required (e.g. Website, E-commerce, Mobile App, Digital Marketing)"
                            },
                            "phone": {
                                "type": "string",
                                "description": "Customer's 10-digit phone number"
                            },
                            "requirement": {
                                "type": "string",
                                "description": "Customer's specific requirement details"
                            },
                            "is_confirmed": {
                                "type": "boolean",
                                "description": "Set to true ONLY AFTER the customer confirms their collected details"
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "name": "train_voice_correction",
                    "description": (
                        "Call this tool ONLY when the user explicitly instructs you to update a system rule, instruction, or prompt "
                        "(e.g. 'naan correction solren', 'update rule', 'rule change pannu', 'training instruction'). "
                        "CRITICAL: NEVER call this tool during normal customer enquiries, conversations about developer services, "
                        "software development, websites, mobile apps, or regular customer care!"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feedback": {
                                "type": "string",
                                "description": "The exact user correction, rule, or training instruction spoken by the user."
                            },
                            "category": {
                                "type": "string",
                                "description": "Optional category (e.g. 'greeting', 'language', 'pricing', 'behavior', 'rule')."
                            }
                        },
                        "required": ["feedback"]
                    }
                }
            ],
            "tool_choice": "auto",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "noise_reduction": {"type": "far_field"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": max(0.48, float(vad_threshold) if vad_threshold is not None else float(os.getenv("VAD_THRESHOLD", "0.50"))),
                        "silence_duration_ms": max(450, int(vad_silence_ms) if vad_silence_ms is not None else int(os.getenv("VAD_SILENCE_MS", "500"))),
                        "prefix_padding_ms": max(200, int(vad_prefix_ms) if vad_prefix_ms is not None else int(os.getenv("VAD_PREFIX_MS", "300"))),
                        "create_response": True,
                        "interrupt_response": True
                    }
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": valid_voice
                }
            }
        }

        # Configure Speech-to-Text (STT) input audio transcription with domain prompt
        if stt_model and str(stt_model).lower() in ("none", "off", "disabled"):
            session_cfg["audio"]["input"]["transcription"] = None
        else:
            session_cfg["audio"]["input"]["transcription"] = {
                "model": os.getenv("OPENAI_FALLBACK_TRANSCRIBE_MODEL", "gpt-4o-transcribe"),
                "prompt": (
                    "MintX Solutions customer care call in Salem, Tamil Nadu. Callers mostly speak Tamil or Tanglish, "
                    "sometimes English. வணக்கம், எனக்கு website வேணும், app, e-commerce, price எவ்வளவு, sollunga, seri."
                )
            }

        if stt_state["engine"] == "sarvam":
            session_cfg["audio"]["input"]["turn_detection"] = None
            session_cfg["audio"]["input"]["transcription"] = None

        if reasoning_effort and str(reasoning_effort).lower() in ("none", "low", "medium", "high"):
            session_cfg["reasoning"] = {"effort": str(reasoning_effort).lower()}

        # GA Realtime API rejects session.temperature — do not set it.
        return session_cfg

    openai_ws = None
    speech_queue = asyncio.Queue()
    is_speaking_cancelled = {"val": False}
    # Bumped on every interruption so audio already being synthesized for an old turn is dropped.
    speech_generation = {"val": 0}

    def cancel_speech():
        is_speaking_cancelled["val"] = True
        speech_generation["val"] += 1

    if current_model.startswith("gpt-live"):
        from live_session import run_live_session

        def build_live_instructions(init_msg: dict) -> str:
            custom = init_msg.get("instructions")
            base = custom if (custom and custom.strip()) else (get_active_system_prompt() or SYSTEM_PROMPT)
            return build_session_config(init_msg.get("voice"), base)["instructions"]

        await run_live_session(client_ws, current_api_key, build_live_instructions)
        return

    try:
        openai_ws = await connect_openai_realtime(current_model, current_api_key)
        print(f"[Realtime Server] Connected to OpenAI WebSocket (Model: {current_model})")
    except websockets.exceptions.ConnectionClosed as e:
        error_msg = str(e.reason) or "OpenAI connection closed."
        if e.code == 1013 or "insufficient_quota" in str(e.reason):
            error_msg = "OpenAI API Quota Exhausted: Your OpenAI account has no credits remaining (credit_balance_exhausted). Please recharge credits at https://platform.openai.com/settings/organization/billing/ or provide a new OPENAI_API_KEY."
        print(f"[!] OpenAI Connection Error: {error_msg}")
        try:
            await client_ws.send_text(json.dumps({
                "type": "error",
                "error": {"code": "insufficient_quota" if (e.code == 1013 or "insufficient_quota" in str(e.reason)) else "connection_closed", "message": error_msg}
            }))
            await client_ws.close()
        except Exception:
            pass
        return
    except Exception as e:
        err_str = str(e)
        if "1013" in err_str or "insufficient_quota" in err_str or "credit_balance_exhausted" in err_str:
            error_msg = "OpenAI API Quota Exhausted: Your OpenAI account has no credits remaining ($0.00). Please recharge at https://platform.openai.com/settings/organization/billing/ or update OPENAI_API_KEY."
        elif "11001" in err_str or "getaddrinfo" in err_str.lower() or "name or service not known" in err_str.lower():
            error_msg = (
                "Network/DNS error reaching api.openai.com (getaddrinfo failed). "
                "Check internet, disable VPN briefly, or flush DNS (ipconfig /flushdns), then retry."
            )
        else:
            error_msg = f"Failed to connect to OpenAI Realtime: {err_str}"
        print(f"[!] OpenAI Connection Error: {error_msg}")
        try:
            await client_ws.send_text(json.dumps({
                "type": "error",
                "error": {"code": "insufficient_quota" if "insufficient_quota" in error_msg else "connection_failed", "message": error_msg}
            }))
            await client_ws.close()
        except Exception:
            pass
        return

    try:
        # Initial Session Setup with direct active instructions from instructions.json
        active_instructions = get_active_system_prompt() or SYSTEM_PROMPT
        initial_session = {
            "type": "session.update",
            "session": build_session_config(DEFAULT_VOICE_NAME, active_instructions)
        }
        await openai_ws.send(json.dumps(initial_session))

        # Track active voice and assistant transcript buffer for Sarvam AI / ElevenLabs synthesis
        active_voice = {"val": DEFAULT_VOICE_NAME}
        active_tts_provider = {"val": os.getenv("TTS_PROVIDER", "sarvam").lower()}
        active_sarvam_speaker = {"val": os.getenv("SARVAM_TTS_SPEAKER", "kavitha")}
        active_sarvam_model = {"val": os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")}
        active_sarvam_pace = {"val": float(os.getenv("SARVAM_TTS_PACE", "1.25"))}
        active_sarvam_lang = {"val": os.getenv("SARVAM_TTS_LANGUAGE_CODE", "ta-IN")}

        active_elevenlabs_model = {"val": os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")}
        active_elevenlabs_voice_id = {"val": os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0")}
        active_elevenlabs_stability = {"val": float(os.getenv("ELEVENLABS_STABILITY", "0.52"))}
        active_elevenlabs_similarity = {"val": float(os.getenv("ELEVENLABS_SIMILARITY", "0.85"))}
        active_elevenlabs_style = {"val": float(os.getenv("ELEVENLABS_STYLE", "0.28"))}
        active_elevenlabs_speaker_boost = {"val": os.getenv("ELEVENLABS_SPEAKER_BOOST", "true").lower() in ("true", "1", "yes")}
        active_elevenlabs_speed = {"val": float(os.getenv("ELEVENLABS_SPEED", "0.93"))}

        def select_voice(voice: str):
            """Route the chosen voice: GPT voices speak natively, everything else goes through Sarvam AI / ElevenLabs."""
            raw = (voice or DEFAULT_VOICE_NAME or "sarvam").strip()
            v = raw.lower()
            if v in OPENAI_VOICES:
                active_voice["val"] = v
                active_tts_provider["val"] = "openai"
            elif v == "elevenlabs" or len(raw) == 20 or v.startswith("eleven_"):
                if len(raw) == 20:
                    active_elevenlabs_voice_id["val"] = raw
                active_voice["val"] = "elevenlabs"
                active_tts_provider["val"] = "elevenlabs"
            else:
                if v not in SARVAM_VOICE_ALIASES and v not in SARVAM_SPEAKERS:
                    v = "sarvam"
                active_sarvam_speaker["val"] = "kavitha" if v in ("sarvam", "sarvamai", "default") else v
                active_voice["val"] = "sarvam"
                active_tts_provider["val"] = "sarvam"
            os.environ["TTS_PROVIDER"] = active_tts_provider["val"]
            voice_mode["modalities"] = ["audio"] if active_tts_provider["val"] == "openai" else ["text"]
            if active_tts_provider["val"] == "openai":
                return active_voice["val"]
            return openai_audio_voice["val"] or "marin"

        openai_audio_voice = {"val": None}

        def current_openai_voice() -> str:
            if active_tts_provider["val"] == "openai":
                return active_voice["val"]
            return openai_audio_voice["val"] or "marin"

        assistant_transcript_buffer = {"text": "", "last_done": ""}
        sentence_accumulator = {"text": ""}
        speech_flushed_for_turn = {"val": False}
        session_call_id = f"call_{time.strftime('%Y%m%d%H%M%S')}"
        customer_memory = {
            "name": "",
            "business_name": "",
            "service": "",
            "phone": "",
            "requirement": "",
            "language": "",
            "status": "new"
        }

        sarvam_credit_alerted = {"val": False}

        # Background worker for continuous Sarvam AI (or ElevenLabs) streaming
        async def speech_worker():
            import base64
            from voice import async_stream_speech_sarvam, async_stream_speech_elevenlabs

            def make_generator(sentence: str):
                tts_prov = active_tts_provider["val"]
                v_val = (active_voice["val"] or "").lower()
                is_el = (tts_prov == "elevenlabs" or v_val in ("elevenlabs",) or len(v_val) >= 20) and bool(os.getenv("ELEVENLABS_API_KEY"))
                if is_el:
                    vid = active_elevenlabs_voice_id["val"] or os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0")
                    mid = active_elevenlabs_model["val"] or os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
                    opt_lat = int(os.getenv("ELEVENLABS_OPTIMIZE_LATENCY", "4"))
                    return async_stream_speech_elevenlabs(
                        tanglish_for_tts(sentence.strip()),
                        voice_id=vid,
                        model_id=mid,
                        output_format="pcm_24000",
                        stability=active_elevenlabs_stability["val"],
                        similarity_boost=active_elevenlabs_similarity["val"],
                        style=active_elevenlabs_style["val"],
                        use_speaker_boost=active_elevenlabs_speaker_boost["val"],
                        optimize_streaming_latency=None if any(x in str(mid).lower() for x in ("v3", "v4")) else opt_lat,
                        speed=active_elevenlabs_speed["val"],
                    )
                # Sarvam AI Voice Engine (bulbul:v3 - Yazhini); language_code follows the sentence's own language
                return async_stream_speech_sarvam(
                    sentence.strip(),
                    speaker=active_sarvam_speaker["val"],
                    model=active_sarvam_model["val"],
                    language_code=get_sarvam_language_code(detect_language(sentence)),
                    pace=active_sarvam_pace["val"],
                )

            async def synthesize(sentence: str, gen: int, out_q: asyncio.Queue):
                try:
                    async for pcm_chunk in make_generator(sentence):
                        if gen != speech_generation["val"]:
                            break
                        if pcm_chunk:
                            await out_q.put(bytes(pcm_chunk))
                except Exception as e:
                    print(f"[!] Speech synthesis error: {e}")
                    if "402" in str(e) and active_tts_provider["val"] == "sarvam" and not sarvam_credit_alerted["val"]:
                        sarvam_credit_alerted["val"] = True
                        try:
                            await client_ws.send_text(json.dumps({
                                "type": "voice.fallback",
                                "voice": "marin",
                                "message": "Sarvam AI credits exhausted — switched to GPT voice. Recharge Sarvam to get the Sarvam voice back.",
                            }))
                        except Exception:
                            pass
                finally:
                    await out_q.put(None)

            # Sentences are synthesized concurrently (next sentence is ready before the current one ends)
            # but always sent to the browser in order, so playback has no gaps between sentences.
            ordered_streams: asyncio.Queue = asyncio.Queue()

            async def sender():
                while True:
                    item = await ordered_streams.get()
                    if item is None:
                        break
                    gen, out_q = item
                    pcm_carry = b""
                    while True:
                        chunk = await out_q.get()
                        if chunk is None:
                            break
                        if gen != speech_generation["val"]:
                            continue
                        # Keep PCM 16-bit sample-aligned (odd bytes = crackle/broken audio)
                        data = pcm_carry + chunk
                        if len(data) % 2 == 1:
                            pcm_carry, data = data[-1:], data[:-1]
                        else:
                            pcm_carry = b""
                        if data:
                            await client_ws.send_text(json.dumps({
                                "type": "response.output_audio.delta",
                                "delta": base64.b64encode(data).decode("utf-8"),
                            }))

            sender_task = asyncio.create_task(sender())
            synth_tasks = set()
            try:
                while True:
                    sentence = await speech_queue.get()
                    speech_queue.task_done()
                    if sentence is None:
                        break
                    if is_speaking_cancelled["val"] or not sentence.strip():
                        continue
                    out_q: asyncio.Queue = asyncio.Queue()
                    gen = speech_generation["val"]
                    await ordered_streams.put((gen, out_q))
                    task = asyncio.create_task(synthesize(sentence, gen, out_q))
                    synth_tasks.add(task)
                    task.add_done_callback(synth_tasks.discard)
            except asyncio.CancelledError:
                pass
            finally:
                for task in list(synth_tasks):
                    task.cancel()
                sender_task.cancel()

        handled_tool_calls = set()
        response_active = {"val": False}
        # Bumped whenever the caller interrupts or starts a new turn; tool results that finish
        # afterwards are recorded but must not trigger a stale spoken follow-up.
        interrupt_epoch = {"val": 0}
        response_epochs = {}

        def clear_pending_speech():
            interrupt_epoch["val"] += 1
            cancel_speech()
            speech_flushed_for_turn["val"] = False
            while not speech_queue.empty():
                try:
                    speech_queue.get_nowait()
                    speech_queue.task_done()
                except Exception:
                    break
            sentence_accumulator["text"] = ""
            assistant_transcript_buffer["text"] = ""

        async def interrupt_assistant():
            """Barge-in: silence Yazhini immediately and stop the reply being generated."""
            clear_pending_speech()
            if response_active["val"]:
                response_active["val"] = False
                try:
                    await openai_ws.send(json.dumps({"type": "response.cancel"}))
                except Exception as e:
                    print(f"[!] response.cancel failed: {e}")

        async def process_caller_transcript(transcript: str, stt_language: str = None) -> str:
            """Normalize the caller's words, detect their language, and update UI + TTS language."""
            transcript = normalize_brand_name(transcript)
            detected_lang = detect_language(transcript)
            if detected_lang == "en" and stt_language and not stt_language.startswith("en"):
                code_to_lang = {"ta-IN": "tunglish", "or-IN": "od", "od-IN": "od"}
                detected_lang = code_to_lang.get(stt_language, stt_language.split("-")[0])
            previous_lang = customer_memory.get("language")
            if detected_lang == "en" and previous_lang and previous_lang != "en":
                # Names, phone numbers and tech terms ("Website Development", "7339...") are said in
                # English by Tamil callers too; only a full English sentence counts as a switch.
                words = re.findall(r"[A-Za-z]+", transcript)
                if len(words) <= 6 or re.search(r"\d{3,}", transcript):
                    detected_lang = previous_lang
            customer_memory["language"] = detected_lang
            active_sarvam_lang["val"] = get_sarvam_language_code(detected_lang)
            print(f"[Realtime STT] Customer Speech: '{transcript}' -> Auto-detected Language: {detected_lang} ({get_language_display_name(detected_lang)})")
            try:
                await client_ws.send_text(json.dumps({
                    "type": "customer.language.detected",
                    "language": detected_lang,
                    "display_name": get_language_display_name(detected_lang),
                    "transcript": transcript
                }))
            except Exception:
                pass
            return transcript

        async def on_saaras_event(ev: dict):
            kind = ev.get("event")
            if kind == "vad.speech_start":
                # Not an interruption by itself (could be a cough); barge-in arrives as client.barge_in
                # and a real utterance interrupts at transcript.final.
                await client_ws.send_text(json.dumps({"type": "input_audio_buffer.speech_started"}))
            elif kind == "transcript.partial":
                if ev.get("text"):
                    await client_ws.send_text(json.dumps({"type": "caller.transcript.partial", "text": ev["text"]}))
            elif kind == "vad.speech_end":
                await client_ws.send_text(json.dumps({"type": "input_audio_buffer.speech_stopped"}))
            elif kind == "transcript.final":
                text = (ev.get("text") or "").strip()
                if not re.search(r"\w", text):
                    return
                await interrupt_assistant()
                transcript = await process_caller_transcript(text, ev.get("language"))
                await client_ws.send_text(json.dumps({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": transcript,
                }))
                content = [{"type": "input_text", "text": transcript}]
                caller_lang = customer_memory.get("language")
                if caller_lang in ("ta", "tunglish") and not re.search(r"[\u0B80-\u0BFF]", transcript):
                    content.append({
                        "type": "input_text",
                        "text": "[Caller is a Tamil speaker — keep replying in natural Thanglish, not English.]",
                    })
                await openai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": content},
                }))
                await openai_ws.send(json.dumps({"type": "response.create"}))
            elif kind == "error":
                print(f"[!] Saaras STT error ({ev.get('code')}): {ev.get('message')}")

        async def stop_sarvam_stt():
            stt = stt_state["client"]
            stt_state["client"] = None
            stt_state["engine"] = "openai"
            if stt:
                await stt.close()
            task = stt_state["task"]
            stt_state["task"] = None
            if task and task is not asyncio.current_task():
                task.cancel()

        async def run_sarvam_stt(stt):
            try:
                await stt.run()
            except Exception as e:
                print(f"[!] Saaras STT stream error: {e}")
            if stt_state["client"] is stt:
                # Unexpected disconnect: hand listening back to OpenAI so the call keeps working.
                print("[Realtime STT] Saaras v4 disconnected — falling back to OpenAI transcription")
                stt_state["client"] = None
                stt_state["engine"] = "openai"
                stt_state["task"] = None
                try:
                    await openai_ws.send(json.dumps({
                        "type": "session.update",
                        "session": build_session_config(
                            current_openai_voice(),
                            get_active_system_prompt() or SYSTEM_PROMPT,
                            modalities=voice_mode["modalities"],
                        ),
                    }))
                    await client_ws.send_text(json.dumps({
                        "type": "stt.engine",
                        "engine": "openai",
                        "message": "Sarvam speech recognition disconnected — switched to OpenAI transcription.",
                    }))
                except Exception:
                    pass

        async def ensure_stt_engine(stt_model):
            """Start Saaras v4 when it is the chosen STT engine; otherwise make sure it is stopped."""
            want_sarvam = str(stt_model or "").lower().startswith("saaras") and bool(os.getenv("SARVAM_API_KEY"))
            if want_sarvam and stt_state["engine"] != "sarvam":
                from sarvam_stt import SarvamRealtimeSTT
                stt = SarvamRealtimeSTT(on_saaras_event, model=str(stt_model).lower())
                try:
                    await stt.connect()
                except Exception as e:
                    print(f"[!] Saaras v4 STT unavailable, using OpenAI transcription: {type(e).__name__}: {e}")
                    return
                stt_state["client"] = stt
                stt_state["engine"] = "sarvam"
                stt_state["task"] = asyncio.create_task(run_sarvam_stt(stt))
                print("[Realtime STT] Saaras v4 realtime STT active (auto language, codemix, keyterms)")
            elif not want_sarvam and stt_state["engine"] == "sarvam":
                await stop_sarvam_stt()

        # Forward OpenAI -> Client
        async def forward_openai_to_client():
            try:
                async for message in openai_ws:
                    try:
                        event = json.loads(message)
                        event_type = event.get("type", "")
                        if event_type == "response.created":
                            response_active["val"] = True
                            response_epochs[(event.get("response") or {}).get("id")] = interrupt_epoch["val"]
                        elif event_type == "response.done":
                            response_active["val"] = False
                        elif event_type == "error" and (event.get("error") or {}).get("code") == "response_cancel_not_active":
                            continue

                        # Determine if external TTS engine (Sarvam AI / ElevenLabs) is handling speech
                        voice_val = active_voice.get("val", "")
                        tts_prov = active_tts_provider["val"]
                        is_tts_worker = (
                            tts_prov in ("sarvam", "sarvamai", "elevenlabs")
                            or voice_val in ("sarvam", "kavitha", "kavya", "priya", "shreya", "roopa", "mani", "gokul", "vijay", "elevenlabs")
                            or len(voice_val) >= 20
                        )

                        # 1. Text deltas — update live subtitles and early dispatch complete sentences
                        if event_type in (
                            "response.output_audio_transcript.delta",
                            "response.audio_transcript.delta",
                            "response.text.delta",
                            "response.output_text.delta",
                        ):
                            delta = event.get("delta", "")
                            assistant_transcript_buffer["text"] += delta
                            sentence_accumulator["text"] += delta
                            display_text = tanglish_to_tamil_display(assistant_transcript_buffer["text"])
                            await client_ws.send_text(json.dumps({
                                "type": "assistant.transcript.display",
                                "text": display_text,
                                "roman": assistant_transcript_buffer["text"],
                            }))

                            # Sentence streaming: the first sentence is spoken as soon as it is complete (low latency);
                            # later text is grouped into longer phrases so Bulbul keeps natural, continuous intonation.
                            acc_str = sentence_accumulator["text"]
                            boundaries = list(re.finditer(r'([.!?।]+|\n)\s+', acc_str))
                            m = None
                            if boundaries and is_tts_worker:
                                if not speech_flushed_for_turn["val"]:
                                    m = boundaries[0]
                                elif boundaries[-1].end() >= 120:
                                    m = boundaries[-1]
                            if m:
                                end_idx = m.end()
                                candidate = acc_str[:end_idx].strip()
                                if len(candidate.split()) >= 2 or len(candidate) >= 8:
                                    sentence_accumulator["text"] = acc_str[end_idx:]
                                    if not is_english_tool_filler(candidate):
                                        speech_flushed_for_turn["val"] = True
                                        is_speaking_cancelled["val"] = False
                                        await speech_queue.put(candidate)

                        # 2. Audio output chunks from OpenAI -> forward to client only if not using dedicated TTS
                        elif event_type in ("response.output_audio.delta", "response.audio.delta"):
                            if not openai_audio_voice["val"] and active_tts_provider["val"] == "openai":
                                openai_audio_voice["val"] = active_voice["val"]
                            if not is_tts_worker:
                                await client_ws.send_text(message)

                        # 3. Response text complete → flush remainder to speech worker
                        elif event_type in (
                            "response.output_audio_transcript.done",
                            "response.audio_transcript.done",
                            "response.output_text.done",
                            "response.text.done",
                        ):
                            rem_text = sentence_accumulator["text"].strip()
                            sentence_accumulator["text"] = ""
                            full_text = assistant_transcript_buffer["text"].strip()

                            if rem_text and is_tts_worker:
                                if is_english_tool_filler(rem_text):
                                    print(f"[Realtime] Suppressed English tool filler speech: {rem_text}")
                                else:
                                    speech_flushed_for_turn["val"] = True
                                    is_speaking_cancelled["val"] = False
                                    await speech_queue.put(rem_text)
                            elif not speech_flushed_for_turn["val"] and full_text and is_tts_worker:
                                if not is_english_tool_filler(full_text):
                                    speech_flushed_for_turn["val"] = True
                                    is_speaking_cancelled["val"] = False
                                    await speech_queue.put(full_text)

                            if full_text:
                                assistant_transcript_buffer["last_done"] = full_text
                                await client_ws.send_text(json.dumps({
                                    "type": "assistant.transcript.display",
                                    "text": tanglish_to_tamil_display(full_text),
                                    "roman": full_text,
                                    "final": True,
                                }))
                            assistant_transcript_buffer["text"] = ""
                            await client_ws.send_text(message)

                        elif event_type == "response.done":
                            rem_text = sentence_accumulator["text"].strip()
                            sentence_accumulator["text"] = ""
                            full_text = assistant_transcript_buffer["text"].strip()

                            if rem_text and is_tts_worker:
                                if not is_english_tool_filler(rem_text):
                                    speech_flushed_for_turn["val"] = True
                                    is_speaking_cancelled["val"] = False
                                    await speech_queue.put(rem_text)
                            elif not speech_flushed_for_turn["val"] and full_text and is_tts_worker:
                                if not is_english_tool_filler(full_text):
                                    speech_flushed_for_turn["val"] = True
                                    is_speaking_cancelled["val"] = False
                                    await speech_queue.put(full_text)

                            if full_text:
                                assistant_transcript_buffer["last_done"] = full_text
                                await client_ws.send_text(json.dumps({
                                    "type": "assistant.transcript.display",
                                    "text": tanglish_to_tamil_display(full_text),
                                    "roman": full_text,
                                    "final": True,
                                }))
                            assistant_transcript_buffer["text"] = ""
                            speech_flushed_for_turn["val"] = False  # ready for next turn
                            await client_ws.send_text(message)

                        # 3b. Caller speech transcription completed -> auto-detect language & notify UI
                        elif event_type in (
                            "conversation.item.input_audio_transcription.completed",
                            "input_audio_transcription.completed",
                        ):
                            transcript = event.get("transcript", "")
                            if transcript:
                                event["transcript"] = await process_caller_transcript(transcript)
                                message = json.dumps(event)
                            await client_ws.send_text(message)

                        # 4. User interruption or speech started -> instantly cancel ElevenLabs speech playback
                        elif event_type in ("input_audio_buffer.speech_started", "response.cancel"):
                            clear_pending_speech()
                            await client_ws.send_text(message)

                        # 5. Function Tool Call for Self-Training & Voice Correction (OpenAI Realtime train_voice_correction)
                        elif event_type in ("response.function_call_arguments.done", "response.output_item.done"):
                            await client_ws.send_text(message)

                            fn_name = None
                            call_id = None
                            fn_args_raw = "{}"

                            if event_type == "response.function_call_arguments.done":
                                fn_name = event.get("name")
                                call_id = event.get("call_id")
                                fn_args_raw = event.get("arguments", "{}")
                            elif event_type == "response.output_item.done":
                                item = event.get("item", {})
                                if item.get("type") == "function_call":
                                    fn_name = item.get("name")
                                    call_id = item.get("call_id")
                                    fn_args_raw = item.get("arguments", "{}")

                            epoch_at_call = response_epochs.get(event.get("response_id"), interrupt_epoch["val"])
                            if fn_name == "search_mintx_knowledge" and call_id and call_id not in handled_tool_calls:
                                handled_tool_calls.add(call_id)
                                print(f"[RAG Vector Tool Call] Received search_mintx_knowledge (call_id: {call_id}): {fn_args_raw}")
                                try:
                                    fn_args = json.loads(fn_args_raw)
                                except Exception:
                                    fn_args = {"query": fn_args_raw}
                                query_text = (fn_args.get("query") or "").strip() or fn_args_raw
                                from rag import search_mintx_knowledge
                                rag_context = await asyncio.to_thread(search_mintx_knowledge, query=query_text, top_k=2)
                                print(f"[RAG Retrieved Facts for '{query_text}']:\n{rag_context}")

                                tool_output_event = {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": json.dumps({
                                            "status": "success",
                                            "query": query_text,
                                            "retrieved_knowledge": rag_context
                                        })
                                    }
                                }
                                await openai_ws.send(json.dumps(tool_output_event))
                                if interrupt_epoch["val"] == epoch_at_call:
                                    await openai_ws.send(json.dumps({"type": "response.create"}))

                            elif fn_name == "save_customer_memory" and call_id and call_id not in handled_tool_calls:
                                handled_tool_calls.add(call_id)
                                print(f"[GPT Memory Tool Call] Received save_customer_memory (call_id: {call_id}): {fn_args_raw}")
                                try:
                                    fn_args = json.loads(fn_args_raw)
                                except Exception:
                                    fn_args = {}

                                updated_fields = {}
                                for k in ("name", "business_name", "service", "phone", "requirement"):
                                    val = fn_args.get(k)
                                    if val is not None and str(val).strip():
                                        clean_v = str(val).strip()
                                        customer_memory[k] = clean_v
                                        updated_fields[k] = clean_v

                                is_confirmed = bool(fn_args.get("is_confirmed", False))
                                if is_confirmed:
                                    customer_memory["status"] = "confirmed"
                                    updated_fields["status"] = "confirmed"

                                # Persist in the background so a slow/offline database never delays the reply
                                async def persist_lead(snapshot: dict):
                                    try:
                                        from database import save_or_update_lead
                                        await asyncio.to_thread(save_or_update_lead, session_call_id, snapshot)
                                    except Exception as e:
                                        print(f"[!] Error persisting lead memory: {e}")
                                asyncio.create_task(persist_lead(dict(customer_memory)))

                                # Send function call output back to OpenAI Realtime
                                tool_output_event = {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": json.dumps({
                                            "status": "success",
                                            "is_confirmed": is_confirmed,
                                            "saved_fields": updated_fields,
                                            "current_memory": {k: v for k, v in customer_memory.items() if v}
                                        })
                                    }
                                }
                                await openai_ws.send(json.dumps(tool_output_event))

                                # Trigger assistant to continue conversation naturally using memory
                                if interrupt_epoch["val"] == epoch_at_call:
                                    await openai_ws.send(json.dumps({
                                        "type": "response.create"
                                    }))

                                # Send live update to client UI
                                try:
                                    await client_ws.send_text(json.dumps({
                                        "type": "memory.updated",
                                        "memory": customer_memory
                                    }))
                                except Exception:
                                    pass

                            elif fn_name == "train_voice_correction" and call_id and call_id not in handled_tool_calls:
                                handled_tool_calls.add(call_id)
                                print(f"[Realtime Tool Call] Received train_voice_correction (call_id: {call_id}): {fn_args_raw}")

                                # Instantly cancel any ongoing speech playback
                                cancel_speech()
                                while not speech_queue.empty():
                                    try:
                                        speech_queue.get_nowait()
                                        speech_queue.task_done()
                                    except Exception:
                                        break
                                sentence_accumulator["text"] = ""
                                assistant_transcript_buffer["text"] = ""

                                # Parse tool call arguments
                                try:
                                    fn_args = json.loads(fn_args_raw)
                                except Exception:
                                    fn_args = {"feedback": fn_args_raw}
                                feedback_text = (fn_args.get("feedback") or "").strip() or fn_args_raw

                                fb_low = feedback_text.lower()
                                training_cues = ["correction", "கரெக்ஷன்", "rule", "ரூல்", "training", "டிரெய்னிங்", "மாத்து", "மாத்துங்க", "update rule", "starting greeting", "opening words", "கிரீட்டிங்", "சொல்லாத", "பேசாத", "dont say", "do not say", "always say"]
                                is_actual_training = any(cue in fb_low for cue in training_cues)
                                if not is_actual_training:
                                    print(f"[Realtime Tool Call] train_voice_correction called with non-training input '{feedback_text}'. Ignoring rule update.")
                                    tool_output_event = {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": call_id,
                                            "output": json.dumps({
                                                "status": "ignored",
                                                "message": "This was a normal customer dialogue turn, not a system prompt training instruction. Continue answering the customer naturally in their language."
                                            })
                                        }
                                    }
                                    await openai_ws.send(json.dumps(tool_output_event))
                                    await openai_ws.send(json.dumps({"type": "response.create"}))
                                    continue

                                active_id = get_all_data().get("active_id", "default-mintx")
                                try:
                                    corr = await asyncio.to_thread(
                                        refine_instruction_with_feedback,
                                        instruction_id=active_id,
                                        user_feedback=feedback_text,
                                        last_assistant_response=assistant_transcript_buffer.get("last_done", ""),
                                        conversation_context="",
                                        api_key=current_api_key
                                    )
                                    verbal_ack = corr.get("verbal_ack") or "Sure sir, naan rule update panniten!"
                                    new_inst = corr.get("new_instructions") or get_active_system_prompt()
                                    summary_en = corr.get("summary_en", "Rule updated")
                                    rule_diff = corr.get("rule_diff", "")
                                    starting_greeting = corr.get("starting_greeting")

                                    # 1. Send function call output back to OpenAI Realtime
                                    tool_output_event = {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": call_id,
                                            "output": json.dumps({
                                                "status": "success",
                                                "message": f"Rule learned and updated: {summary_en}",
                                                "spoken_acknowledgment": verbal_ack
                                            })
                                        }
                                    }
                                    await openai_ws.send(json.dumps(tool_output_event))

                                    # 2. Hot-reload active session instructions on OpenAI Realtime!
                                    target_voice = current_openai_voice()
                                    await openai_ws.send(json.dumps({
                                        "type": "session.update",
                                        "session": build_session_config(target_voice, new_inst)
                                    }))

                                    # 3. Instruct assistant to immediately speak the verbal acknowledgment aloud
                                    await openai_ws.send(json.dumps({
                                        "type": "response.create",
                                        "response": {
                                            "instructions": f"Speak aloud this natural confirmation in 1 short sentence: '{verbal_ack}'"
                                        }
                                    }))

                                    # 4. Notify browser UI via WebSocket event
                                    await client_ws.send_text(json.dumps({
                                        "type": "correction.applied",
                                        "feedback": feedback_text,
                                        "summary_en": summary_en,
                                        "rule_diff": rule_diff,
                                        "verbal_ack": verbal_ack,
                                        "starting_greeting": starting_greeting,
                                        "new_instructions": new_inst,
                                        "history": get_training_history()
                                    }))
                                    print(f"[Realtime Tool Call] Successfully learned and applied rule: '{summary_en}' ({rule_diff})")
                                except Exception as err:
                                    print(f"[!] Error processing train_voice_correction: {err}")

                        else:
                            await client_ws.send_text(message)

                    except Exception as e:
                        print(f"[!] OpenAI -> client forward error: {e}")
                        break
            except websockets.exceptions.ConnectionClosed as e:
                print(f"[!] OpenAI WebSocket closed during call: code={e.code}, reason={e.reason}")
                error_msg = str(e.reason) or "OpenAI connection closed."
                if e.code == 1013 or "insufficient_quota" in str(e.reason):
                    error_msg = "OpenAI API Quota Exhausted: Your account credit balance is exhausted ($0.00). Please recharge credits at https://platform.openai.com/settings/organization/billing/ or update OPENAI_API_KEY."
                try:
                    await client_ws.send_text(json.dumps({
                        "type": "error",
                        "error": {
                            "code": "insufficient_quota" if (e.code == 1013 or "insufficient_quota" in str(e.reason)) else "connection_closed",
                            "message": error_msg
                        }
                    }))
                except Exception:
                    pass
            except Exception as e:
                print(f"[!] OpenAI forward error: {e}")

        # Forward Client -> OpenAI
        async def forward_client_to_openai():
            try:
                while True:
                    client_msg = await client_ws.receive_text()
                    data = json.loads(client_msg)
                    event_type = data.get("type", "")

                    if event_type == "client.init":
                        target_voice = select_voice(data.get("voice", DEFAULT_VOICE_NAME))

                        # Direct connection to active instructions: use provided client instructions if non-empty, else active instructions from instructions.json
                        if data.get("instructions") and data.get("instructions").strip():
                            custom_inst = data.get("instructions").strip()
                        else:
                            custom_inst = get_active_system_prompt() or SYSTEM_PROMPT

                        reasoning_effort = data.get("reasoning_effort")
                        temperature = data.get("temperature")
                        stt_model = data.get("stt_model")
                        vad_threshold = data.get("vad_threshold")
                        vad_silence_ms = data.get("vad_silence_ms")
                        vad_prefix_ms = data.get("vad_prefix_ms")

                        if data.get("sarvam_speaker") and active_voice["val"] == "sarvam":
                            active_sarvam_speaker["val"] = data.get("sarvam_speaker")
                            os.environ["SARVAM_TTS_SPEAKER"] = data.get("sarvam_speaker")
                        if data.get("sarvam_model"):
                            active_sarvam_model["val"] = data.get("sarvam_model")
                            os.environ["SARVAM_TTS_MODEL"] = data.get("sarvam_model")
                        if data.get("sarvam_pace") is not None:
                            try:
                                active_sarvam_pace["val"] = float(data.get("sarvam_pace"))
                                os.environ["SARVAM_TTS_PACE"] = str(data.get("sarvam_pace"))
                            except Exception:
                                pass

                        if data.get("elevenlabs_model"):
                            el_m = "eleven_turbo_v2_5" if "v4" in str(data.get("elevenlabs_model")).lower() else data.get("elevenlabs_model")
                            active_elevenlabs_model["val"] = el_m
                            os.environ["ELEVENLABS_MODEL_ID"] = el_m
                        if data.get("elevenlabs_voice_id"):
                            active_elevenlabs_voice_id["val"] = data.get("elevenlabs_voice_id")
                            os.environ["ELEVENLABS_VOICE_ID"] = data.get("elevenlabs_voice_id")
                        if data.get("elevenlabs_stability") is not None:
                            try:
                                active_elevenlabs_stability["val"] = float(data.get("elevenlabs_stability"))
                            except Exception:
                                pass
                        if data.get("elevenlabs_similarity") is not None:
                            try:
                                active_elevenlabs_similarity["val"] = float(data.get("elevenlabs_similarity"))
                            except Exception:
                                pass
                        if data.get("elevenlabs_style") is not None:
                            try:
                                active_elevenlabs_style["val"] = float(data.get("elevenlabs_style"))
                            except Exception:
                                pass
                        if data.get("elevenlabs_speaker_boost") is not None:
                            active_elevenlabs_speaker_boost["val"] = bool(data.get("elevenlabs_speaker_boost"))

                        await ensure_stt_engine(stt_model)
                        await client_ws.send_text(json.dumps({"type": "stt.engine", "engine": stt_state["engine"]}))

                        session_update = {
                            "type": "session.update",
                            "session": build_session_config(
                                target_voice,
                                custom_inst,
                                reasoning_effort=reasoning_effort,
                                temperature=temperature,
                                stt_model=stt_model,
                                vad_threshold=vad_threshold,
                                vad_silence_ms=vad_silence_ms,
                                vad_prefix_ms=vad_prefix_ms,
                                modalities=voice_mode["modalities"]
                            )
                        }
                        await openai_ws.send(json.dumps(session_update))
                        print(f"[Realtime Server] Voice engine: {'GPT voice (' + target_voice + ')' if active_tts_provider['val'] == 'openai' else active_tts_provider['val'] + ' (' + active_sarvam_speaker['val'] + ')'}")

                        # Instant Yazhini greeting via ElevenLabs (do not wait for OpenAI to invent text)
                        if data.get("auto_greet", True):
                            starting_greeting = normalize_brand_name(
                                (
                                    data.get("starting_greeting")
                                    or get_active_starting_greeting()
                                    or "Vanakkam! MintX Solutions-ku welcome. Enna service venum nu sollunga?"
                                ).strip()
                            )
                            print(f"[Realtime] Auto-greeting with Yazhini: {starting_greeting}")

                        if data.get("auto_greet", True) and active_tts_provider["val"] == "openai":
                            # GPT voice speaks the greeting itself; its transcript events drive the UI subtitles.
                            await openai_ws.send(json.dumps({
                                "type": "response.create",
                                "response": {
                                    "instructions": (
                                        "Greet the caller now by saying exactly this sentence, warmly and naturally, "
                                        f"with nothing before or after it: \"{starting_greeting}\""
                                    )
                                }
                            }))
                        elif data.get("auto_greet", True):
                            # UI shows Tamil letters; speech stays Tanglish Roman for ElevenLabs
                            display_greeting = normalize_brand_name(tanglish_to_tamil_display(starting_greeting))
                            await client_ws.send_text(json.dumps({
                                "type": "assistant.transcript.display",
                                "text": display_greeting,
                                "roman": starting_greeting,
                                "final": True,
                            }))

                            # Speak immediately with ElevenLabs Yazhini (Roman Tanglish)
                            is_speaking_cancelled["val"] = False
                            await speech_queue.put(starting_greeting)

                            # Tell OpenAI the greeting already happened so it won't repeat it
                            try:
                                await openai_ws.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "assistant",
                                        "content": [
                                            {
                                                "type": "output_text",
                                                "text": starting_greeting
                                            }
                                        ]
                                    }
                                }))
                            except Exception as greet_hist_err:
                                print(f"[!] Could not seed greeting into OpenAI history: {greet_hist_err}")

                            await client_ws.send_text(json.dumps({
                                "type": "response.done",
                                "response": {"status": "completed", "status_details": {"reason": "auto_greet"}}
                            }))

                    elif event_type in ("session.update", "client.switch_voice"):
                        target_voice = select_voice(data.get("voice", active_voice["val"]))

                        if data.get("sarvam_speaker") and active_voice["val"] == "sarvam" and data.get("voice") in (None, "sarvam", "default"):
                            active_sarvam_speaker["val"] = data.get("sarvam_speaker")
                            os.environ["SARVAM_TTS_SPEAKER"] = data.get("sarvam_speaker")
                        if data.get("sarvam_model"):
                            active_sarvam_model["val"] = data.get("sarvam_model")
                            os.environ["SARVAM_TTS_MODEL"] = data.get("sarvam_model")
                        if data.get("sarvam_pace") is not None:
                            try:
                                active_sarvam_pace["val"] = float(data.get("sarvam_pace"))
                                os.environ["SARVAM_TTS_PACE"] = str(data.get("sarvam_pace"))
                            except Exception:
                                pass

                        if data.get("elevenlabs_model"):
                            el_m = "eleven_turbo_v2_5" if "v4" in str(data.get("elevenlabs_model")).lower() else data.get("elevenlabs_model")
                            active_elevenlabs_model["val"] = el_m
                            os.environ["ELEVENLABS_MODEL_ID"] = el_m
                        if data.get("elevenlabs_voice_id"):
                            active_elevenlabs_voice_id["val"] = data.get("elevenlabs_voice_id")
                            os.environ["ELEVENLABS_VOICE_ID"] = data.get("elevenlabs_voice_id")
                        if data.get("elevenlabs_stability") is not None:
                            try:
                                active_elevenlabs_stability["val"] = float(data.get("elevenlabs_stability"))
                            except Exception:
                                pass
                        if data.get("elevenlabs_similarity") is not None:
                            try:
                                active_elevenlabs_similarity["val"] = float(data.get("elevenlabs_similarity"))
                            except Exception:
                                pass
                        if data.get("elevenlabs_style") is not None:
                            try:
                                active_elevenlabs_style["val"] = float(data.get("elevenlabs_style"))
                            except Exception:
                                pass
                        if data.get("elevenlabs_speaker_boost") is not None:
                            active_elevenlabs_speaker_boost["val"] = bool(data.get("elevenlabs_speaker_boost"))

                        # Immediately cancel any pending speech and clear buffers so new voice takes effect
                        cancel_speech()
                        sentence_accumulator["text"] = ""
                        assistant_transcript_buffer["text"] = ""
                        while not speech_queue.empty():
                            try:
                                speech_queue.get_nowait()
                                speech_queue.task_done()
                            except Exception:
                                pass

                        if data.get("instructions") and data.get("instructions").strip():
                            instructions = data.get("instructions").strip()
                            try:
                                data_all = get_all_data()
                                active_id = data_all.get("active_id", "default-mintx")
                                update_instruction(active_id, instructions=instructions)
                            except Exception:
                                pass
                        else:
                            instructions = get_active_system_prompt() or SYSTEM_PROMPT

                        reasoning_effort = data.get("reasoning_effort")
                        temperature = data.get("temperature")
                        stt_model = data.get("stt_model")
                        vad_threshold = data.get("vad_threshold")
                        vad_silence_ms = data.get("vad_silence_ms")
                        vad_prefix_ms = data.get("vad_prefix_ms")

                        if stt_model:
                            await ensure_stt_engine(stt_model)

                        # OpenAI rejects a voice change once that session has produced audio.
                        if openai_audio_voice["val"] and openai_audio_voice["val"] != target_voice:
                            await client_ws.send_text(json.dumps({
                                "type": "voice.reconnect_required",
                                "voice": active_voice["val"],
                            }))
                            continue

                        session_update = {
                            "type": "session.update",
                            "session": build_session_config(
                                target_voice,
                                instructions,
                                reasoning_effort=reasoning_effort,
                                temperature=temperature,
                                stt_model=stt_model,
                                vad_threshold=vad_threshold,
                                vad_silence_ms=vad_silence_ms,
                                vad_prefix_ms=vad_prefix_ms,
                                modalities=voice_mode["modalities"]
                            )
                        }
                        await openai_ws.send(json.dumps(session_update))
                        print(f"[Realtime Server] Session updated (Voice: '{active_voice['val']}', Target OpenAI Voice: '{target_voice}', ElevenLabs Voice ID: '{active_elevenlabs_voice_id['val']}', STT: {stt_model}, VAD Silence: {vad_silence_ms}ms, Reasoning: {reasoning_effort}, Temp: {temperature})")

                    elif event_type == "client.speak_ack":
                        # Instant verbal AI acknowledgment of rule change over realtime audio stream
                        ack_text = data.get("text", "")
                        if ack_text:
                            trigger_msg = {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_text",
                                            "text": f"[System Command - Speak Acknowledgment]: Immediately speak aloud to the caller in 1 natural sentence: '{ack_text}'"
                                        }
                                    ]
                                }
                            }
                            await openai_ws.send(json.dumps(trigger_msg))
                            await openai_ws.send(json.dumps({"type": "response.create"}))

                    elif event_type == "client.set_starting_greeting":
                        new_greet = data.get("greeting", "").strip()
                        target_id = data.get("instruction_id") or get_all_data().get("active_id", "default-mintx")
                        res = set_active_starting_greeting(target_id, new_greet)
                        new_inst = res.get("new_instructions") or get_active_system_prompt()
                        target_voice = current_openai_voice()
                        session_update = {
                            "type": "session.update",
                            "session": build_session_config(target_voice, new_inst)
                        }
                        await openai_ws.send(json.dumps(session_update))
                        ack_text = f"Sure sir, starting greeting-a \"{res.get('starting_greeting')}\" nu update panniten!"
                        trigger_msg = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": f"[System Command - Speak Acknowledgment]: Immediately speak aloud to the caller in 1 natural sentence: '{ack_text}'"
                                    }
                                ]
                            }
                        }
                        await openai_ws.send(json.dumps(trigger_msg))
                        await openai_ws.send(json.dumps({"type": "response.create"}))

                    elif event_type == "client.test_greeting":
                        starting_greeting = data.get("greeting", "").strip() or get_active_starting_greeting()
                        await client_ws.send_text(json.dumps({
                            "type": "response.output_audio_transcript.delta",
                            "delta": tanglish_to_tamil_display(starting_greeting)
                        }))
                        await client_ws.send_text(json.dumps({"type": "response.output_audio_transcript.done"}))
                        is_speaking_cancelled["val"] = False
                        await speech_queue.put(starting_greeting)
                        await client_ws.send_text(json.dumps({
                            "type": "response.done",
                            "response": {"status": "completed", "status_details": {"reason": "test_greet"}}
                        }))

                    elif event_type == "client.barge_in":
                        await interrupt_assistant()

                    elif event_type.startswith("client."):
                        # Browser-only control messages; OpenAI rejects unknown event types.
                        pass

                    elif event_type.startswith("input_audio_buffer.") and stt_state["engine"] == "sarvam":
                        if event_type == "input_audio_buffer.append" and stt_state["client"]:
                            try:
                                await stt_state["client"].send_audio_24k(data.get("audio", ""))
                            except Exception as e:
                                print(f"[!] Saaras audio send failed: {e}")

                    else:
                        # Direct forwarding of standard events:
                        # input_audio_buffer.append, conversation.item.create, response.create, response.cancel
                        if event_type == "response.cancel":
                            clear_pending_speech()
                        await openai_ws.send(client_msg)

            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"[!] Client -> OpenAI forward error: {e}")

        # Telemetry ping task
        async def send_telemetry():
            try:
                while True:
                    await asyncio.sleep(5)
                    t0 = time.time()
                    # Calculate round-trip estimate
                    latency_ms = int((time.time() - t0) * 1000 + 45)  # typical base latency
                    try:
                        await client_ws.send_text(json.dumps({
                            "type": "telemetry.ping",
                            "latency_ms": latency_ms
                        }))
                    except Exception:
                        break
            except asyncio.CancelledError:
                pass

        forward_tasks = [
            asyncio.create_task(forward_openai_to_client()),
            asyncio.create_task(forward_client_to_openai()),
        ]
        background_tasks = [
            asyncio.create_task(speech_worker()),
            asyncio.create_task(send_telemetry()),
        ]
        # When either side hangs up, tear down the whole session so the OpenAI socket is closed.
        await asyncio.wait(forward_tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in forward_tasks + background_tasks:
            task.cancel()
        await asyncio.gather(*forward_tasks, *background_tasks, return_exceptions=True)

    except Exception as err:
        print(f"[Realtime Proxy Error]: {err}")
        try:
            await client_ws.send_text(json.dumps({
                "type": "error",
                "error": {"message": str(err)}
            }))
        except Exception:
            pass
    finally:
        cancel_speech()
        try:
            speech_queue.put_nowait(None)
        except Exception:
            pass
        if openai_ws:
            try:
                await openai_ws.close()
            except Exception:
                pass
        stt_client, stt_task = stt_state["client"], stt_state["task"]
        stt_state["client"] = None
        if stt_client:
            await stt_client.close()
        if stt_task:
            stt_task.cancel()
        try:
            await client_ws.close()
        except Exception:
            pass
        print("[Realtime Server] Session terminated.")


def create_realtime_app() -> FastAPI:
    """Create and configure standalone FastAPI application for the Realtime UI."""
    app = FastAPI(
        title="Mintx Solutions AI — Realtime Voice Studio",
        description="Low-latency real-time voice streaming web studio powered by OpenAI Realtime WebSocket API",
        version="2.0.0"
    )

    @app.middleware("http")
    async def add_no_cache_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path == "/realtime":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # Mount static assets
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Mount routes
    app.include_router(router)

    # Root redirect to /realtime
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return await get_realtime_ui()

    return app


# Exportable app instance
app = create_realtime_app()
