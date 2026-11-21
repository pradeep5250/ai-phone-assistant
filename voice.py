"""
voice.py — Voice Processing

Handles speech-to-text (STT), text-to-speech (TTS), microphone recording,
and speaker playback.
- STT: OpenAI Whisper / Sarvam (supports English and Tamil)
- TTS: ElevenLabs only (Yazhini / configured voice_id) when TTS_PROVIDER=elevenlabs
"""
import os
import io
import time
import uuid
import queue
import threading
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

import asyncio
import sounddevice as sd
import soundfile as sf

# Load environment from .env and .env.local
load_dotenv()
load_dotenv(dotenv_path=".env.local", override=True)

from tanglish_script import normalize_brand_name, tanglish_for_tts

# ──     Configuration ───────────────────────────────────────────────────

AUDIO_DIR = Path(__file__).parent / "audio_files"
AUDIO_DIR.mkdir(exist_ok=True)

# OpenAI client (Brain & STT)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Sarvam AI TTS Configuration ---
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "sk_k3r7kg86_Q9EDFRrqWsBDXuIZ2i9MSVyx")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saaras:v3")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "kavitha")
SARVAM_TTS_LANGUAGE_CODE = os.getenv("SARVAM_TTS_LANGUAGE_CODE", "ta-IN")
SARVAM_TTS_PACE = float(os.getenv("SARVAM_TTS_PACE", "1.25"))

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "sarvam").lower()

SARVAM_SPEAKERS = {
    "kavitha": "kavitha",
    "yazhini": "kavitha",
    "kavya": "kavya",
    "priya": "priya",
    "shreya": "shreya",
    "roopa": "roopa",
    "shruti": "shruti",
    "suhani": "suhani",
    "pooja": "pooja",
    "neha": "neha",
    "mani": "mani",
    "gokul": "gokul",
    "vijay": "vijay",
    "aditya": "aditya",
}

# ElevenLabs configuration (optional fallback)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
ELEVENLABS_OPTIMIZE_LATENCY = int(os.getenv("ELEVENLABS_OPTIMIZE_LATENCY", "4"))

DEFAULT_ELEVENLABS_STABILITY = float(os.getenv("ELEVENLABS_STABILITY", "0.42"))
DEFAULT_ELEVENLABS_SIMILARITY = float(os.getenv("ELEVENLABS_SIMILARITY", "0.82"))
DEFAULT_ELEVENLABS_STYLE = float(os.getenv("ELEVENLABS_STYLE", "0.35"))
DEFAULT_ELEVENLABS_SPEED = float(os.getenv("ELEVENLABS_SPEED", "1.0"))

# --- STT Configuration (OpenAI GPT Live Transcribe / Sarvam) ---
STT_PROVIDER = os.getenv("STT_PROVIDER", "openai").lower()

_elevenlabs_client = None
_elevenlabs_library_restricted = False


def get_elevenlabs_client():
    """Get or lazily initialize the ElevenLabs client."""
    global _elevenlabs_client
    if _elevenlabs_client is None:
        key = os.getenv("ELEVENLABS_API_KEY")
        if key:
            try:
                from elevenlabs.client import ElevenLabs
                _elevenlabs_client = ElevenLabs(api_key=key)
            except Exception as e:
                print(f"[!] Could not initialize ElevenLabs client: {e}")
    return _elevenlabs_client


# Default voice is Sarvam Kavitha (Yazhini persona)
DEFAULT_VOICE = os.getenv("SARVAM_TTS_SPEAKER", "kavitha")
DEFAULT_TAMIL_VOICE = os.getenv("TAMIL_VOICE", "ta-IN-PallaviNeural")

TAMIL_VOICES = {
    "tamil-female": "ta-IN-PallaviNeural",
    "tamil-male": "ta-IN-ValluvarNeural",
    "pallavi": "ta-IN-PallaviNeural",
    "valluvar": "ta-IN-ValluvarNeural",
    "saranya": "ta-LK-SaranyaNeural",
    "kumar": "ta-LK-KumarNeural",
    "venba": "ta-SG-VenbaNeural",
    "anbu": "ta-SG-AnbuNeural",
}

SUPPORTED_VOICES = [
    "kavitha", "kavya", "priya", "shreya", "roopa", "mani", "gokul", "vijay",
    "marin", "alloy", "echo", "fable", "onyx", "nova", "shimmer", "verse", "ballad", "ash", "sage", "coral",
    "pallavi", "valluvar", "saranya", "kumar", "tamil-female", "tamil-male"
]
STANDARD_TTS_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "sage", "coral"}

# OpenAI Realtime Session Configuration (Strictly gpt-realtime-2.1)
REALTIME_SESSION_CONFIG = {
    "session": {
        "type": "realtime",
        "model": os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1"),
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": 24000
                },
                "transcription": {
                    "model": os.getenv("OPENAI_TRANSCRIPTION_MODEL", os.getenv("OPENAI_REALTIME_TRANSCRIPTION_MODEL", "gpt-live-transcribe"))
                },
                "noise_reduction": {
                    "type": "far_field"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 600
                }
            },
            "output": {
                "format": {
                    "type": "audio/pcm",
                    "rate": 24000
                },
                "voice": "marin"
            }
        },
        "output_modalities": [
            "audio"
        ],
        "tools": [],
        "max_output_tokens": 2048,
        "reasoning": {
            "effort": "medium"
        }
    }
}


def get_realtime_session_config(
    model: str = None,
    voice: str = "marin",
    reasoning_effort: str = "medium",
    modalities: list = None,
    stt_model: str = None,
    vad_threshold: float = 0.5,
    silence_duration_ms: int = 500,
    prefix_padding_ms: int = 300,
    temperature: float = None
) -> dict:
    """
    Get customized OpenAI Realtime Session Configuration strictly locked to gpt-realtime-2.1 with full STT and TTS parameters.
    """
    active_model = model or os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
    active_stt = stt_model if (stt_model and stt_model != "none") else os.getenv("OPENAI_TRANSCRIPTION_MODEL", os.getenv("OPENAI_REALTIME_TRANSCRIPTION_MODEL", "gpt-live-transcribe"))

    config = {
        "session": {
            "type": "realtime",
            "model": active_model,
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    },
                    "transcription": {
                        "model": active_stt
                    },
                    "noise_reduction": {
                        "type": "far_field"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": vad_threshold if vad_threshold is not None else 0.5,
                        "prefix_padding_ms": prefix_padding_ms if prefix_padding_ms is not None else 300,
                        "silence_duration_ms": silence_duration_ms if silence_duration_ms is not None else 500
                    }
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    },
                    "voice": voice or "marin"
                }
            },
            "output_modalities": modalities or ["audio"],
            "tools": [],
            "max_output_tokens": 2048,
            "reasoning": {
                "effort": reasoning_effort or "medium"
            }
        }
    }
    # GA Realtime API no longer accepts session.temperature
    return config

# Pygame mixer for audio playback
_pygame_initialized = False


def _init_audio_playback():
    """Lazy initialize pygame mixer for audio output."""
    global _pygame_initialized
    if not _pygame_initialized:
        import pygame
        pygame.mixer.init()
        _pygame_initialized = True


# ── Microphone Recording ────────────────────────────────────────────

def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> bytes:
    """
    Record audio from the microphone for a fixed duration.

    Args:
        duration: Seconds to record.
        sample_rate: Audio sample rate (16000 Hz is standard for Whisper).

    Returns:
        WAV audio bytes.
    """
    print(f"🎙️  Recording for {duration:.1f} seconds... Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    print("✅ Recording finished.")

    # Convert to in-memory WAV bytes
    wav_io = io.BytesIO()
    sf.write(wav_io, audio_data, sample_rate, format="WAV")
    wav_io.seek(0)
    return wav_io.read()


def record_audio_press_enter(sample_rate: int = 16000) -> bytes:
    """
    Start recording audio when called, and stop when user presses Enter.

    Returns:
        WAV audio bytes.
    """
    audio_chunks = []
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            pass
        audio_chunks.append(indata.copy())

    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", callback=callback)
    
    print("\n🎙️  [RECORDING STARTED] Speak into your microphone...")
    print("👉 Press [ENTER] when you are finished speaking...")
    with stream:
        input()  # Wait for user to press Enter
        stop_event.set()

    print("🛑 Recording stopped. Processing audio...")

    if not audio_chunks:
        return b""

    import numpy as np
    full_audio = np.concatenate(audio_chunks, axis=0)

    wav_io = io.BytesIO()
    sf.write(wav_io, full_audio, sample_rate, format="WAV")
    wav_io.seek(0)
    return wav_io.read()


# ── Speech-to-Text (STT) & Translation ─────────────────────────────

def sarvam_speech_to_text(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    model: str = None,
    language_code: str = "unknown",
) -> str:
    """
    Convert speech audio to text using Sarvam AI (Saaras:v3 / Saarika).
    Specialized for Indian languages (Tamil, English, Tanglish, Hindi, etc.).
    """
    key = os.getenv("SARVAM_API_KEY", "").strip() or SARVAM_API_KEY
    if not key:
        raise ValueError("Sarvam AI API key not configured")

    target_model = model or os.getenv("SARVAM_MODEL", SARVAM_MODEL or "saaras:v3")
    target_lang = language_code or "unknown"

    import httpx

    ext = os.path.splitext(filename)[1].lower() if filename else ".wav"
    mime_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".m4a": "audio/m4a",
        ".webm": "audio/webm",
        ".flac": "audio/flac",
    }
    content_type = mime_types.get(ext, "audio/wav")

    headers = {
        "api-subscription-key": key,
    }
    files = {
        "file": (filename or "audio.wav", audio_bytes, content_type)
    }
    data = {
        "model": target_model,
        "language_code": target_lang,
    }

    with httpx.Client(timeout=20.0) as client_http:
        resp = client_http.post(
            "https://api.sarvam.ai/speech-to-text",
            headers=headers,
            data=data,
            files=files,
        )

        # Fallback to saarika:v2.5 if saaras:v3 query failed
        if resp.status_code != 200 and target_model != "saarika:v2.5":
            data["model"] = "saarika:v2.5"
            files = {
                "file": (filename or "audio.wav", audio_bytes, content_type)
            }
            resp = client_http.post(
                "https://api.sarvam.ai/speech-to-text",
                headers=headers,
                data=data,
                files=files,
            )

        resp.raise_for_status()
        res_json = resp.json()
        return normalize_brand_name(res_json.get("transcript", "").strip())


def speech_to_text(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Convert speech audio to text using the configured provider.
    Supports English, Tamil, and Tanglish automatically.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return ""

    provider = os.getenv("STT_PROVIDER", STT_PROVIDER).lower()

    # Use Sarvam only when it is explicitly selected.
    if provider == "sarvam":
        try:
            transcript = sarvam_speech_to_text(audio_bytes, filename)
            if transcript:
                return transcript
        except Exception as e:
            print(f"[!] Sarvam AI STT warning/fallback: {e}")

    # 2. OpenAI Whisper STT (Direct whisper-1 avoids 404 delays on REST endpoint)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename

    model_name = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
    if not model_name or any(x in model_name.lower() for x in ("realtime", "live", "transcribe")):
        model_name = "whisper-1"

    try:
        transcript = client.audio.transcriptions.create(
            model=model_name,
            file=audio_file,
            prompt="MintX Solutions, Tamil, Tanglish, English, website, app, software, வணக்கம், வெல்கம்",
            response_format="text",
        )
    except Exception:
        audio_file.seek(0)
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            prompt="MintX Solutions, Tamil, Tanglish, English, website, app, software, வணக்கம், வெல்கம்",
            response_format="text",
        )

    text = transcript.strip() if isinstance(transcript, str) else str(transcript).strip()
    return normalize_brand_name(text)


def translate_audio_to_english(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Translate spoken audio in any language (e.g. Tamil) to English text.
    Uses Sarvam AI translation endpoint or Whisper/LLM fallback.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return ""

    provider = os.getenv("STT_PROVIDER", "openai").lower()
    key = os.getenv("SARVAM_API_KEY", "").strip() or SARVAM_API_KEY
    if provider == "sarvam" and key:
        try:
            import httpx
            ext = os.path.splitext(filename)[1].lower() if filename else ".wav"
            content_type = "audio/mpeg" if ext == ".mp3" else "audio/wav"
            resp = httpx.post(
                "https://api.sarvam.ai/speech-to-text-translate",
                headers={"api-subscription-key": key},
                data={"model": os.getenv("SARVAM_MODEL", "saaras:v3")},
                files={"file": (filename or "audio.wav", audio_bytes, content_type)},
                timeout=20.0
            )
            if resp.status_code == 200:
                transcript = resp.json().get("transcript", "").strip()
                if transcript:
                    return normalize_brand_name(transcript)
        except Exception as e:
            print(f"[!] Sarvam translation warning/fallback: {e}")

    # Fallback to speech_to_text + OpenAI translation
    original_text = speech_to_text(audio_bytes, filename)
    if not original_text:
        return ""

    reasoning_model = os.getenv("OPENAI_REASONING_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.6-sol"))
    if "realtime" in reasoning_model.lower():
        reasoning_model = "gpt-5.6-sol"

    try:
        completion = client.chat.completions.create(
            model=reasoning_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert translator. Translate the user input into clear, natural English. Output ONLY the translated text."
                },
                {"role": "user", "content": original_text}
            ],
            temperature=0.3,
            max_tokens=250,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert translator. Translate the user input into clear, natural English. Output ONLY the translated text."
                    },
                    {"role": "user", "content": original_text}
                ],
                temperature=0.3,
                max_tokens=250,
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            return original_text


def speech_to_text_from_file(file_path: str) -> str:
    """Convert a saved audio file to text."""
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    filename = os.path.basename(file_path)
    return speech_to_text(audio_bytes, filename)


# ── Text-to-Speech (TTS) ───────────────────────────────────────────

def text_to_speech_tamil(text: str, voice: str = "ta-IN-PallaviNeural") -> bytes:
    """
    Generate authentic Tamil speech using high-quality neural voice models (edge-tts / gTTS).
    Voices:
        - ta-IN-PallaviNeural / pallavi / tamil-female (Female, India)
        - ta-IN-ValluvarNeural / valluvar / tamil-male (Male, India)
        - ta-LK-SaranyaNeural / saranya (Female, Sri Lanka)
        - ta-LK-KumarNeural / kumar (Male, Sri Lanka)
    """
    voice_name = TAMIL_VOICES.get(voice.lower(), voice)
    
    # Try edge-tts first for pristine neural quality
    try:
        import asyncio
        import edge_tts

        async def _synthesize():
            communicate = edge_tts.Communicate(text, voice_name)
            audio_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
            return bytes(audio_data)

        # Run async in current or new event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In running loop thread, use a worker thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(lambda: asyncio.run(_synthesize())).result()
            else:
                return loop.run_until_complete(_synthesize())
        except RuntimeError:
            return asyncio.run(_synthesize())

    except Exception as e:
        # Fallback to gTTS if edge-tts fails
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang="ta", slow=False)
            mp3_io = io.BytesIO()
            tts.write_to_fp(mp3_io)
            mp3_io.seek(0)
            return mp3_io.read()
        except Exception:
            # Fallback to OpenAI TTS-1
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text,
                response_format="mp3",
            )
            return response.content


def _looks_like_tanglish(text: str) -> bool:
    """Heuristic: Tanglish / Tamil-phone speech (Roman Tamil words)."""
    if not text:
        return False
    if any("\u0B80" <= c <= "\u0BFF" for c in text):
        return True
    lower = text.lower()
    markers = (
        "vanakkam", "sollunga", "venum", "unga", "seri", "konjam", "irukka",
        "pannunga", "theriyala", "appa", "enna", "idhu", "athu", "nga",
        "aa?", "la?", "nu ", "pathi", "romba",
    )
    return any(m in lower for m in markers)



# ── Sarvam AI Text-to-Speech (TTS) Engine ───────────────────────────

def wav_to_pcm_chunks(wav_bytes: bytes, target_rate: int = 24000, chunk_size: int = 2400):
    """
    Parse a standard WAV file (e.g. Sarvam 22050Hz 16-bit mono) and resample to
    target_rate (24000Hz 16-bit mono), yielding 16-bit PCM byte chunks of chunk_size.
    Preserves pitch and tempo perfectly using linear interpolation.
    """
    import wave
    import numpy as np

    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        orig_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw_frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        dtype = np.int16 if sampwidth == 2 else (np.uint8 if sampwidth == 1 else np.int32)
        samples = np.frombuffer(raw_frames, dtype=dtype).astype(np.float32)
    else:
        samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if orig_rate != target_rate:
        orig_len = len(samples)
        target_len = int(round(orig_len * (target_rate / orig_rate)))
        orig_indices = np.linspace(0, orig_len - 1, orig_len)
        target_indices = np.linspace(0, orig_len - 1, target_len)
        samples = np.interp(target_indices, orig_indices, samples)

    pcm_int16 = np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

    for i in range(0, len(pcm_int16), chunk_size):
        chunk = pcm_int16[i:i + chunk_size]
        if len(chunk) % 2 == 1:
            chunk = chunk[:-1]
        if chunk:
            yield chunk


def text_to_speech_sarvam(
    text: str,
    speaker: str = None,
    model: str = None,
    language_code: str = None,
    pace: float = None,
) -> bytes:
    """
    Generate authentic Indian speech using Sarvam AI Text-to-Speech (bulbul:v3).
    Default speaker: 'kavitha' (Yazhini persona - warm, natural Tamil female).
    Returns RIFF/WAV 22050Hz 16-bit mono audio bytes.
    """
    import base64
    import httpx

    prepared = _prepare_sarvam_request(text, speaker, model, language_code, pace)
    if not prepared:
        return b""
    headers, payload, spk = prepared

    with httpx.Client(timeout=25.0) as client_http:
        resp = client_http.post(
            "https://api.sarvam.ai/text-to-speech",
            headers=headers,
            json=payload,
        )
        if resp.status_code != 200:
            if spk != "kavitha":
                # Fallback to kavitha if specific voice failed
                payload["speaker"] = "kavitha"
                resp = client_http.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers=headers,
                    json=payload,
                )
            resp.raise_for_status()

        data = resp.json()
        audios = data.get("audios", [])
        if not audios or not audios[0]:
            raise ValueError(f"Empty audio returned by Sarvam AI: {data}")
        return base64.b64decode(audios[0])


def _prepare_sarvam_request(
    text: str,
    speaker: str = None,
    model: str = None,
    language_code: str = None,
    pace: float = None,
):
    """Build (headers, payload, speaker) for Sarvam TTS, or None when there is nothing to speak."""
    key = os.getenv("SARVAM_API_KEY", "").strip() or SARVAM_API_KEY
    if not key:
        raise ValueError("SARVAM_API_KEY is not configured in .env or .env.local")

    tts_text = tanglish_for_tts(text)
    if not tts_text.strip():
        return None

    # Detect language code: en-IN for English, ta-IN for Tamil/Tanglish
    if not language_code:
        try:
            from language_detector import detect_language, get_sarvam_language_code
            lang = get_sarvam_language_code(detect_language(tts_text))
        except Exception:
            lang = os.getenv("SARVAM_TTS_LANGUAGE_CODE", "ta-IN")
    else:
        lang = language_code

    spk = (speaker or os.getenv("SARVAM_TTS_SPEAKER", "kavitha")).lower()
    if spk in ("yazhini", "default", "sarvam", "sarvamai", "elevenlabs"):
        spk = "kavitha"
    spk = SARVAM_SPEAKERS.get(spk, spk)

    mdl = model or os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
    sp_pace = pace if pace is not None else float(os.getenv("SARVAM_TTS_PACE", "1.25"))

    payload = {
        "text": tts_text,
        "language_code": lang,
        "speaker": spk,
        "model": mdl,
        "pace": max(0.5, min(2.0, float(sp_pace))),
    }
    if "v3" in str(mdl):
        # Bulbul v3 expressiveness: higher = more natural, conversational intonation (API default 0.6)
        try:
            temp = float(os.getenv("SARVAM_TTS_TEMPERATURE", "0.7"))
        except ValueError:
            temp = 0.75
        payload["temperature"] = max(0.01, min(2.0, temp))

    headers = {
        "api-subscription-key": key,
        "Content-Type": "application/json",
    }
    return headers, payload, spk


_sarvam_http_client = None
_sarvam_http_lock = threading.Lock()


def _get_sarvam_http_client():
    """Shared keep-alive HTTP client so each sentence skips a fresh TLS handshake to api.sarvam.ai."""
    global _sarvam_http_client
    import httpx
    with _sarvam_http_lock:
        if _sarvam_http_client is None:
            _sarvam_http_client = httpx.Client(
                timeout=httpx.Timeout(25.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=8, max_connections=16, keepalive_expiry=60.0),
            )
        return _sarvam_http_client


def stream_speech_sarvam(
    text: str,
    speaker: str = None,
    model: str = None,
    language_code: str = None,
    pace: float = None,
    chunk_size: int = 2400,
):
    """
    Generate speech with Sarvam AI and yield 24000Hz 16-bit PCM chunks for Web Audio streaming.
    Uses Sarvam's HTTP streaming endpoint (first audio in ~0.3s); falls back to the REST endpoint.
    """
    prepared = _prepare_sarvam_request(text, speaker, model, language_code, pace)
    if not prepared:
        return
    headers, payload, _spk = prepared
    stream_payload = {**payload, "output_audio_codec": "linear16", "speech_sample_rate": 24000}

    yielded_any = False
    try:
        with _get_sarvam_http_client().stream(
            "POST",
            "https://api.sarvam.ai/text-to-speech/stream",
            headers=headers,
            json=stream_payload,
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                raise RuntimeError(f"Sarvam stream HTTP {resp.status_code}: {resp.text[:200]}")
            for chunk in resp.iter_bytes(chunk_size):
                if chunk:
                    yielded_any = True
                    yield chunk
        if yielded_any:
            return
    except Exception as e:
        if yielded_any:
            raise
        print(f"[!] Sarvam streaming TTS failed, falling back to REST: {e}")

    wav_bytes = text_to_speech_sarvam(
        text=text,
        speaker=speaker,
        model=model,
        language_code=language_code,
        pace=pace,
    )
    if not wav_bytes:
        return
    for chunk in wav_to_pcm_chunks(wav_bytes, target_rate=24000, chunk_size=chunk_size):
        yield chunk


async def async_stream_speech_sarvam(
    text: str,
    speaker: str = None,
    model: str = None,
    language_code: str = None,
    pace: float = None,
):
    """
    Asynchronously stream Sarvam AI 24000Hz PCM chunks without blocking the asyncio event loop.
    """
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()
    sentinel = object()

    def _worker():
        try:
            for chunk in stream_speech_sarvam(
                text=text,
                speaker=speaker,
                model=model,
                language_code=language_code,
                pace=pace,
            ):
                loop.call_soon_threadsafe(q.put_nowait, chunk)
        except Exception as ex:
            loop.call_soon_threadsafe(q.put_nowait, ex)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, sentinel)

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        item = await q.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def _build_elevenlabs_voice_settings(
    stability=None,
    similarity_boost=None,
    style=None,
    use_speaker_boost=None,
    speed=None,
):
    """Softer, more natural Tamil-woman delivery defaults."""
    stab = stability if stability is not None else DEFAULT_ELEVENLABS_STABILITY
    sim = similarity_boost if similarity_boost is not None else DEFAULT_ELEVENLABS_SIMILARITY
    sty = style if style is not None else DEFAULT_ELEVENLABS_STYLE
    spd = speed if speed is not None else DEFAULT_ELEVENLABS_SPEED
    voice_settings = {}
    for key, val in (
        ("stability", stab),
        ("similarity_boost", sim),
        ("style", sty),
        ("speed", max(0.7, min(1.2, float(spd)))),
    ):
        try:
            voice_settings[key] = float(val)
        except Exception:
            pass
    if use_speaker_boost is None:
        use_speaker_boost = os.getenv("ELEVENLABS_SPEAKER_BOOST", "true").lower() in ("true", "1", "yes")
    voice_settings["use_speaker_boost"] = bool(use_speaker_boost)
    return voice_settings


def text_to_speech_elevenlabs(
    text: str,
    voice_id: str = None,
    model_id: str = None,
    output_format: str = "mp3_44100_128",
    stability: float = None,
    similarity_boost: float = None,
    style: float = None,
    use_speaker_boost: bool = None,
    optimize_streaming_latency: int = None,
) -> bytes:
    """
    Generate speech using ElevenLabs API (Yazhini) with natural Tamil-woman delivery.
    """
    tts_text = tanglish_for_tts(text)
    client_el = get_elevenlabs_client()
    if not client_el:
        raise ValueError("ELEVENLABS_API_KEY is not set in .env or .env.local")

    global _elevenlabs_library_restricted
    vid = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0")
    mid = model_id or os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
    opt_latency = optimize_streaming_latency if optimize_streaming_latency is not None else ELEVENLABS_OPTIMIZE_LATENCY

    target_vid = "hpp4J3VqNfWAUOO0d1Us" if (_elevenlabs_library_restricted and vid == "mGboHvCVOXWYeFL8KTR0") else vid

    kwargs = {
        "voice_id": target_vid,
        "model_id": mid,
        "text": tts_text,
    }
    if output_format:
        kwargs["output_format"] = output_format

    # Models like eleven_v4_turbo or eleven_v3 reject optimize_streaming_latency
    supports_latency_opt = not any(x in str(mid).lower() for x in ("v3", "v4", "multilingual_v1"))
    if opt_latency is not None and supports_latency_opt:
        kwargs["optimize_streaming_latency"] = opt_latency

    # Dynamic language code: English vs Tamil/Tanglish
    try:
        from language_detector import detect_language
        is_en = detect_language(tts_text) == "en"
    except Exception:
        is_en = False

    if is_en:
        kwargs["language_code"] = "en"
    elif any(x in str(mid).lower() for x in ("flash", "turbo", "v3", "multilingual")):
        kwargs["language_code"] = os.getenv("ELEVENLABS_LANGUAGE_CODE", "ta")

    kwargs["voice_settings"] = _build_elevenlabs_voice_settings(
        stability=stability,
        similarity_boost=similarity_boost,
        style=style,
        use_speaker_boost=use_speaker_boost,
    )

    try:
        response = client_el.text_to_speech.convert(**kwargs)
        return b"".join(response)
    except Exception as e:
        err_msg = str(e).lower()
        if "language_code" in err_msg and "language_code" in kwargs:
            kwargs.pop("language_code", None)
            response = client_el.text_to_speech.convert(**kwargs)
            return b"".join(response)
        if "optimize_streaming_latency" in err_msg and "optimize_streaming_latency" in kwargs:
            kwargs.pop("optimize_streaming_latency", None)
            response = client_el.text_to_speech.convert(**kwargs)
            return b"".join(response)
        if ("402" in err_msg or "paid_plan_required" in err_msg or "library voices" in err_msg) and target_vid != "hpp4J3VqNfWAUOO0d1Us":
            print(f"[!] ElevenLabs 402: Voice '{vid}' requires paid plan via API. Falling back to premade voice 'hpp4J3VqNfWAUOO0d1Us' (Bella).")
            _elevenlabs_library_restricted = True
            kwargs["voice_id"] = "hpp4J3VqNfWAUOO0d1Us"
            response = client_el.text_to_speech.convert(**kwargs)
            return b"".join(response)
        raise e


def stream_speech_elevenlabs(
    text: str,
    voice_id: str = None,
    model_id: str = None,
    output_format: str = "pcm_24000",
    stability: float = None,
    similarity_boost: float = None,
    style: float = None,
    use_speaker_boost: bool = None,
    optimize_streaming_latency: int = None,
    speed: float = None,
):
    """
    Stream audio chunks from ElevenLabs with sub-500ms real-time latency and accuracy controls.
    """
    tts_text = tanglish_for_tts(text)
    global _elevenlabs_library_restricted
    client_el = get_elevenlabs_client()
    if not client_el:
        raise ValueError("ELEVENLABS_API_KEY is not set.")

    vid = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0")
    mid = model_id or os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
    opt_latency = optimize_streaming_latency if optimize_streaming_latency is not None else ELEVENLABS_OPTIMIZE_LATENCY

    target_vid = "hpp4J3VqNfWAUOO0d1Us" if (_elevenlabs_library_restricted and vid == "mGboHvCVOXWYeFL8KTR0") else vid

    kwargs = {
        "voice_id": target_vid,
        "model_id": mid,
        "text": tts_text,
        "output_format": output_format,
    }
    # Models like eleven_v4_turbo or eleven_v3 reject optimize_streaming_latency
    supports_latency_opt = not any(x in str(mid).lower() for x in ("v3", "v4", "multilingual_v1"))
    if opt_latency is not None and supports_latency_opt:
        kwargs["optimize_streaming_latency"] = opt_latency

    # Dynamic language code: English vs Tamil/Tanglish
    try:
        from language_detector import detect_language
        is_en = detect_language(tts_text) == "en"
    except Exception:
        is_en = False

    if is_en:
        kwargs["language_code"] = "en"
    elif any(x in str(mid).lower() for x in ("flash", "turbo", "v3", "multilingual")):
        kwargs["language_code"] = os.getenv("ELEVENLABS_LANGUAGE_CODE", "ta")

    kwargs["voice_settings"] = _build_elevenlabs_voice_settings(
        stability=stability,
        similarity_boost=similarity_boost,
        style=style,
        use_speaker_boost=use_speaker_boost,
        speed=speed,
    )

    try:
        gen = client_el.text_to_speech.stream(**kwargs)
        for chunk in gen:
            yield chunk
    except Exception as e:
        err_msg = str(e).lower()
        if "language_code" in err_msg and "language_code" in kwargs:
            kwargs.pop("language_code", None)
            gen = client_el.text_to_speech.stream(**kwargs)
            for chunk in gen:
                yield chunk
            return
        if "optimize_streaming_latency" in err_msg and "optimize_streaming_latency" in kwargs:
            kwargs.pop("optimize_streaming_latency", None)
            gen = client_el.text_to_speech.stream(**kwargs)
            for chunk in gen:
                yield chunk
            return
        if ("402" in err_msg or "paid_plan_required" in err_msg or "library voices" in err_msg) and target_vid != "hpp4J3VqNfWAUOO0d1Us":
            print(f"[!] ElevenLabs 402: Voice '{vid}' requires paid plan via API. Falling back to premade voice 'hpp4J3VqNfWAUOO0d1Us' (Bella).")
            _elevenlabs_library_restricted = True
            kwargs["voice_id"] = "hpp4J3VqNfWAUOO0d1Us"
            gen = client_el.text_to_speech.stream(**kwargs)
            for chunk in gen:
                yield chunk
        else:
            raise e


async def async_stream_speech_elevenlabs(
    text: str,
    voice_id: str = None,
    model_id: str = None,
    output_format: str = "pcm_24000",
    stability: float = None,
    similarity_boost: float = None,
    style: float = None,
    use_speaker_boost: bool = None,
    optimize_streaming_latency: int = None,
    speed: float = None,
):
    """
    Asynchronously stream ElevenLabs audio chunks without blocking the asyncio event loop.
    """
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()
    sentinel = object()

    def _worker():
        try:
            for chunk in stream_speech_elevenlabs(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                output_format=output_format,
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                use_speaker_boost=use_speaker_boost,
                optimize_streaming_latency=optimize_streaming_latency,
                speed=speed,
            ):
                loop.call_soon_threadsafe(q.put_nowait, chunk)
        except Exception as ex:
            loop.call_soon_threadsafe(q.put_nowait, ex)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, sentinel)

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        item = await q.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def text_to_speech_openai(
    text: str,
    voice: str = None,
    model: str = None,
    response_format: str = "mp3"
) -> bytes:
    """
    Generate speech using OpenAI GPT Text-to-Speech (TTS-1 / TTS-1-HD).
    Supports standard voices: nova, alloy, shimmer, echo, fable, onyx.
    """
    raw_voice = voice or os.getenv("OPENAI_VOICE", "nova")
    target_voice = str(raw_voice).lower()
    if target_voice not in {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}:
        target_voice = "nova"

    tts_model = model or os.getenv("OPENAI_TTS_MODEL", "tts-1")
    if tts_model not in ("tts-1", "tts-1-hd"):
        tts_model = "tts-1"

    response = client.audio.speech.create(
        model=tts_model,
        voice=target_voice,
        input=text,
        response_format=response_format,
    )
    return response.content


def text_to_speech(text: str, voice: str = None) -> bytes:
    """
    Convert text to speech audio.

    Defaults to Sarvam AI (bulbul:v3 with speaker kavitha) for authentic,
    high-quality Indian speech.
    """
    clean_txt = (text or "").strip()
    if not clean_txt:
        return b""

    provider = os.getenv("TTS_PROVIDER", "sarvam").lower()
    voice_str = str(voice or "").lower()

    # 1. Sarvam AI Voice Path (Default)
    is_sarvam = (
        provider in ("sarvam", "sarvamai")
        or voice_str in ("sarvam", "kavitha", "kavya", "priya", "shreya", "roopa", "mani", "gokul", "vijay", "yazhini")
        or os.getenv("OPENAI_VOICE", "").lower() in ("sarvam", "kavitha")
    )
    if is_sarvam or (provider != "elevenlabs" and provider not in ("openai", "gpt", "gpt-speech", "tts")):
        target_spk = (
            voice
            if (voice and voice_str not in ("sarvam", "default", "yazhini", "elevenlabs") and len(str(voice)) < 20)
            else os.getenv("SARVAM_TTS_SPEAKER", "kavitha")
        )
        return text_to_speech_sarvam(clean_txt, speaker=target_spk)

    # 2. ElevenLabs fallback if explicitly set
    if provider == "elevenlabs" or voice_str in ("elevenlabs",) or len(str(voice or "")) >= 20:
        eleven_key = os.getenv("ELEVENLABS_API_KEY")
        if eleven_key:
            target_vid = (
                voice
                if (voice and voice_str not in ("elevenlabs",) and len(str(voice)) >= 20)
                else os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0")
            )
            try:
                return text_to_speech_elevenlabs(clean_txt, voice_id=target_vid)
            except Exception as e:
                print(f"[!] ElevenLabs error: {e}. Falling back to Sarvam AI (kavitha)...")
                return text_to_speech_sarvam(clean_txt, speaker="kavitha")
        else:
            return text_to_speech_sarvam(clean_txt, speaker="kavitha")

    # 3. Explicit OpenAI TTS only when requested
    if provider in ("openai", "gpt", "gpt-speech", "tts"):
        return text_to_speech_openai(clean_txt, voice=voice)

    return text_to_speech_sarvam(clean_txt, speaker="kavitha")


def speak_tamil(text: str, voice: str = "kavitha", play_aloud: bool = True) -> bytes:
    """
    Convert Tamil text to speech using Sarvam AI and optionally play aloud.
    """
    audio_bytes = text_to_speech(text, voice=voice)
    if play_aloud:
        play_audio_bytes(audio_bytes)
    return audio_bytes


def play_audio_bytes(audio_bytes: bytes):
    """
    Play MP3 or WAV audio bytes through the computer speakers.
    Uses sounddevice + soundfile for WAV and pygame for MP3.
    """
    if not audio_bytes:
        return

    # Check if RIFF WAV format
    if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:12]:
        try:
            wav_io = io.BytesIO(audio_bytes)
            data, fs = sf.read(wav_io, dtype='float32')
            sd.play(data, fs)
            sd.wait()
            return
        except Exception as e:
            print(f"[!] Sounddevice WAV play warning: {e}")

    _init_audio_playback()
    import pygame

    audio_io = io.BytesIO(audio_bytes)
    pygame.mixer.music.load(audio_io)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.05)


def play_audio_pcm_24k(pcm_bytes: bytes):
    """
    Play raw 24000Hz mono 16-bit PCM audio bytes directly.
    """
    if not pcm_bytes:
        return
    import numpy as np
    audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
    sd.play(audio_data, samplerate=24000)
    sd.wait()


def speak(text: str, voice: str = None, play_aloud: bool = True) -> bytes:
    """
    Convert text to speech and optionally play it aloud immediately.
    """
    voice_to_use = voice or DEFAULT_VOICE
    audio_bytes = text_to_speech(text, voice=voice_to_use)
    if play_aloud:
        play_audio_bytes(audio_bytes)
    return audio_bytes


def text_to_speech_save(text: str, voice: str = None) -> str:
    """Convert text to speech and save to an audio file (WAV or MP3) with instant disk caching."""
    import hashlib
    clean_txt = str(text).strip()
    if not clean_txt:
        return ""

    provider = os.getenv("TTS_PROVIDER", "sarvam").lower()
    voice_to_use = voice or (os.getenv("SARVAM_TTS_SPEAKER", "kavitha") if provider in ("sarvam", "sarvamai") else DEFAULT_VOICE)
    model_id = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3") if provider in ("sarvam", "sarvamai") else os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
    cache_key = hashlib.md5(f"{clean_txt}:{voice_to_use}:{model_id}".encode("utf-8")).hexdigest()
    ext = "wav" if provider in ("sarvam", "sarvamai") else "mp3"
    cache_file = AUDIO_DIR / f"tts_{cache_key}.{ext}"

    if cache_file.exists() and cache_file.stat().st_size > 1000:
        return str(cache_file)

    audio_bytes = text_to_speech(clean_txt, voice_to_use)
    with open(cache_file, "wb") as f:
        f.write(audio_bytes)

    return str(cache_file)


# ── Audio Utilities ─────────────────────────────────────────────────

def get_audio_file_path(filename: str) -> str:
    """Get the full path to an audio file in the audio directory."""
    return str(AUDIO_DIR / filename)


def cleanup_audio_file(file_path: str):
    """Delete a temporary audio file after it's been played, preserving cache."""
    try:
        filename = os.path.basename(file_path)
        if filename.startswith("tts_"):
            return  # keep in cache for zero-latency instant replay
        os.remove(file_path)
    except OSError:
        pass
