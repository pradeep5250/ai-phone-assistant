"""
voice.py — Voice Processing

Handles speech-to-text (STT), text-to-speech (TTS), microphone recording,
and speaker playback using OpenAI APIs.
- STT: OpenAI Whisper (supports English and Tamil)
- TTS: OpenAI TTS-1 (natural-sounding voice)
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

import sounddevice as sd
import soundfile as sf

# Load environment
load_dotenv()

# ── Configuration ───────────────────────────────────────────────────

AUDIO_DIR = Path(__file__).parent / "audio_files"
AUDIO_DIR.mkdir(exist_ok=True)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Default voice and supported voices
DEFAULT_VOICE = os.getenv("OPENAI_VOICE", "marin")
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
    "marin", "alloy", "echo", "fable", "onyx", "nova", "shimmer", "verse", "ballad", "ash", "sage", "coral",
    "pallavi", "valluvar", "saranya", "kumar", "tamil-female", "tamil-male"
]
STANDARD_TTS_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "sage", "coral"}

# OpenAI Realtime Session Configuration (Realtime Mode)
REALTIME_SESSION_CONFIG = {
    "session": {
        "type": "realtime",
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": 24000
                },
                "transcription": {
                    "model": "gpt-realtime-whisper"
                },
                "noise_reduction": {
                    "type": "far_field"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "idle_timeout_ms": None
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
        "max_output_tokens": "inf",
        "reasoning": {
            "effort": "low"
        }
    }
}


def get_realtime_session_config(voice: str = None, reasoning_effort: str = "low", modalities: list = None) -> dict:
    """
    Get customized OpenAI Realtime Session Configuration.
    
    Args:
        voice: Output voice name (e.g. 'marin', 'alloy', 'coral', 'sage', 'ash', 'shimmer').
        reasoning_effort: Reasoning effort level ('low', 'medium', 'high').
        modalities: Output modalities (e.g. ['audio'] or ['text', 'audio']).
    """
    config = {
        "session": {
            "type": "realtime",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    },
                    "transcription": {
                        "model": "gpt-realtime-whisper"
                    },
                    "noise_reduction": {
                        "type": "far_field"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "idle_timeout_ms": None
                    }
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    },
                    "voice": voice or DEFAULT_VOICE
                }
            },
            "output_modalities": modalities or ["audio"],
            "tools": [],
            "max_output_tokens": "inf",
            "reasoning": {
                "effort": reasoning_effort
            }
        }
    }
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


# ── Speech-to-Text (STT) ───────────────────────────────────────────

def speech_to_text(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Convert speech audio to text using OpenAI Whisper.
    Supports English and Tamil automatically.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return ""

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text",
    )

    text = transcript.strip()
    return text


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


def text_to_speech(text: str, voice: str = None) -> bytes:
    """
    Convert text to natural speech audio using OpenAI TTS or Native Tamil Neural TTS.
    - Standard voices: marin, alloy, nova, shimmer, echo, onyx, fable, ash, sage, coral
    - Tamil voices: pallavi (female), valluvar (male), saranya, kumar, tamil-female, tamil-male
    """
    target_voice = (voice or DEFAULT_VOICE).lower()

    # If a Tamil voice is specified, route to dedicated Tamil neural synthesizer
    if target_voice in TAMIL_VOICES or target_voice.startswith("ta-"):
        return text_to_speech_tamil(text, voice=target_voice)

    # For standard TTS-1 endpoint, map preview/realtime-only voices if needed
    tts_voice = target_voice
    if tts_voice not in STANDARD_TTS_VOICES:
        tts_voice = "alloy"

    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=tts_voice,
            input=text,
            response_format="mp3",
        )
        return response.content
    except Exception as e:
        if tts_voice != "alloy":
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="mp3",
            )
            return response.content
        raise e


def speak_tamil(text: str, voice: str = "pallavi", play_aloud: bool = True) -> bytes:
    """
    Convert Tamil text to speech using Tamil Neural voice and optionally play aloud.
    """
    audio_bytes = text_to_speech_tamil(text, voice=voice)
    if play_aloud:
        play_audio_bytes(audio_bytes)
    return audio_bytes


def play_audio_bytes(audio_bytes: bytes):
    """
    Play MP3 or WAV audio bytes through the computer speakers.
    """
    if not audio_bytes:
        return

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
    """Convert text to speech and save to a file."""
    voice_to_use = voice or DEFAULT_VOICE
    audio_bytes = text_to_speech(text, voice_to_use)
    filename = f"{uuid.uuid4().hex}.mp3"
    file_path = AUDIO_DIR / filename

    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    return str(file_path)


# ── Audio Utilities ─────────────────────────────────────────────────

def get_audio_file_path(filename: str) -> str:
    """Get the full path to an audio file in the audio directory."""
    return str(AUDIO_DIR / filename)


def cleanup_audio_file(file_path: str):
    """Delete a temporary audio file after it's been played."""
    try:
        os.remove(file_path)
    except OSError:
        pass
