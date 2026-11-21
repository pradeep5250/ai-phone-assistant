"""
realtime_chat.py — OpenAI Realtime Voice Streaming Assistant

Connects directly to OpenAI's Realtime WebSocket API using:
- Model: gpt-realtime-2.1
- Voice: marin
- User Audio Transcription: gpt-realtime-whisper
- Format: PCM 24000Hz (input & output)
- Turn detection: server_vad
- Noise reduction: far_field
- Reasoning effort: low

Usage:
    python realtime_chat.py
"""

import os
import sys
import json
import base64
import asyncio
import queue
import threading
import inspect
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

import websockets
import sounddevice as sd
import numpy as np
from voice import REALTIME_SESSION_CONFIG, DEFAULT_VOICE
from instructions import get_active_system_prompt

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
VOICE = os.getenv("OPENAI_VOICE", "marin")
WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"


async def audio_streamer(ws):
    """Capture mic audio at 24000Hz PCM and stream to Realtime API."""
    loop = asyncio.get_event_loop()
    audio_queue = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            pass
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

    stream = sd.InputStream(samplerate=24000, channels=1, dtype="int16", callback=callback)
    with stream:
        print("\n🎙️  [LIVE REALTIME STREAMING] Speak into your microphone anytime...")
        print("👉 Press Ctrl+C in terminal to stop.")
        while True:
            audio_chunk = await audio_queue.get()
            audio_bytes = audio_chunk.tobytes()
            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            event = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }
            await ws.send(json.dumps(event))


def audio_playback_worker(audio_play_queue: queue.Queue, stop_event: threading.Event):
    """Background worker that continuously streams audio out to speakers at 24000Hz."""
    play_stream = sd.OutputStream(samplerate=24000, channels=1, dtype="int16")
    with play_stream:
        while not stop_event.is_set():
            try:
                audio_data = audio_play_queue.get(timeout=0.05)
                play_stream.write(audio_data)
            except queue.Empty:
                pass


async def audio_receiver(ws):
    """Receive audio and text streams from OpenAI Realtime API and play aloud."""
    audio_play_queue = queue.Queue()
    stop_event = threading.Event()

    # Start audio playback thread for smooth, non-blocking speaker playback
    playback_thread = threading.Thread(
        target=audio_playback_worker,
        args=(audio_play_queue, stop_event),
        daemon=True
    )
    playback_thread.start()

    speaking_header_printed = False
    assistant_text_buffer = []
    tts_prov = os.getenv("TTS_PROVIDER", "sarvam").lower()
    use_sarvam = tts_prov in ("sarvam", "sarvamai") or bool(os.getenv("SARVAM_API_KEY"))

    try:
        while True:
            message = await ws.recv()
            event = json.loads(message)
            event_type = event.get("type", "")

            # 1. Audio output chunks from OpenAI (used if not external TTS)
            if event_type in ("response.output_audio.delta", "response.audio.delta"):
                if not use_sarvam and tts_prov not in ("elevenlabs",):
                    delta_b64 = event.get("delta", "")
                    if delta_b64:
                        pcm_chunk = base64.b64decode(delta_b64)
                        audio_data = np.frombuffer(pcm_chunk, dtype=np.int16)
                        audio_play_queue.put(audio_data)

            # 2. Assistant spoken text transcript streaming
            elif event_type in ("response.output_audio_transcript.delta", "response.audio_transcript.delta", "response.text.delta"):
                delta_text = event.get("delta", "")
                assistant_text_buffer.append(delta_text)
                if not speaking_header_printed:
                    spk_name = os.getenv("SARVAM_TTS_SPEAKER", "kavitha")
                    print(f"\n🤖 [Mintx Solutions AI (Sarvam AI • {spk_name})]: ", end="", flush=True)
                    speaking_header_printed = True
                print(delta_text, end="", flush=True)

            elif event_type in ("response.output_audio_transcript.done", "response.audio_transcript.done", "response.done"):
                if speaking_header_printed:
                    print()
                    speaking_header_printed = False

                # Play audio using Sarvam AI voice
                full_text = "".join(assistant_text_buffer).strip()
                assistant_text_buffer.clear()

                if full_text:
                    try:
                        from voice import text_to_speech_sarvam, wav_to_pcm_chunks
                        wav_bytes = text_to_speech_sarvam(full_text)
                        pcm_chunks = list(wav_to_pcm_chunks(wav_bytes, target_rate=24000))
                        full_pcm = b"".join(pcm_chunks)
                        audio_data = np.frombuffer(full_pcm, dtype=np.int16)
                        audio_play_queue.put(audio_data)
                    except Exception as e:
                        print(f"\n[!] Sarvam AI audio error: {e}")

            # 3. User speech transcription completed
            elif event_type in ("conversation.item.input_audio_transcription.completed", "input_audio_transcription.completed"):
                transcript = event.get("transcript", "").strip()
                if transcript:
                    print(f"\n🗣️  [You (Voice)]: \"{transcript}\"")
                    speaking_header_printed = False

            # 4. User stopped speaking -> ensure response starts if not automatically triggered
            elif event_type == "input_audio_buffer.speech_stopped":
                pass

            # 5. Error handling
            elif event_type == "error":
                print(f"\n❌ [Realtime Error]: {event.get('error')}")

    finally:
        stop_event.set()


async def main():
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY is not configured in .env file.")
        return

    print("=" * 65)
    print("  ⚡ OpenAI Realtime Voice Assistant (WebSocket Mode) ⚡")
    print(f"  🏢 Company: Mintx Solutions")
    print(f"  🔊 Voice: {VOICE} | Model: {REALTIME_MODEL}")
    print(f"  🎙️ Transcription Model: gpt-realtime-whisper")
    print("  🌐 Tip: Run with --ui to launch the interactive Web Studio UI!")
    print("=" * 65)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    # Support websockets 14+ (additional_headers) and older versions (extra_headers)
    sig = inspect.signature(websockets.connect)
    header_kwarg = "additional_headers" if "additional_headers" in sig.parameters else "extra_headers"
    connect_kwargs = {header_kwarg: headers}

    try:
        async with websockets.connect(WS_URL, **connect_kwargs) as ws:
            print("✅ Connected to OpenAI Realtime API!")

            # Prepare session update with system prompt & exact settings
            session_payload = json.loads(json.dumps(REALTIME_SESSION_CONFIG))
            session_payload["session"]["instructions"] = get_active_system_prompt()
            session_payload["session"]["audio"]["output"]["voice"] = VOICE

            session_update = {
                "type": "session.update",
                "session": session_payload["session"]
            }
            await ws.send(json.dumps(session_update))

            # Run streamer and receiver concurrently
            await asyncio.gather(
                audio_streamer(ws),
                audio_receiver(ws)
            )

    except Exception as e:
        print(f"\n[Realtime Session Ended]: {e}")


if __name__ == "__main__":
    if "--ui" in sys.argv or "--web" in sys.argv:
        from realtime_ui import main as launch_ui
        launch_ui()
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nSession ended. Goodbye!")
