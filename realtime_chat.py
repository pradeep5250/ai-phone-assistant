"""
realtime_chat.py — OpenAI Realtime Voice Streaming Assistant

Connects directly to OpenAI's Realtime WebSocket API using:
- Voice: marin
- Modality: Realtime Audio streaming
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
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

import websockets
import sounddevice as sd
import numpy as np
from voice import REALTIME_SESSION_CONFIG, DEFAULT_VOICE

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
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

    # 24000Hz, 1 channel, 16-bit PCM
    stream = sd.InputStream(samplerate=24000, channels=1, dtype="int16", callback=callback)
    with stream:
        print("🎙️  [LIVE REALTIME STREAMING] Speak into your microphone anytime...")
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


async def audio_receiver(ws):
    """Receive audio from OpenAI Realtime API and play aloud at 24000Hz."""
    # Playback output stream
    play_stream = sd.OutputStream(samplerate=24000, channels=1, dtype="int16")
    play_stream.start()

    try:
        while True:
            message = await ws.recv()
            event = json.loads(message)
            event_type = event.get("type", "")

            if event_type == "response.audio.delta":
                delta_b64 = event.get("delta", "")
                if delta_b64:
                    pcm_chunk = base64.b64decode(delta_b64)
                    audio_data = np.frombuffer(pcm_chunk, dtype=np.int16)
                    play_stream.write(audio_data)

            elif event_type == "response.audio_transcript.delta":
                delta_text = event.get("delta", "")
                print(delta_text, end="", flush=True)

            elif event_type == "response.audio_transcript.done":
                print()  # New line after assistant finishes speaking

            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "").strip()
                if transcript:
                    print(f"\n🗣️  [You (Realtime)]: {transcript}")
                    print("🤖 [AI Assistant (Voice: marin)]: ", end="", flush=True)

            elif event_type == "error":
                print(f"\n[Realtime Error]: {event.get('error')}")

    finally:
        play_stream.stop()
        play_stream.close()


async def main():
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY is not configured in .env file.")
        return

    print("=" * 65)
    print(f"  ⚡ OpenAI Realtime Voice Assistant (WebSocket Mode) ⚡")
    print(f"  🔊 Voice: {VOICE} | Model: {REALTIME_MODEL}")
    print("=" * 65)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    try:
        async with websockets.connect(WS_URL, extra_headers=headers) as ws:
            print(" Connected to OpenAI Realtime API!")

            # Send session update matching user configuration
            session_update = {
                "type": "session.update",
                "session": REALTIME_SESSION_CONFIG["session"]
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended. Goodbye!")
