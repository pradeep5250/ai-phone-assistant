"""
server.py — ElevenLabs Speech Engine Upstream WebSocket Server for MintX Solutions

Connects:
ElevenLabs Speech Engine (ASR/TTS) ⟷ OpenAI LLM (AI Brain) ⟷ Client Audio
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from ai_assistant import AIAssistant
from instructions import get_active_system_prompt, get_active_instruction
from elevenlabs.speech_engine.session import SpeechEngineSession
from elevenlabs.speech_engine.types import ConversationMessage

app = FastAPI(title="MintX Voice Assistant — ElevenLabs Speech Engine Server")

# Global assistant session cache per conversation
sessions: dict[str, AIAssistant] = {}

def get_or_create_assistant(conv_id: str, custom_instruction: str = None) -> AIAssistant:
    if conv_id not in sessions:
        prompt = custom_instruction or get_active_system_prompt()
        sessions[conv_id] = AIAssistant(call_id=f"el_engine_{conv_id}", system_prompt=prompt)
    return sessions[conv_id]


@app.get("/health")
async def health():
    engine_id = os.getenv("ELEVENLABS_SPEECH_ENGINE_ID", "seng_8401m2ms7t5pfv9azvcsn9tdgwr1")
    active_inst = get_active_instruction()
    return {
        "status": "ok",
        "service": "MintX ElevenLabs Speech Engine Server",
        "speech_engine_id": engine_id,
        "active_voice": os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0"),
        "active_instruction": active_inst.get("name") if active_inst else "MintX Default"
    }


@app.websocket("/ws")
async def speech_engine_websocket_endpoint(websocket: WebSocket):
    """
    Upstream WebSocket endpoint called by ElevenLabs Speech Engine.
    ElevenLabs connects here when a user initiates a conversation.
    """
    await websocket.accept()
    print("\n[+] ElevenLabs Speech Engine WebSocket connected!")

    session = SpeechEngineSession(websocket, debug=True)
    conv_id = "default"

    async def on_init(conversation_id: str):
        nonlocal conv_id
        conv_id = conversation_id
        print(f"[ElevenLabs Init] Conversation ID: {conversation_id}")
        get_or_create_assistant(conv_id)

    async def on_transcript(transcript: list[ConversationMessage]):
        if not transcript:
            return

        last_msg = transcript[-1]
        user_text = last_msg.content.strip()
        print(f"\n🗣️ [User Transcript]: {user_text}")

        if not user_text:
            return

        # Get AI response from MintX Assistant (OpenAI LLM + System Prompt)
        bot = get_or_create_assistant(conv_id)
        reply = bot.chat(user_text)
        print(f"🤖 [Agent Response]: {reply}")

        # Send back to ElevenLabs for TTS synthesis
        await session.send_response(reply)

    async def on_close():
        print(f"[-] ElevenLabs session closed: {conv_id}")
        if conv_id in sessions:
            del sessions[conv_id]

    async def on_error(err: Exception):
        print(f"[!] ElevenLabs session error: {err}")

    session.on("init", on_init)
    session.on("user_transcript", on_transcript)
    session.on("close", on_close)
    session.on("error", on_error)

    try:
        await session.run()
    except WebSocketDisconnect:
        print(f"[-] ElevenLabs WebSocket disconnected: {conv_id}")
    except Exception as e:
        print(f"[!] WebSocket loop error: {e}")


def main():
    port = int(os.getenv("SERVER_PORT", "8000"))
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    engine_id = os.getenv("ELEVENLABS_SPEECH_ENGINE_ID", "seng_8401m2ms7t5pfv9azvcsn9tdgwr1")
    public_ws = os.getenv("PUBLIC_WS_URL", "wss://placed-sponsorship-utilization-checking.trycloudflare.com/ws")

    print("\n" + "=" * 62)
    print("  🎙️  MINTX SOLUTIONS — ELEVENLABS SPEECH ENGINE SERVER  🎙️")
    print("=" * 62)
    print(f"  ⚡ Speech Engine ID:  {engine_id}")
    print(f"  🔊 TTS Voice ID:     {os.getenv('ELEVENLABS_VOICE_ID', 'mGboHvCVOXWYeFL8KTR0')} (Yazhini)")
    print(f"  🌐 Public WSS URL:   {public_ws}")
    print(f"  🔌 Local Server:     http://localhost:{port}")
    print("=" * 62 + "\n")

    uvicorn.run("server:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
