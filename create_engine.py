import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import AsyncElevenLabs, ElevenLabs

load_dotenv()

async def main():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("[!] Error: ELEVENLABS_API_KEY is not set in .env")
        return

    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0")
    ws_url = os.getenv("PUBLIC_WS_URL", "wss://flex-nationally-wishing-stopped.trycloudflare.com/ws")
    print(f"[*] Registering Speech Engine with WebSocket URL: {ws_url}")

    elevenlabs_async = AsyncElevenLabs(api_key=api_key)

    engine = await elevenlabs_async.speech_engine.create(
        name="MintX Voice Assistant",
        speech_engine={
            "ws_url": ws_url
        },
    )

    engine_id = getattr(engine, "speech_engine_id", getattr(engine, "engine_id", engine))
    print(f"[+] Created Speech Engine ID: {engine_id}")

    # Set TTS voice to Yazhini (or configured voice_id)
    try:
        client_sync = ElevenLabs(api_key=api_key)
        client_sync.speech_engine.update(
            engine_id,
            tts={"voice_id": voice_id}
        )
        print(f"[+] Configured TTS voice to: {voice_id}")
    except Exception as e:
        print(f"[!] Warning updating voice: {e}")

    # Automatically update .env and .env.local files
    import re
    for env_name in [".env", ".env.local"]:
        env_path = Path(__file__).parent / env_name
        if env_path.exists():
            content = env_path.read_text(encoding="utf-8")
            if "ELEVENLABS_SPEECH_ENGINE_ID=" in content:
                content = re.sub(r"ELEVENLABS_SPEECH_ENGINE_ID=.*", f"ELEVENLABS_SPEECH_ENGINE_ID={engine_id}", content)
            else:
                content += f"\nELEVENLABS_SPEECH_ENGINE_ID={engine_id}\n"
            env_path.write_text(content, encoding="utf-8")
            print(f"[+] Updated {env_name} with ELEVENLABS_SPEECH_ENGINE_ID={engine_id}")

    print(f"\n[OK] Speech Engine successfully configured!")
    print(f"   ID:       {engine_id}")
    print(f"   WSS URL:  {ws_url}")
    print(f"   Voice:    {voice_id}")

if __name__ == "__main__":
    asyncio.run(main())
