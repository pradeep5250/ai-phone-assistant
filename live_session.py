"""
GPT-Live (gpt-live-1) bridge for the Realtime Voice Studio.

GPT-Live is a full-duplex speech-to-speech model on its own API (wss://api.openai.com/v1/live/sessions):
it listens, decides when to talk and speaks with its own voice. This module connects one browser call
to one Live session and translates the Live events into the Realtime-style events the UI already handles.
"""

import asyncio
import inspect
import json
import socket
import time

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from language_detector import detect_language, get_language_display_name

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
LIVE_MODEL = "gpt-live-1"
LIVE_VOICES = (
    "marin", "cedar", "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse",
    "quartz", "ripple", "vesper", "willow", "stone", "gleam", "meridian", "bossa", "tempo",
    "beacon", "delta", "cinder",
)
DEFAULT_LIVE_VOICE = "marin"

# Live transcript deltas have no end-of-turn event; a turn is considered finished after this much quiet.
CALLER_TURN_GAP_S = 1.5
ASSISTANT_TURN_GAP_S = 1.2

LIVE_DIRECTIVE = (
    "\n\n## GPT-LIVE CALL MODE (overrides any tool instructions above):\n"
    "* You are on a live phone call and hear the caller directly. Speak naturally; you can listen while speaking.\n"
    "* You have no tools in this call. Do not mention or call `search_mintx_knowledge`, `save_customer_memory` or "
    "`train_voice_correction`. Answer MintX questions from the MINTX VERIFIED KNOWLEDGE section.\n"
    "* Only if a MintX fact is genuinely missing from that section, delegate the question to the backend and tell "
    "the caller in one short phrase that you are checking.\n"
    "* Keep every reply short (1-2 spoken sentences) and ask only one question at a time.\n"
)


def resolve_live_voice(voice: str | None) -> str:
    v = (voice or "").strip().lower()
    return v if v in LIVE_VOICES else DEFAULT_LIVE_VOICE


async def _connect(api_key: str):
    headers = {"Authorization": f"Bearer {api_key}"}
    header_kwarg = "additional_headers" if "additional_headers" in inspect.signature(websockets.connect).parameters else "extra_headers"
    kwargs = {header_kwarg: headers, "open_timeout": 15, "ping_interval": 20, "ping_timeout": 20,
              "max_size": None, "proxy": None, "family": socket.AF_INET}
    try:
        return await websockets.connect(LIVE_URL, **kwargs)
    except TypeError:
        kwargs.pop("family", None)
        kwargs.pop("proxy", None)
        return await websockets.connect(LIVE_URL, **kwargs)


async def run_live_session(client_ws: WebSocket, api_key: str, build_instructions):
    """
    Serve one browser call with GPT-Live.

    build_instructions(init_msg) -> str returns the full persona + knowledge prompt for this call.
    """
    async def to_client(payload: dict):
        try:
            await client_ws.send_text(json.dumps(payload))
        except Exception:
            pass

    # The browser sends client.init (voice, greeting, instructions) as its first message.
    init_msg = {}
    pre_audio: list[str] = []
    try:
        while True:
            msg = json.loads(await asyncio.wait_for(client_ws.receive_text(), timeout=15))
            if msg.get("type") == "client.init":
                init_msg = msg
                break
            if msg.get("type") == "input_audio_buffer.append" and msg.get("audio"):
                pre_audio.append(msg["audio"])
    except (asyncio.TimeoutError, WebSocketDisconnect, json.JSONDecodeError):
        pass

    voice = resolve_live_voice(init_msg.get("voice"))
    instructions = build_instructions(init_msg) + LIVE_DIRECTIVE
    greeting = (init_msg.get("starting_greeting") or "").strip()

    try:
        live_ws = await _connect(api_key)
    except Exception as e:
        err = str(e)
        code = "insufficient_quota" if ("insufficient_quota" in err or "1013" in err) else "connection_failed"
        await to_client({"type": "error", "error": {"code": code, "message": f"Failed to connect to GPT-Live: {err}"}})
        try:
            await client_ws.close()
        except Exception:
            pass
        return

    await live_ws.send(json.dumps({
        "type": "session.start",
        "event_id": "event_start",
        "session": {
            "model": LIVE_MODEL,
            "instructions": instructions,
            "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": voice}},
            "delegation": {"type": "client"},
        },
    }))
    print(f"[GPT-Live] Session starting (voice: {voice}, instructions: {len(instructions)} chars)")

    started = asyncio.Event()
    state = {
        "caller_text": "", "caller_last": 0.0,
        "assistant_text": "", "assistant_last": 0.0,
        "language": None,
        "closed": False,
    }
    last_caller_utterance = {"text": ""}

    async def finish_caller_turn():
        text = state["caller_text"].strip()
        state["caller_text"] = ""
        if not text:
            return
        last_caller_utterance["text"] = text
        await to_client({"type": "conversation.item.input_audio_transcription.completed", "transcript": text})
        try:
            lang = detect_language(text)
            if lang != state["language"]:
                state["language"] = lang
                await to_client({"type": "customer.language.detected", "language": lang,
                                 "display_name": get_language_display_name(lang)})
        except Exception:
            pass

    async def finish_assistant_turn():
        if state["assistant_text"].strip():
            state["assistant_text"] = ""
            await to_client({"type": "response.output_audio_transcript.done"})

    async def turn_watcher():
        while not state["closed"]:
            await asyncio.sleep(0.2)
            now = time.monotonic()
            if state["caller_text"] and now - state["caller_last"] > CALLER_TURN_GAP_S:
                await finish_caller_turn()
            if state["assistant_text"] and now - state["assistant_last"] > ASSISTANT_TURN_GAP_S:
                await finish_assistant_turn()

    async def answer_delegation(delegation_id: str):
        question = (state["caller_text"] or last_caller_utterance["text"]).strip()
        facts = ""
        if question:
            try:
                from rag import search_mintx_knowledge
                facts = await asyncio.to_thread(search_mintx_knowledge, question, 3)
            except Exception as e:
                print(f"[GPT-Live] Knowledge lookup failed: {e}")
        content = (
            f"Verified MintX Solutions facts for the caller's question: {facts[:1800]}"
            if facts else
            "No verified MintX Solutions information was found for this question. Tell the caller the team will share exact details."
        )
        await live_ws.send(json.dumps({
            "type": "session.commentary.append",
            "event_id": f"delegation_{delegation_id}",
            "delegation_id": delegation_id,
            "content": content,
        }))

    async def live_to_client():
        async for raw in live_ws:
            ev = json.loads(raw)
            t = ev.get("type", "")
            if t == "session.output_audio.delta":
                await to_client({"type": "response.output_audio.delta", "delta": ev.get("delta", "")})
            elif t == "session.output_transcript.delta":
                if state["caller_text"]:
                    await finish_caller_turn()
                state["assistant_text"] += ev.get("delta", "")
                state["assistant_last"] = time.monotonic()
                await to_client({"type": "response.output_audio_transcript.delta", "delta": ev.get("delta", "")})
            elif t == "session.input_transcript.delta":
                now = time.monotonic()
                state["caller_text"] += ev.get("delta", "")
                state["caller_last"] = now
                await to_client({"type": "caller.transcript.partial", "text": state["caller_text"].strip()})
            elif t == "session.started":
                started.set()
                print(f"[GPT-Live] Session ready: {ev.get('session', {}).get('id')}")
                await to_client({"type": "stt.engine", "engine": "gpt-live"})
                if init_msg.get("auto_greet", True):
                    line = f" Say: '{greeting}'" if greeting else ""
                    # commentary.append greets reliably; instructions.append was sometimes ignored.
                    await live_ws.send(json.dumps({
                        "type": "session.commentary.append",
                        "event_id": "greeting",
                        "delegation_id": None,
                        "content": f"Greet the caller now as Yazhini from MintX Solutions in friendly Thanglish.{line} Then pause and listen.",
                    }))
            elif t == "session.delegation.created":
                delegation = ev.get("delegation") or {}
                if delegation.get("target") == "client" and delegation.get("id"):
                    asyncio.create_task(answer_delegation(delegation["id"]))
            elif t == "error":
                err = ev.get("error") or {}
                print(f"[GPT-Live] Error: {err}")
                await to_client({"type": "error", "error": err})
            elif t == "session.closed":
                print(f"[GPT-Live] Session closed ({ev.get('reason')}), usage: {ev.get('usage')}")
                break

    async def client_to_live():
        while True:
            msg = json.loads(await client_ws.receive_text())
            t = msg.get("type", "")
            if t == "input_audio_buffer.append" and msg.get("audio") and started.is_set():
                await live_ws.send(json.dumps({"type": "session.input_audio.append", "audio": msg["audio"]}))
            elif t == "client.hangup":
                break
            # Realtime-only commands (session.update, response.create, response.cancel, ...) don't apply to GPT-Live.

    watcher = asyncio.create_task(turn_watcher())
    tasks = [asyncio.create_task(live_to_client()), asyncio.create_task(client_to_live())]
    try:
        start_waiter = asyncio.create_task(started.wait())
        await asyncio.wait([start_waiter, tasks[0]], timeout=20, return_when=asyncio.FIRST_COMPLETED)
        if not started.is_set():
            start_waiter.cancel()
            raise asyncio.TimeoutError
        for audio in pre_audio:
            await live_ws.send(json.dumps({"type": "session.input_audio.append", "audio": audio}))
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.TimeoutError:
        await to_client({"type": "error", "error": {"code": "live_start_timeout", "message": "GPT-Live session did not start."}})
    except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
        pass
    except Exception as e:
        print(f"[GPT-Live] Session error: {e}")
    finally:
        state["closed"] = True
        watcher.cancel()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await live_ws.send(json.dumps({"type": "session.close"}))
            await asyncio.wait_for(_drain_until_closed(live_ws), timeout=5)
        except Exception:
            pass
        try:
            await live_ws.close()
        except Exception:
            pass


async def _drain_until_closed(live_ws):
    async for raw in live_ws:
        if json.loads(raw).get("type") == "session.closed":
            return
