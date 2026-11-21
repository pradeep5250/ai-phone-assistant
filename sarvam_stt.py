"""
Sarvam Saaras v4 realtime speech-to-text client.

Browser audio arrives as 24 kHz PCM16 (the OpenAI Realtime format); Saaras accepts
16 kHz or 8 kHz only, so audio is resampled 3:2 before it is forwarded.
"""

import asyncio
import base64
import json
import os
import socket
from typing import Awaitable, Callable, Optional
from urllib.parse import urlencode

import numpy as np
import websockets

SARVAM_STT_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"

DEFAULT_KEYTERMS = [
    "MintX",
    "MintX Solutions",
    "Website Development",
    "Software Development",
    "Full Stack Development",
    "UI UX",
    "SEO",
    "AI Solutions",
    "AI Voice Assistant",
    "e-commerce",
    "mobile application",
    "React",
    "Python",
    "PHP",
    "MySQL",
    "WordPress",
    "WhatsApp",
]

EventHandler = Callable[[dict], Awaitable[None]]


class Resampler24kTo16k:
    """Streaming 24 kHz -> 16 kHz PCM16 resampler (linear interpolation, 3 in -> 2 out)."""

    def __init__(self):
        self._carry = np.zeros(0, dtype=np.float32)

    def process(self, pcm24: bytes) -> bytes:
        if len(pcm24) % 2:
            pcm24 = pcm24[:-1]
        samples = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32)
        buf = np.concatenate([self._carry, samples]) if self._carry.size else samples
        usable = (buf.size // 3) * 3
        self._carry = buf[usable:]
        if usable == 0:
            return b""
        groups = buf[:usable].reshape(-1, 3)
        out = np.empty((groups.shape[0], 2), dtype=np.float32)
        out[:, 0] = groups[:, 0]
        out[:, 1] = (groups[:, 1] + groups[:, 2]) * 0.5
        return np.clip(out.reshape(-1), -32768, 32767).astype(np.int16).tobytes()


class SarvamRealtimeSTT:
    def __init__(
        self,
        on_event: EventHandler,
        api_key: Optional[str] = None,
        model: str = "saaras:v4",
        language_code: str = "auto",
        mode: Optional[str] = None,
        keyterms: Optional[list] = None,
    ):
        self.on_event = on_event
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.model = model
        self.language_code = language_code
        self.mode = mode or os.getenv("SARVAM_STT_MODE", "codemix")
        self.keyterms = keyterms if keyterms is not None else DEFAULT_KEYTERMS
        self.ws = None
        self.closed = False
        self._resampler = Resampler24kTo16k()

    def _url(self) -> str:
        params = {
            "model": self.model,
            "language_code": self.language_code,
            "stream_type": os.getenv("SARVAM_STT_STREAM_TYPE", "fast"),
            "mode": self.mode,
            "endpointing": "vad",
            "encoding": "linear16",
            "sample_rate": "16000",
            "threshold": os.getenv("SARVAM_STT_VAD_THRESHOLD", "0.35"),
            "silence_duration_ms": os.getenv("SARVAM_STT_SILENCE_MS", "500"),
            "prefix_padding_ms": "300",
            "min_speech_duration_ms": "250",
        }
        if self.keyterms and self.model == "saaras:v4":
            params["keyterms"] = json.dumps(self.keyterms[:50])
        return f"{SARVAM_STT_URL}?{urlencode(params)}"

    async def connect(self, timeout: float = 6.0, attempts: int = 3) -> dict:
        """Open the socket and wait for session.begin, retrying transient failures. Raises on failure."""
        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY is not configured")
        last_err = None
        for attempt in range(1, attempts + 1):
            try:
                self.ws = await asyncio.wait_for(
                    websockets.connect(
                        self._url(),
                        additional_headers={"api-subscription-key": self.api_key},
                        max_size=None,
                        ping_interval=20,
                        proxy=None,
                        family=socket.AF_INET,
                    ),
                    timeout=timeout,
                )
                first = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=timeout))
                if first.get("event") == "error":
                    # Rejections such as exhausted credits or a bad key won't succeed on retry
                    await self.ws.close()
                    self.ws = None
                    raise PermissionError(first.get("message") or str(first))
                return first
            except PermissionError:
                raise
            except Exception as e:
                last_err = e
                print(f"[!] Saaras connect attempt {attempt}/{attempts} failed: {type(e).__name__}: {e}")
                if self.ws:
                    try:
                        await self.ws.close()
                    except Exception:
                        pass
                    self.ws = None
                await asyncio.sleep(0.4 * attempt)
        raise last_err

    async def send_audio_24k(self, b64_pcm24: str):
        if self.closed or not self.ws:
            return
        pcm16k = self._resampler.process(base64.b64decode(b64_pcm24))
        if pcm16k:
            await self.ws.send(json.dumps({
                "event": "audio_input",
                "audio": base64.b64encode(pcm16k).decode("ascii"),
            }))

    async def run(self):
        """Read events until the socket closes; each event is passed to on_event."""
        keepalive = asyncio.create_task(self._keepalive())
        try:
            async for raw in self.ws:
                try:
                    event = json.loads(raw)
                except Exception:
                    continue
                await self.on_event(event)
        finally:
            keepalive.cancel()
            self.closed = True

    async def _keepalive(self):
        try:
            while not self.closed:
                await asyncio.sleep(15)
                await self.ws.send(json.dumps({"event": "ping"}))
        except Exception:
            pass

    async def close(self):
        if self.closed or not self.ws:
            return
        self.closed = True
        try:
            await self.ws.send(json.dumps({"event": "end"}))
        except Exception:
            pass
        try:
            await self.ws.close()
        except Exception:
            pass
