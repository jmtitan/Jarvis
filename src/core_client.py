"""
CoreClient: Windows-side WebSocket client that talks to the Jarvis core
running in WSL (~/workspace/Jarvis/core).

The desktop process keeps all hardware-bound work (microphone capture, Whisper
STT, Edge-TTS synthesis, audio playback, hotkeys, tray UI). The reply is
*streamed* back sentence-by-sentence so the desktop can synthesize and play
each sentence while the next is still being generated.

Protocol (JSON text frames over a single shared port):
    Win  -> Core : {"type":"chat",      "id": "<uuid>", "text": "..."}
    Core -> Win  : {"type":"chunk",     "id": "<uuid>", "text": "<one sentence>"}  (repeated)
    Core -> Win  : {"type":"done",      "id": "<uuid>", "count": N[, "interrupted": true]}
    Win  -> Core : {"type":"interrupt", "id": "<uuid>"}
    Win  -> Core : {"type":"stats",     "id": "<uuid>"}
    Core -> Win  : {"type":"stats",     "id": "<uuid>", "data": {...}}
    Win  -> Core : {"type":"ping"}      Core -> {"type":"pong"}
    Core -> Win  : {"type":"error",     "id": "<uuid>", "msg": "..."}
"""

import asyncio
import json
import uuid

import websockets


class CoreClient:
    def __init__(self, config: dict):
        core_cfg = config.get("core", {})
        host = core_cfg.get("host", "127.0.0.1")
        # The shared config.yaml may carry the core *bind* address (0.0.0.0).
        # From Windows we connect to it through WSL2 localhost-forwarding.
        # Use 127.0.0.1 (IPv4) rather than "localhost" to avoid an IPv6 (::1)
        # resolution that WSL2 forwarding does not bind.
        if host in ("0.0.0.0", "localhost", "", None):
            host = "127.0.0.1"
        self.port = int(core_cfg.get("port", 8765))
        self.uri = f"ws://{host}:{self.port}"

        self.ws = None
        self._connected = False
        self._connect_lock = asyncio.Lock()
        self._recv_task = None
        self._pending = {}            # request id -> Future       (stats)
        self._streams = {}            # request id -> asyncio.Queue (chat stream)
        self._last_chat_id = None     # for targeted interrupts

    async def connect(self):
        """Ensure a live connection. Safe to call repeatedly."""
        if self._connected and self.ws is not None:
            return
        async with self._connect_lock:
            if self._connected and self.ws is not None:
                return
            self.ws = await websockets.connect(
                self.uri, ping_interval=20, max_size=None
            )
            self._connected = True
            self._recv_task = asyncio.create_task(self._recv_loop())
            print(f"Connected to Jarvis core at {self.uri}")

    async def _recv_loop(self):
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                rid = msg.get("id")
                if rid and rid in self._streams:
                    # chat stream messages: chunk / done / error
                    self._streams[rid].put_nowait(msg)
                elif rid and rid in self._pending:
                    fut = self._pending.pop(rid)
                    if not fut.done():
                        fut.set_result(msg)
                # "pong" and anything else: ignore
        except Exception as e:
            print(f"Core connection lost: {e}")
        finally:
            self._connected = False
            self.ws = None
            # Unblock any in-flight requests so callers don't hang
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("core connection lost"))
            self._pending.clear()
            for q in self._streams.values():
                q.put_nowait({"type": "error", "msg": "core connection lost"})

    async def chat_stream(self, text: str, timeout: float = 120.0):
        """Async generator: yields reply sentences as they arrive from the core.

        Stops on `done`/`error`/timeout/connection-loss. Stores the request id
        in `_last_chat_id` so a concurrent interrupt() can target it.
        """
        try:
            await self.connect()
        except Exception as e:
            print(f"Could not connect to Jarvis core at {self.uri}: {e}")
            return

        rid = uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue()
        self._streams[rid] = q
        self._last_chat_id = rid
        try:
            await self.ws.send(json.dumps({"type": "chat", "id": rid, "text": text}))
        except Exception as e:
            self._streams.pop(rid, None)
            print(f"Failed to send chat to core: {e}")
            return

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    print(f"Core stream timed out after {timeout}s")
                    return
                mtype = msg.get("type")
                if mtype == "chunk":
                    sentence = msg.get("text", "")
                    if sentence:
                        yield sentence
                elif mtype == "done":
                    return
                elif mtype == "error":
                    print(f"Core error: {msg.get('msg')}")
                    return
        finally:
            self._streams.pop(rid, None)

    async def chat(self, text: str, timeout: float = 120.0) -> str:
        """Convenience: collect the streamed sentences into one reply string."""
        parts = []
        async for sentence in self.chat_stream(text, timeout=timeout):
            parts.append(sentence)
        return " ".join(parts).strip()

    async def _request(self, payload: dict, timeout: float):
        """Send a request carrying an id and await the matching response."""
        try:
            await self.connect()
        except Exception as e:
            print(f"Could not connect to Jarvis core at {self.uri}: {e}")
            return None
        rid = uuid.uuid4().hex
        payload = {**payload, "id": rid}
        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        try:
            await self.ws.send(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            self._pending.pop(rid, None)
            print(f"Failed to send to core: {e}")
            return None
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            print(f"Core request timed out after {timeout}s")
            return None
        except ConnectionError as e:
            print(f"Core request aborted: {e}")
            return None

    async def interrupt(self):
        """Tell the core to cancel the most recent chat (best-effort)."""
        if self._connected and self.ws is not None and self._last_chat_id:
            try:
                await self.ws.send(json.dumps(
                    {"type": "interrupt", "id": self._last_chat_id}
                ))
            except Exception:
                pass

    async def get_stats(self, timeout: float = 5.0):
        """Fetch model + memory statistics from the core."""
        msg = await self._request({"type": "stats"}, timeout=timeout)
        if msg and msg.get("type") == "stats":
            return msg.get("data", {})
        return {}

    async def close(self):
        if self._recv_task:
            self._recv_task.cancel()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        self._connected = False
        self.ws = None
        print("CoreClient closed.")
