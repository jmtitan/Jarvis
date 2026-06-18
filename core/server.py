"""
Jarvis core WebSocket server (runs in WSL).

Single shared port. The Windows desktop client connects over WebSocket and
exchanges JSON text frames:

  Win  -> Core : {"type":"chat",      "id": "<uuid>", "text": "..."}
  Core -> Win  : {"type":"chunk",     "id": "<uuid>", "text": "<one sentence>"}  (repeated)
  Core -> Win  : {"type":"done",      "id": "<uuid>", "count": N[, "interrupted": true]}
  Win  -> Core : {"type":"interrupt", "id": "<uuid>"}
  Win  -> Core : {"type":"stats"}      Core -> {"type":"stats", "data": {...}}
  Win  -> Core : {"type":"ping"}       Core -> {"type":"pong"}
  Core -> Win  : {"type":"error",     "id": "<uuid>", "msg": "..."}

The reply is streamed sentence-by-sentence as `chunk` messages so the desktop
side can synthesize and play each sentence while the next is generated.

Bind to 0.0.0.0 so WSL2 localhost-forwarding lets Windows reach it at
ws://localhost:<port>.
"""

import asyncio
import json
import os
import signal
import sys

import yaml
import websockets

from chat_service import ChatService


def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class JarvisCoreServer:
    def __init__(self, config: dict):
        self.config = config
        self.svc = ChatService(config)
        core_cfg = config.get("core", {})
        self.host = core_cfg.get("host", "0.0.0.0")
        self.port = int(core_cfg.get("port", 8765))
        # Track in-flight chat tasks per request id so interrupt can cancel them
        self._tasks: dict[str, asyncio.Task] = {}

    async def _send(self, ws, obj: dict):
        try:
            await ws.send(json.dumps(obj, ensure_ascii=False))
        except Exception as e:
            print(f"send failed: {e}")

    async def _run_chat(self, ws, req_id: str, text: str):
        """Stream the reply sentence-by-sentence: a series of `chunk`
        messages followed by a single `done` message."""
        n = 0
        try:
            async for sentence in self.svc.handle_stream(text):
                n += 1
                await self._send(ws, {"type": "chunk", "id": req_id, "text": sentence})
            await self._send(ws, {"type": "done", "id": req_id, "count": n})
        except asyncio.CancelledError:
            print(f"chat {req_id} cancelled (interrupted) after {n} sentences")
            # Tell the client the stream ended so it can stop waiting.
            await self._send(ws, {"type": "done", "id": req_id, "count": n, "interrupted": True})
            raise
        except Exception as e:
            print(f"chat {req_id} failed: {e}")
            await self._send(ws, {"type": "error", "id": req_id, "msg": str(e)})
        finally:
            self._tasks.pop(req_id, None)

    async def handler(self, ws):
        peer = getattr(ws, "remote_address", "?")
        print(f"client connected: {peer}")
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    await self._send(ws, {"type": "error", "msg": "invalid json"})
                    continue

                mtype = msg.get("type")
                if mtype == "chat":
                    req_id = msg.get("id", "")
                    text = msg.get("text", "")
                    print(f"chat[{req_id}]: {text!r}")
                    task = asyncio.create_task(self._run_chat(ws, req_id, text))
                    self._tasks[req_id] = task
                elif mtype == "interrupt":
                    req_id = msg.get("id", "")
                    task = self._tasks.get(req_id)
                    if task and not task.done():
                        task.cancel()
                    print(f"interrupt[{req_id}]")
                elif mtype == "stats":
                    await self._send(ws, {"type": "stats", "id": msg.get("id"), "data": self.svc.stats()})
                elif mtype == "ping":
                    await self._send(ws, {"type": "pong"})
                else:
                    await self._send(ws, {"type": "error", "msg": f"unknown type: {mtype}"})
        except websockets.ConnectionClosed:
            pass
        finally:
            print(f"client disconnected: {peer}")

    async def serve(self):
        stop = asyncio.Future()

        def _request_stop(*_):
            if not stop.done():
                stop.set_result(None)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except NotImplementedError:
                pass

        async with websockets.serve(self.handler, self.host, self.port, ping_interval=20):
            print(f"Jarvis core listening on ws://{self.host}:{self.port}")
            await stop

        print("shutting down core...")
        await self.svc.shutdown()
        print("core shutdown complete")


def main():
    # Always run with CWD = this file dir so relative paths (config.yaml,
    # prompts/, memory_data/) resolve correctly.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    config = load_config()
    server = JarvisCoreServer(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
