# Jarvis split deployment: WSL core + Windows desktop

Jarvis is split into two processes that talk over **one shared port**
(WebSocket, default `8765`):

| Process | Location | Responsibilities |
|---|---|---|
| **Core** | WSL: `core/` (deploy to e.g. `~/workspace/Jarvis/core`) | LLM (vLLM) + memory (build prompt, generate, store) |
| **Desktop** | Windows: repo root (`src/`) | Mic capture, Whisper STT, Edge-TTS, audio playback, hotkeys, tray UI |

```
Windows desktop (src/main.py)                    WSL core (core/server.py)
  mic -> AudioListener -> WhisperSTT                  WebSocket :8765 (bind 0.0.0.0)
        |  wake word / continuous / interrupt           |
        |  text  ──────────────  ws://127.0.0.1:8765 ──▶ build_enhanced_prompt
        |                                                OpenAIClient.generate_stream
        |  chunk (per sentence) ◀──────────────────────  (streaming, sentence split)
        |  done                                          process_interaction (memory)
  per-sentence TTS -> pygame playback                    └─ vLLM OpenAI API :8000 (Qwen3-8B)
```

The reply is **streamed sentence-by-sentence**: the core emits one `chunk`
message per sentence as the LLM produces it, and the desktop synthesizes +
plays each sentence while the next is still being generated (so it starts
speaking on the first sentence instead of waiting for the whole reply).

## Components

- **vLLM** serves Qwen3-8B over an OpenAI-compatible API at `127.0.0.1:8000`.
  Qwen3 "thinking" is disabled per-request (`enable_thinking: false`) — it is
  pure latency overhead for a voice assistant.
- **Core** (`core/server.py` + `core/chat_service.py`) calls vLLM, segments the
  token stream into sentences, and streams them to the desktop; it also owns
  the memory subsystem (`core/memory/`, data in `core/memory_data/`).
- **Desktop** (`src/main.py` + `src/core_client.py`) does everything hardware-
  bound and runs the per-sentence synth/playback pipeline.

## Networking note (important)
- The core binds **`0.0.0.0`** (`core/config.yaml` -> `core.host`).
- Windows connects to **`127.0.0.1`** (`config.yaml` -> `core.host`), **not**
  `localhost` — `localhost` may resolve to IPv6 `::1`, which WSL2's
  localhost-forwarding does not bind, causing connection hangs.

## One-time setup

### WSL: vLLM
```bash
cd ~/workspace/Jarvis
python3 -m venv vllm-venv
./vllm-venv/bin/pip install vllm
# Triton JIT-compiles a small CUDA helper at startup and needs Python headers:
sudo apt-get install -y python3-dev build-essential
```
> vLLM disables pinned host memory under WSL by default, which makes the V1
> engine fail with `RuntimeError: UVA is not available`. `core/serve_vllm.py`
> re-enables pinned memory in-process before the (forked) engine starts —
> launch vLLM through it, not via `vllm serve` directly.

### WSL: core
```bash
cd ~/workspace/Jarvis/core
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### Windows: desktop (conda env `jarvis`, python 3.8)
```powershell
conda activate jarvis
pip install websockets==12.0
```

## Running (start vLLM, then core, then desktop)

1. **WSL** — vLLM server (downloads Qwen3-8B on first run, ~16GB):
   ```bash
   ~/workspace/Jarvis/vllm-venv/bin/python ~/workspace/Jarvis/core/serve_vllm.py
   # -> "Application startup complete" / Uvicorn on :8000
   ```
2. **WSL** — core:
   ```bash
   ~/workspace/Jarvis/core/run.sh
   # -> "Jarvis core listening on ws://0.0.0.0:8765"
   ```
3. **Windows** — desktop app:
   ```
   bat\jarvis_window.bat        (or: python src/main.py)
   ```

If the core/vLLM are started after the desktop, the desktop reconnects on the
first utterance.

## Smoke tests
- Core only (non-streaming + stats): `core/.venv/bin/python` a small WS client.
- Windows -> WSL end to end: `python scripts/test_core_client.py`
  (collects the streamed reply and prints memory stats from the core).

## Performance notes
- On an RTX 4090, Qwen3-8B in vLLM is far faster than the voice pipeline needs;
  perceived latency is dominated by streaming + per-sentence TTS, which this
  design already overlaps.
- For maximum throughput, install `python3-dev build-essential` (above) and run
  vLLM **without** `--enforce-eager` so CUDA graphs are enabled. `serve_vllm.py`
  currently passes `--enforce-eager` as a safe default for hosts missing the
  dev headers; remove it once headers are installed.

## Memory data
Lives only on the core side: `core/memory_data/` (git-ignored). The desktop no
longer reads or writes memory. Do **not** run two core instances against the
same `memory_data/` — they race on the JSON/`vectors.npy` files.
