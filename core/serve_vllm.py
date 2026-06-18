#!/usr/bin/env python
"""
Launch vLLM with pinned host memory re-enabled under WSL.

vLLM's V1 engine requires UVA (which needs pinned host memory), but vLLM
hard-disables pinned memory when it detects WSL, citing an old NVIDIA
limitation. Pinned memory in fact works on this machine (modern WSL2 + recent
driver), so we re-enable it here, in-process, BEFORE the engine is launched.

The EngineCore subprocess is created with multiprocessing 'fork' (vLLM's
default when CUDA has not yet been initialized in this process), so it inherits
this monkeypatch. We therefore avoid initializing CUDA in this launcher.

This is a project-owned launcher (no vendored files edited, nothing auto-runs
on interpreter startup). Run it instead of `vllm serve`.
"""

import sys

# Patch the platform check before the engine starts. Importing this module does
# not initialize CUDA, so the engine is still forked (and inherits the patch).
import vllm.platforms.interface as _iface

_iface.Platform.is_pin_memory_available = classmethod(lambda cls: True)

from vllm.entrypoints.cli.main import main  # noqa: E402

DEFAULT_ARGS = [
    "serve", "Qwen/Qwen3-8B",
    "--host", "0.0.0.0", "--port", "8000",
    "--max-model-len", "16384",
    "--gpu-memory-utilization", "0.90",
    "--served-model-name", "Qwen3-8B",
    # --enforce-eager disables torch.compile / CUDA-graph capture, which on a
    # fresh WSL needs python3-dev (Python.h) to JIT-compile a Triton helper.
    # Eager mode avoids that dependency; on a 4090 an 8B model is still plenty
    # fast for voice. For max throughput: `sudo apt install python3-dev
    # build-essential`, then remove this flag.
    "--enforce-eager",
]

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += DEFAULT_ARGS
    sys.exit(main())
