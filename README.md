# Jarvis Voice Assistant
<p align="center">
  <img src="./pics/ironman.jpg" width="256" height="256">
</p>

  
[English](#english) | [中文](#chinese)
<p align="center">


<a name="english"></a>

## English Documentation

Jarvis is a local voice assistant featuring real-time voice activation, speech recognition, local LLM processing, and voice synthesis. It runs as **two cooperating processes** — a GPU-accelerated **core in WSL** (LLM + memory) and a **desktop process on Windows** (microphone, STT, TTS, playback, UI) — that talk over a single local port.

### ⚡ Architecture & Performance

Jarvis now uses a split, GPU-accelerated design built for low latency:

```
Windows desktop (src/)                         WSL core (core/)
  mic → Whisper STT                              WebSocket :8765
       │  text  ─────────  ws://127.0.0.1:8765 ──▶ memory-enhanced prompt
       │  reply (per sentence) ◀──── streaming ──  vLLM (Qwen3-8B, OpenAI API)
  per-sentence TTS → playback                     └─ memory store
```

- **vLLM replaces Ollama.** LLM inference now runs on [vLLM](https://github.com/vllm-project/vllm) (serving **Qwen3‑8B** over an OpenAI-compatible API) instead of Ollama. vLLM's PagedAttention, continuous batching and prefix caching deliver dramatically higher throughput and lower time‑to‑first‑token on NVIDIA GPUs than Ollama/llama.cpp. The model is loaded once and stays resident on the GPU.
- **Runs in WSL2 for full GPU acceleration.** The core runs under WSL2 (Ubuntu) so it can use the Linux CUDA stack that vLLM targets — getting full RTX-class GPU acceleration while the desktop app stays native Windows. The two halves share a single WebSocket port; WSL2 localhost-forwarding makes the WSL service reachable from Windows at `127.0.0.1`.
- **Streaming + per-sentence playback.** The reply is streamed sentence‑by‑sentence: the desktop synthesizes and plays each sentence *while the next is still being generated*, so Jarvis starts speaking on the first sentence instead of waiting for the full response.
- **Low-latency model config.** Qwen3's "thinking" mode is disabled per-request (pure latency overhead for voice), and pinned host memory is re-enabled under WSL so vLLM's fast path works.

> Full setup and tuning details are in **[`DEPLOYMENT_WSL_SPLIT.md`](./DEPLOYMENT_WSL_SPLIT.md)**.

### Key Features

- Real-time voice activation with WebRTC VAD
- Offline speech recognition powered by Whisper.cpp
- **GPU-accelerated local LLM via vLLM (Qwen3-8B) running in a WSL core**
- **Streaming responses with sentence-by-sentence synthesis for low latency**
- **Interruptible speech — talk over Jarvis to cut in**
- **Long-term memory (conversation/profile/facts) handled in the core**
- Multiple voice synthesis options with Edge TTS
- System tray integration and global hotkeys
- Customizable voice, speech rate and volume; settings UI

### System Requirements

- Windows 10/11 with **WSL2 (Ubuntu)** for the core
- Python 3.8+ on Windows (conda env `jarvis`); Python 3.10+ inside WSL
- At least 6-core CPU, 16GB RAM
- **NVIDIA GPU recommended — ~16GB VRAM for Qwen3-8B in FP16** (use an AWQ/quantized or smaller model for less VRAM)
- Recent NVIDIA driver with WSL CUDA support

### Detailed Installation Guide

The desktop (Windows) and core (WSL) are installed separately. See **[`DEPLOYMENT_WSL_SPLIT.md`](./DEPLOYMENT_WSL_SPLIT.md)** for the complete, authoritative steps; a summary follows.

#### 1. Windows desktop environment

1. Install Python 3.8+ and create the conda env:
   ```cmd
   conda create -n jarvis python=3.8
   conda activate jarvis
   pip install -r requirements.txt
   ```
   Note: if PyAudio fails to install, use a wheel:
   ```cmd
   pip install pipwin
   pipwin install pyaudio
   ```

#### 2. Installing Whisper (STT, runs on Windows)

The project uses Whisper.cpp for speech recognition, which requires model files and binaries.

##### Whisper Model Files
1. The project includes `tiny.en.bin` and `base.en.bin` models in the `models` directory
2. If you need to download them manually:
   - Visit [ggerganov/whisper.cpp/releases](https://github.com/ggerganov/whisper.cpp/releases)
   - Download either `ggml-tiny.en.bin` or `ggml-base.en.bin`
   - Rename the files to `tiny.en.bin` or `base.en.bin`
   - Place them in the `models` directory

##### Whisper.cpp Binaries
1. The project includes necessary binaries in the `whisper_bin` directory
2. If you need to install them manually:
   - Visit [ggerganov/whisper.cpp/releases](https://github.com/ggerganov/whisper.cpp/releases)
   - Download the Windows release zip file
   - Extract the contents
   - Copy `whisper.dll`, `main.exe` and related files to the `whisper_bin` directory

#### 3. Installing the WSL core (vLLM, replaces Ollama)

The LLM + memory core runs in WSL2. From a WSL (Ubuntu) shell:

```bash
cd ~/workspace/Jarvis          # the repo's core/ directory deployed here

# vLLM server (its own venv); Triton needs Python headers to JIT-compile
python3 -m venv vllm-venv
./vllm-venv/bin/pip install vllm
sudo apt-get install -y python3-dev build-essential

# core service (LLM client + memory)
cd core
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

> vLLM disables pinned host memory under WSL by default, which crashes the V1
> engine (`UVA is not available`). Launch vLLM via `core/serve_vllm.py` (not
> `vllm serve`), which re-enables it in-process. Qwen3-8B (~16GB) downloads on
> first run.

#### 4. Configure the Application

1. Review and modify `config.yaml` (desktop) and `core/config.yaml` (core):
   - Audio / voice / hotkey settings (desktop)
   - `core:` host/port for the WebSocket link
   - `llm:` vLLM endpoint, model and parameters (core)

### Usage Instructions

Start the core first (in WSL), then the desktop (on Windows):

1. **WSL — vLLM server:** `~/workspace/Jarvis/vllm-venv/bin/python ~/workspace/Jarvis/core/serve_vllm.py`
2. **WSL — core service:** `~/workspace/Jarvis/core/run.sh` → `listening on ws://0.0.0.0:8765`
3. **Windows — desktop app:**
   - Double click `start_jarvis.bat` in the root directory
   - Or run `bat/jarvis_window.bat` directly

The system initializes and shows in the system tray. If the core/vLLM are started after the desktop, the desktop reconnects on the first utterance.

4. System tray options:
   - Status: Shows current assistant status (incl. model + memory stats from the core)
   - Settings: Opens the settings window
   - Exit: Closes the application

5. Default global hotkeys:
   - Ctrl + F1: Toggle listening mode
   - Ctrl + F2: Switch between available voices
   - Ctrl + F3: Adjust speech rate

6. Using the assistant:
   - The desktop monitors your microphone and transcribes speech with Whisper
   - The text is sent to the WSL core, which builds a memory-enhanced prompt and streams the reply from vLLM
   - The desktop synthesizes and plays each sentence as it arrives — and you can interrupt by speaking

### Troubleshooting

- **No audio input detected**: Check your microphone settings and ensure the correct device is selected
- **Speech recognition issues**: Try using a larger model like `base.en.bin` for better accuracy
- **LLM not responding**: Ensure the vLLM server is up (`:8000`) and the core is running (`ws://127.0.0.1:8765`); from Windows use `127.0.0.1`, not `localhost`
- **vLLM fails to start in WSL**: Install `python3-dev build-essential`, and launch via `core/serve_vllm.py`; see `DEPLOYMENT_WSL_SPLIT.md`
- **TTS not working**: Check your internet connection as Edge TTS requires connectivity

### Future Plans

#### 1. MCP (Model Context Protocol)
- Allow Jarvis to operate desktop files and basic software
- Implement email viewing/replying functionality
- Schedule planning and management
- Basic agent capabilities, such as:
  * File management and organization
  * Application control
  * Calendar and reminder management
  * Email processing and response
  * Simple automation tasks

#### 2. Performance & model options
- Streaming token-level interruption (cancel generation mid-sentence)
- AWQ/quantized and smaller models for lower-VRAM GPUs
- Local TTS option (e.g. Piper/Kokoro) to remove the Edge TTS network round-trip

> Note: interruptible speech, long-term memory, and streaming responses are now implemented (see Architecture & Performance).

## License

MIT License 

---
<a name="chinese"></a>

## 中文文档

Jarvis 是一个本地语音助手，具有实时语音激活、语音识别、本地大模型处理与语音合成功能。它以**两个协作进程**运行——一个 GPU 加速的 **WSL 核心**（LLM + 记忆）和一个 **Windows 桌面进程**（麦克风、STT、TTS、播放、界面）——两者通过单个本地端口通信。

### ⚡ 架构与性能

Jarvis 现采用拆分式、GPU 加速的设计，专为低延迟优化：

```
Windows 桌面 (src/)                            WSL 核心 (core/)
  麦克风 → Whisper STT                           WebSocket :8765
        │  文本  ─────────  ws://127.0.0.1:8765 ──▶ 带记忆的提示词
        │  回复（逐句）◀──────── 流式 ──────────  vLLM (Qwen3-8B, OpenAI API)
  逐句 TTS → 播放                                 └─ 写入记忆
```

- **vLLM 替代 Ollama。** LLM 推理改由 [vLLM](https://github.com/vllm-project/vllm)（以 OpenAI 兼容接口提供 **Qwen3‑8B**）运行，不再使用 Ollama。vLLM 的 PagedAttention、连续批处理与前缀缓存，在 NVIDIA GPU 上相比 Ollama/llama.cpp 带来显著更高的吞吐与更低的首字延迟（TTFT），模型一次加载后常驻显存。
- **运行于 WSL2，获得完整 GPU 加速。** 核心运行在 WSL2（Ubuntu）下，从而使用 vLLM 所需的 Linux CUDA 栈，获得完整的 RTX 级 GPU 加速；桌面端则保持原生 Windows。两端共享同一个 WebSocket 端口，借助 WSL2 的 localhost 转发，Windows 通过 `127.0.0.1` 即可访问 WSL 服务。
- **流式 + 逐句播放。** 回复按句子流式返回：桌面端在生成下一句的同时合成并播放当前句，因此 Jarvis 在第一句就能开口，而不必等待整段回复。
- **低延迟模型配置。** 按请求关闭 Qwen3 的"思考"模式（对语音纯属延迟开销），并在 WSL 下重新启用锁页内存以走通 vLLM 快路径。

> 完整的安装与调优细节见 **[`DEPLOYMENT_WSL_SPLIT.md`](./DEPLOYMENT_WSL_SPLIT.md)**。

### 主要特性

- 基于 WebRTC VAD 的实时语音激活
- 基于 Whisper.cpp 的离线语音识别
- **通过运行于 WSL 核心的 vLLM（Qwen3-8B）进行 GPU 加速的本地大模型推理**
- **流式回复 + 逐句合成，低延迟**
- **可打断——说话即可打断 Jarvis**
- **长期记忆（对话/画像/事实）由核心处理**
- 支持 Edge TTS 的多种语音合成选项
- 系统托盘集成与全局热键
- 可自定义语音、语速、音量；设置界面

### 系统要求

- Windows 10/11，并安装 **WSL2（Ubuntu）** 用于运行核心
- Windows 上 Python 3.8+（conda 环境 `jarvis`）；WSL 内 Python 3.10+
- 至少 6 核 CPU、16GB 内存
- **推荐 NVIDIA GPU——Qwen3-8B FP16 约需 16GB 显存**（显存较小可用 AWQ/量化或更小模型）
- 较新的、支持 WSL CUDA 的 NVIDIA 驱动

### 详细安装指南

桌面端（Windows）与核心（WSL）分别安装。完整且权威的步骤见 **[`DEPLOYMENT_WSL_SPLIT.md`](./DEPLOYMENT_WSL_SPLIT.md)**，以下为概要。

#### 1. Windows 桌面环境

1. 安装 Python 3.8+ 并创建 conda 环境：
   ```cmd
   conda create -n jarvis python=3.8
   conda activate jarvis
   pip install -r requirements.txt
   ```
   注意：若 PyAudio 安装失败，可用 wheel 安装：
   ```cmd
   pip install pipwin
   pipwin install pyaudio
   ```

#### 2. 安装 Whisper（STT，运行于 Windows）

项目使用 Whisper.cpp 进行语音识别，需要模型文件和二进制文件。

##### Whisper 模型文件
1. 项目在 `models` 目录中包含 `tiny.en.bin` 和 `base.en.bin` 模型
2. 如需手动下载：
   - 访问 [ggerganov/whisper.cpp/releases](https://github.com/ggerganov/whisper.cpp/releases)
   - 下载 `ggml-tiny.en.bin` 或 `ggml-base.en.bin`
   - 将文件重命名为 `tiny.en.bin` 或 `base.en.bin`
   - 放置在 `models` 目录中

##### Whisper.cpp 二进制文件
1. 项目在 `whisper_bin` 目录中包含必要的二进制文件
2. 如需手动安装：
   - 访问 [ggerganov/whisper.cpp/releases](https://github.com/ggerganov/whisper.cpp/releases)
   - 下载 Windows 版本的 zip 文件
   - 解压内容
   - 将 `whisper.dll`、`main.exe` 等相关文件复制到 `whisper_bin` 目录

#### 3. 安装 WSL 核心（vLLM，替代 Ollama）

LLM + 记忆核心运行在 WSL2 中。在 WSL（Ubuntu）终端：

```bash
cd ~/workspace/Jarvis          # 将仓库的 core/ 部署到此处

# vLLM 服务（独立 venv）；Triton 需要 Python 头文件来即时编译
python3 -m venv vllm-venv
./vllm-venv/bin/pip install vllm
sudo apt-get install -y python3-dev build-essential

# 核心服务（LLM 客户端 + 记忆）
cd core
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

> vLLM 在 WSL 下默认禁用锁页内存，会导致 V1 引擎崩溃（`UVA is not available`）。
> 请通过 `core/serve_vllm.py`（而非 `vllm serve`）启动，它会在进程内重新启用。
> Qwen3-8B（约 16GB）会在首次运行时下载。

#### 4. 配置应用程序

1. 查看并修改 `config.yaml`（桌面）与 `core/config.yaml`（核心）：
   - 音频 / 语音 / 热键设置（桌面）
   - `core:` WebSocket 链接的主机/端口
   - `llm:` vLLM 端点、模型与参数（核心）

### 使用说明

先启动核心（在 WSL），再启动桌面（在 Windows）：

1. **WSL — vLLM 服务：** `~/workspace/Jarvis/vllm-venv/bin/python ~/workspace/Jarvis/core/serve_vllm.py`
2. **WSL — 核心服务：** `~/workspace/Jarvis/core/run.sh` → `listening on ws://0.0.0.0:8765`
3. **Windows — 桌面程序：**
   - 双击根目录下的 `start_jarvis.bat`
   - 或直接运行 `bat/jarvis_window.bat`

系统将初始化并显示在系统托盘中。若核心/vLLM 在桌面之后启动，桌面会在第一次说话时自动重连。

4. 系统托盘选项：
   - 状态：显示当前助手状态（含来自核心的模型与记忆统计）
   - 设置：打开设置窗口
   - 退出：关闭应用程序

5. 默认全局热键：
   - Ctrl + F1：切换监听模式
   - Ctrl + F2：切换可用语音
   - Ctrl + F3：调整语速

6. 使用助手：
   - 桌面端监控麦克风并用 Whisper 转录语音
   - 文本发送到 WSL 核心，核心构建带记忆的提示词并从 vLLM 流式获取回复
   - 桌面端逐句合成并播放——你也可以通过说话来打断

### 故障排除

- **未检测到音频输入**：检查麦克风设置并确保选择了正确的设备
- **语音识别问题**：尝试使用更大的模型如 `base.en.bin` 以提高准确性
- **LLM 无响应**：确认 vLLM 服务已启动（`:8000`）且核心在运行（`ws://127.0.0.1:8765`）；Windows 端请用 `127.0.0.1` 而非 `localhost`
- **vLLM 在 WSL 启动失败**：安装 `python3-dev build-essential`，并通过 `core/serve_vllm.py` 启动；详见 `DEPLOYMENT_WSL_SPLIT.md`
- **TTS 不工作**：检查网络连接，因为 Edge TTS 需要联网

### 未来计划

#### 1. MCP（Model Context Protocol）
- 允许 Jarvis 操作桌面文件和基本软件
- 实现查看/回复邮件功能
- 制定和管理日程计划
- 基本的 agent 功能，如：
  * 文件管理和组织
  * 应用程序控制
  * 日历和提醒管理
  * 邮件处理和回复
  * 简单的自动化任务

#### 2. 性能与模型选项
- token 级流式打断（在句子中途取消生成）
- AWQ/量化及更小模型，适配低显存 GPU
- 本地 TTS 选项（如 Piper/Kokoro），去掉 Edge TTS 的网络往返

> 注：可打断语音、长期记忆、流式回复均已实现（见"架构与性能"）。

---
