# Jarvis Voice Assistant / Jarvis 语音助手
![ironman](./pics/ironman.jpg)

[English](#english) | [中文](#chinese)

<a name="chinese"></a>

## 中文文档

Jarvis是一个基于Windows的本地语音助手系统，具有实时语音激活、语音识别、本地语言模型处理和语音合成功能。它完全离线运行，在提供全面语音交互体验的同时确保隐私安全。

### 快速开始

开始使用Jarvis：

```bash
# 克隆仓库
git clone https://github.com/jmtitan/Jarvis.git
cd Jarvis

# 设置conda环境并安装依赖
conda create -n jarvis python=3.8
conda activate jarvis
pip install -r requirements.txt

# 下载所需模型文件（未包含在仓库中）
# 详见"安装Whisper"部分
```

### 主要特性

- 基于WebRTC VAD的实时语音激活
- 基于Whisper.cpp的离线语音识别
- 通过Ollama进行本地大语言模型处理
- 支持Edge TTS的多种语音合成选项
- 系统托盘集成，便于访问
- 支持全局热键快速控制
- 可自定义语音、语速和音量
- 设置界面便于配置

### 系统要求

- Windows 10/11
- Python 3.8+
- 至少6核CPU
- 16GB RAM
- 推荐：NVIDIA GPU（8GB+ VRAM）

### 详细安装指南

#### 1. 设置Python环境

1. 从[python.org](https://www.python.org/downloads/)安装Python 3.8或更新版本
2. 创建并激活conda环境：
   ```cmd
   conda create -n jarvis python=3.8
   conda activate jarvis
   ```
3. 安装所需依赖：
   ```cmd
   pip install -r requirements.txt
   ```
   
   注意：如果安装PyAudio时遇到问题，可能需要从wheel文件安装：
   ```cmd
   pip install pipwin
   pipwin install pyaudio
   ```

#### 2. 安装Whisper

项目使用Whisper.cpp进行语音识别，需要模型文件和二进制文件。

##### Whisper模型文件
1. 项目在`models`目录中包含`tiny.en.bin`和`base.en.bin`模型
2. 如需手动下载：
   - 访问[ggerganov/whisper.cpp/releases](https://github.com/ggerganov/whisper.cpp/releases)
   - 下载`ggml-tiny.en.bin`或`ggml-base.en.bin`
   - 将文件重命名为`tiny.en.bin`或`base.en.bin`
   - 放置在`models`目录中

##### Whisper.cpp二进制文件
1. 项目在`whisper_bin`目录中包含必要的二进制文件
2. 如需手动安装：
   - 访问[ggerganov/whisper.cpp/releases](https://github.com/ggerganov/whisper.cpp/releases)
   - 下载Windows版本的zip文件
   - 解压内容
   - 将`whisper.dll`、`main.exe`等相关文件复制到`whisper_bin`目录

#### 3. 安装Ollama（用于LLM支持）

1. 安装Ollama：
   ```cmd
   winget install Ollama
   ```
   或从[Ollama官网](https://ollama.com/download)下载

2. 安装后，下载兼容的模型：
   ```cmd
   ollama pull llama2
   ```
   （可以用其他模型替换llama2，如mistral或phi）

#### 4. 配置应用程序

1. 根据个人偏好查看和修改`config.yaml`：
   - 调整音频设置
   - 更改语音设置
   - 配置热键
   - 设置LLM参数

### 使用说明

1. 启动助手：
   - 双击根目录下的`start_jarvis.bat`
   - 或直接运行`bat/jarvis_window.bat`

2. 系统将初始化并显示在系统托盘中。

3. 系统托盘选项：
   - 状态：显示当前助手状态
   - 设置：打开设置窗口
   - 退出：关闭应用程序

4. 默认全局热键：
   - Ctrl + F1：切换监听模式
   - Ctrl + F2：切换可用语音
   - Ctrl + F3：调整语速

5. 使用助手：
   - 系统监控麦克风的语音输入
   - 检测到语音时，使用Whisper进行转录
   - 将转录发送到本地LLM进行处理
   - LLM的响应通过语音合成播放

### 故障排除

- **未检测到音频输入**：检查麦克风设置并确保选择了正确的设备
- **语音识别问题**：尝试使用更大的模型如`base.en.bin`以提高准确性
- **LLM无响应**：确保Ollama正在运行且已下载模型
- **TTS不工作**：检查网络连接，因为Edge TTS需要联网

### 未来计划

#### 1. 更拟人的对话体验
- 允许用户打断Jarvis的发言
- Jarvis只截取最近的用户音频作为对话主题
- 更自然的对话流程和语气
- 上下文感知和情绪理解

#### 2. Memory功能
- 用户可以为Jarvis打上自己的思想烙印
- 长期记忆存储和检索
- 个性化对话风格适应
- 用户偏好学习和记忆

#### 3. MCP（Master Control Program）
- 允许Jarvis操作桌面文件和基本软件
- 实现查看/回复邮件功能
- 制定和管理日程计划
- 基本的agent功能，如：
  * 文件管理和组织
  * 应用程序控制
  * 日历和提醒管理
  * 邮件处理和回复
  * 简单的自动化任务

---

<a name="english"></a>

## English Documentation

Jarvis is a Windows-based local voice assistant system featuring real-time voice activation, speech recognition, local language model processing, and voice synthesis. It's designed to operate completely offline, ensuring privacy while providing a comprehensive voice interaction experience.

### Quick Start

To get started with Jarvis:

```bash
# Clone the repository
git clone https://github.com/jmtitan/Jarvis.git
cd Jarvis

# Set up conda environment and install dependencies
conda create -n jarvis python=3.8
conda activate jarvis
pip install -r requirements.txt

# Download required model files (not included in repo)
# See "Installing Whisper" section for details
```

### Key Features

- Real-time voice activation with WebRTC VAD
- Offline speech recognition powered by Whisper.cpp
- Local large language model processing via Ollama
- Multiple voice synthesis options with Edge TTS
- System tray integration for easy access
- Global hotkey support for quick controls
- Customizable voice, speech rate and volume
- Settings UI for easy configuration

### System Requirements

- Windows 10/11
- Python 3.8+
- At least 6-core CPU
- 16GB RAM
- Recommended: NVIDIA GPU (8GB+ VRAM)

### Detailed Installation Guide

#### 1. Setting up Python Environment

1. Install Python 3.8 or newer from [python.org](https://www.python.org/downloads/)
2. Create and activate conda environment:
   ```cmd
   conda create -n jarvis python=3.8
   conda activate jarvis
   ```
3. Install the required dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
   
   Note: If you encounter issues installing PyAudio, you may need to install it from a wheel file:
   ```cmd
   pip install pipwin
   pipwin install pyaudio
   ```

#### 2. Installing Whisper

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

#### 3. Installing Ollama (for LLM support)

1. Install Ollama:
   ```cmd
   winget install Ollama
   ```
   Or download from [Ollama's official website](https://ollama.com/download)

2. After installation, download a compatible model:
   ```cmd
   ollama pull llama2
   ```
   (You can replace llama2 with another model of your choice, such as mistral or phi)

#### 4. Configure the Application

1. Review and modify `config.yaml` according to your preferences:
   - Adjust audio settings
   - Change voice settings
   - Configure hotkeys
   - Set LLM parameters

### Usage Instructions

1. Start the assistant:
   - Double click `start_jarvis.bat` in the root directory
   - Or run `bat/jarvis_window.bat` directly

2. The system will initialize and show in the system tray.

3. System tray options:
   - Status: Shows current assistant status
   - Settings: Opens the settings window
   - Exit: Closes the application

4. Default global hotkeys:
   - Ctrl + F1: Toggle listening mode
   - Ctrl + F2: Switch between available voices
   - Ctrl + F3: Adjust speech rate

5. Using the assistant:
   - The system monitors your microphone for speech
   - When speech is detected, it's transcribed using Whisper
   - The transcription is sent to the local LLM for processing
   - The LLM's response is synthesized into speech and played back

### Troubleshooting

- **No audio input detected**: Check your microphone settings and ensure the correct device is selected
- **Speech recognition issues**: Try using a larger model like `base.en.bin` for better accuracy
- **LLM not responding**: Ensure Ollama is running and you've pulled a model
- **TTS not working**: Check your internet connection as Edge TTS requires connectivity

### Future Plans

#### 1. More Human-like Conversation Experience
- Allow users to interrupt Jarvis while speaking
- Jarvis only processes the most recent user audio as conversation topic
- More natural conversation flow and tone
- Context awareness and emotional understanding

#### 2. Memory Functionality
- Users can imprint their own thought patterns on Jarvis
- Long-term memory storage and retrieval
- Personalized conversation style adaptation
- User preference learning and memory

#### 3. MCP (Master Control Program)
- Allow Jarvis to operate desktop files and basic software
- Implement email viewing/replying functionality
- Schedule planning and management
- Basic agent capabilities, such as:
  * File management and organization
  * Application control
  * Calendar and reminder management
  * Email processing and response
  * Simple automation tasks

## License

MIT License 

