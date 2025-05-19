# Jarvis Voice Assistant
![ironman](./pics/ironman.jpg =100x100)
Jarvis is a Windows-based local voice assistant system featuring real-time voice activation, speech recognition, local language model processing, and voice synthesis. It's designed to operate completely offline, ensuring privacy while providing a comprehensive voice interaction experience.



## Getting Started

To get started with Jarvis:

```bash
# Clone the repository
git clone https://github.com/jmtitan/Jarvis.git
cd Jarvis

# Set up environment and install dependencies
python -m venv jarvis
jarvis\Scripts\activate
pip install -r requirements.txt

# Download required model files (not included in repo)
# See "Installing Whisper" section for details
```

## Key Features

- Real-time voice activation with WebRTC VAD
- Offline speech recognition powered by Whisper.cpp
- Local large language model processing via Ollama
- Multiple voice synthesis options with Edge TTS
- System tray integration for easy access
- Global hotkey support for quick controls
- Customizable voice, speech rate and volume
- Settings UI for easy configuration

## System Requirements

- Windows 10/11
- Python 3.8+
- At least 6-core CPU
- 16GB RAM
- Recommended: NVIDIA GPU (8GB+ VRAM)

## Detailed Installation Guide

### 1. Setting up Python Environment

1. Install Python 3.8 or newer from [python.org](https://www.python.org/downloads/)
2. Create and activate a virtual environment (recommended):
   ```cmd
   python -m venv jarvis
   jarvis\Scripts\activate
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

### 2. Installing Whisper

The project uses Whisper.cpp for speech recognition, which requires model files and binaries.

#### Whisper Model Files
1. The project includes `tiny.en.bin` and `base.en.bin` models in the `models` directory
2. If you need to download them manually:
   - Visit [ggerganov/whisper.cpp/releases](https://github.com/ggerganov/whisper.cpp/releases)
   - Download either `ggml-tiny.en.bin` or `ggml-base.en.bin` 
   - Rename the files to `tiny.en.bin` or `base.en.bin`
   - Place them in the `models` directory

#### Whisper.cpp Binaries
1. The project includes necessary binaries in the `whisper_bin` directory
2. If you need to install them manually:
   - Visit [ggerganov/whisper.cpp/releases](https://github.com/ggerganov/whisper.cpp/releases)
   - Download the Windows release zip file
   - Extract the contents
   - Copy `whisper.dll`, `main.exe` and related files to the `whisper_bin` directory

### 3. Installing Ollama (for LLM support)

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

### 4. Configure the Application

1. Review and modify `config.yaml` according to your preferences:
   - Adjust audio settings
   - Change voice settings
   - Configure hotkeys
   - Set LLM parameters

## Usage Instructions

1. Start the assistant:
   ```cmd
   python src/main.py
   ```

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

## Troubleshooting

- **No audio input detected**: Check your microphone settings and ensure the correct device is selected
- **Speech recognition issues**: Try using a larger model like `base.en.bin` for better accuracy
- **LLM not responding**: Ensure Ollama is running and you've pulled a model
- **TTS not working**: Check your internet connection as Edge TTS requires connectivity

## License

MIT License 
