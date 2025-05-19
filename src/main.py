import asyncio
import yaml
import sys
import os
from typing import Optional
import win32gui
import win32con
import win32api
import keyboard
import threading
import queue
import subprocess
from pathlib import Path

# Import playsound library
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
    print("Found playsound library")
except ImportError:
    PLAYSOUND_AVAILABLE = False
    print("playsound not available, please install with: pip install playsound==1.2.2")
    # Don't exit, other playback methods may be available

# Import custom modules
from audio.listener import AudioListener
from stt.whisper_stt import WhisperSTT
from llm.ollama_client import OllamaClient
from tts.voice_synthesizer import VoiceSynthesizer
from ui_manager import SettingsUI, TrayIconUI

class JarvisAssistant:
    def __init__(self):
        self.config = self._load_config()
        self.audio_listener = None
        self.stt_engine = None
        self.llm_client = None
        self.tts_engine = None
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.ui_manager = None
        self.tray_icon = None
        
    def _load_config(self) -> dict:
        """Load configuration file"""
        try:
            print("Loading configuration file...")
            with open('config.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            sys.exit(1)
            
    async def initialize(self):
        """Initialize all components"""
        print("Initializing all components...")
        # Initialize audio listener
        self.audio_listener = AudioListener(self.config)
        print("Audio listener initialization complete")
        
        # Initialize speech recognition
        self.stt_engine = WhisperSTT(self.config)
        print("Speech recognition initialization complete")
        
        # Initialize LLM client
        self.llm_client = OllamaClient(self.config)
        print("LLM client initialization complete")
        
        # Initialize speech synthesis
        self.tts_engine = VoiceSynthesizer(self.config)
        await self.tts_engine.initialize()
        print("Speech synthesis initialization complete")
        
        # Initialize UI components
        self._init_ui_components()
        
    def _init_ui_components(self):
        """Initialize UI components"""
        # Initialize settings UI manager
        self.ui_manager = SettingsUI(
            tts_engine=self.tts_engine,
            config=self.config,
            save_config_callback=self._save_config,
            play_audio_callback=self._play_audio,
            setup_hotkeys_callback=self._setup_hotkeys
        )
        
        # Initialize system tray
        self.tray_icon = TrayIconUI(
            show_status_callback=self._show_status,
            show_settings_callback=self._show_settings,
            quit_callback=self._quit
        )
        
        # Set icon
        icon_path = Path(os.getcwd()) / 'pics' / 'ironman.jpg'
        self.tray_icon.create_tray_icon(icon_path)
        
    def _save_config(self):
        """Save configuration to file"""
        with open('config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)
        
    def _show_status(self):
        """Show status information"""
        print("Showing status information...")
        status = {
            "Listening status": "Running" if self.is_running else "Stopped",
            "Current model": self.llm_client.get_model_info()["name"],
            "Current voice": self.tts_engine.get_current_settings()["voice"]
        }
        print(status)
        
    def _show_settings(self):
        """Show settings window"""
        print("Showing settings window...")
        self.ui_manager.show_settings()
        
    def _quit(self):
        """Exit program"""
        print("Exiting program...")
        self.is_running = False
        self.tray_icon.stop()
        
    def _setup_hotkeys(self):
        """Set up global hotkeys"""
        print("Setting up global hotkeys...")
        keyboard.add_hotkey(
            self.config['hotkeys']['toggle_listening'],
            self._toggle_listening
        )
        keyboard.add_hotkey(
            self.config['hotkeys']['switch_voice'],
            self._switch_voice
        )
        keyboard.add_hotkey(
            self.config['hotkeys']['adjust_speed'],
            self._adjust_speed
        )
        print("Global hotkeys setup complete")
        
    def _toggle_listening(self):
        """Toggle listening status"""
        print("Toggling listening status...")
        if self.is_running:
            print("Stopping audio listening...")
            self.audio_listener.stop()
            self.stt_engine.stop()
        else:
            print("Starting audio listening...")
            self.audio_listener.start(self._on_audio_data)
            self.stt_engine.start(self._on_stt_result)
        self.is_running = not self.is_running
        
    def _switch_voice(self):
        """Switch voice"""
        print("Switching voice...")
        voices = list(self.tts_engine.get_available_voices().keys())
        current_index = voices.index(self.tts_engine.voice)
        next_index = (current_index + 1) % len(voices)
        self.tts_engine.set_voice(voices[next_index])
        print(f"Voice switched: {self.tts_engine.voice}->{voices[next_index]}")
        
    def _adjust_speed(self):
        """Adjust speech rate"""
        current_rate = self.tts_engine.rate
        new_rate = current_rate + 0.1 if current_rate < 1.0 else 0.1
        print(f"Adjusting speech rate: {current_rate}->new_rate")
        self.tts_engine.set_rate(new_rate)
        
    def _on_audio_data(self, audio_data):
        """Process audio data"""
        print("Received audio data...")
        self.stt_engine.process_audio(audio_data)
        
    def _on_stt_result(self, text):
        """Process speech recognition result"""
        print(f"Speech recognition result: {text}")
        asyncio.run(self._process_text(text))
        
    async def _play_audio(self, file_path):
        """Play audio file"""
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return
            
        try:
            print("Playing audio...")
            # Use playsound to play audio
            if PLAYSOUND_AVAILABLE:
                print("Using: playsound library")
                # Use thread executor to run playsound (as it's blocking)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: playsound(file_path))
            else:
                # Fallback: use system default player
                print("Using system default player")
                if os.name == 'nt':  # Windows
                    os.startfile(file_path)
                else:  # Linux/Mac
                    subprocess.run(['xdg-open', file_path])
        except Exception as e:
            print(f"Playback failed: {e}")
        
    async def _process_text(self, text):
        """Process text and generate response"""
        print("Processing text...")
        # Call LLM to generate response
        response = await self.llm_client.generate(text)
        if response:
            print(f"LLM response: {response}")
            # Synthesize speech
            audio_path = await self.tts_engine.speak(response)
            if audio_path:
                print(f"Speech synthesis complete, saved to: {audio_path}")
                # Play audio
                await self._play_audio(audio_path)
                
    async def run(self):
        """Run assistant"""
        print("Starting assistant...")
        await self.initialize()
        self._setup_hotkeys()
        
        # Start system tray
        self.tray_icon.run()
        
if __name__ == "__main__":
    assistant = JarvisAssistant()
    asyncio.run(assistant.run()) 