import asyncio
import yaml
import sys
import re
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
import time

# Import pygame for controllable audio playback
try:
    import pygame
    PYGAME_AVAILABLE = True
    print("Found pygame library")
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not available, please install with: pip install pygame")

# Import playsound library as fallback
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
from memory.memory_manager import MemoryManager

class JarvisAssistant:
    def __init__(self):
        self.config = self._load_config()
        self.audio_listener = None
        self.stt_engine = None
        self.llm_client = None
        self.tts_engine = None
        self.memory_manager = None
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.ui_manager = None
        self.tray_icon = None
        
        # Threading and asyncio management
        self.main_loop = None
        self.async_task_queue = queue.Queue()
        self.async_thread = None
        self.loop_ready_event = threading.Event()
        self.shutdown_event = threading.Event()
        
        # Continuous conversation mode
        self.continuous_mode = False
        self.continuous_timer = None
        self.continuous_mode_duration = 10.0  # 10 seconds for natural conversation pauses
        
        # Audio playback interruption
        self.is_playing_audio = False
        self.should_interrupt_audio = False
        self.audio_playback_thread = None
        self.current_audio_process = None
        self.audio_interruption_lock = threading.Lock()
        self.processing_interrupted = False  # Global interruption flag
        
        # Initialize pygame mixer if available
        self.pygame_available = PYGAME_AVAILABLE
        if self.pygame_available:
            try:
                pygame.mixer.init()
                print("Pygame mixer initialized successfully")
            except Exception as e:
                print(f"Failed to initialize pygame mixer: {e}")
                self.pygame_available = False
        
        # Prepare wake words from config
        self.wake_words_config = self.config.get('wake_word', {})
        self.wake_words_enabled = self.wake_words_config.get('enabled', False)
        # Ensure wake_words is a list and lowercase for matching
        raw_wake_words = self.wake_words_config.get('words', [])
        if isinstance(raw_wake_words, str): # Handle if user puts a single string
            self.lowercase_wake_words = [raw_wake_words.lower().strip()]
        elif isinstance(raw_wake_words, list):
            self.lowercase_wake_words = [str(w).lower().strip() for w in raw_wake_words if str(w).strip()]
        else:
            self.lowercase_wake_words = []
        if self.wake_words_enabled and not self.lowercase_wake_words:
            print("Warning: Wake word is enabled, but no wake words are defined in config.yaml under wake_word.words")
        elif self.wake_words_enabled:
            print(f"Wake words loaded: {self.lowercase_wake_words}")
        
    def _load_config(self) -> dict:
        """Load configuration file"""
        try:
            print("Loading configuration file...")
            with open('config.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            sys.exit(1)
    
    def _run_async_loop(self):
        """Run asyncio event loop in background thread"""
        print("Starting background asyncio thread...")
        self.main_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.main_loop)
        
        try:
            # Signal that the loop is ready
            self.loop_ready_event.set()
            print("Background asyncio loop is ready.")
            
            # Start the async main coroutine
            self.main_loop.run_until_complete(self.async_main())
        except Exception as e:
            print(f"Error in async loop: {e}")
        finally:
            print("Background asyncio loop finished.")
            self.main_loop.close()
    
    async def async_main(self):
        """Main asyncio coroutine running in background thread"""
        print("Starting async main...")
        
        # Initialize async components
        await self.initialize_async_components()
        
        # Start the async task processor
        task_processor = self.main_loop.create_task(self._async_task_processor())
        print("Async task processor started.")
        
        # Wait for shutdown signal
        while not self.shutdown_event.is_set():
            await asyncio.sleep(0.1)
        
        print("Shutdown signal received, stopping async components...")
        
        # Cancel task processor
        task_processor.cancel()
        try:
            await task_processor
        except asyncio.CancelledError:
            pass
        
        # Shutdown async components
        await self._shutdown_async_components()
        print("Async components shutdown complete.")
            
    async def initialize_async_components(self):
        """Initialize async components only"""
        print("Initializing async components...")
        
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
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(self.config)
        self.memory_manager.set_llm_client(self.llm_client)
        print("Memory manager initialization complete")
        
    async def _shutdown_async_components(self):
        """Shutdown async components gracefully"""
        try:
            if self.memory_manager:
                await self.memory_manager.shutdown()
            if self.audio_listener:
                self.audio_listener.stop()
            if self.stt_engine:
                self.stt_engine.stop()
            if self.llm_client:
                await self.llm_client.close()
        except Exception as e:
            print(f"Error during async components shutdown: {e}")

    def _init_ui_components(self):
        """Initialize UI components (runs in main thread)"""
        print("Initializing UI components...")
        
        # Initialize settings UI manager
        self.ui_manager = SettingsUI(
            tts_engine=self.tts_engine,
            config=self.config,
            save_config_callback=self._save_config,
            play_audio_callback=self._play_audio_thread_safe,
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
        print("UI components initialization complete")
        
    def _play_audio_thread_safe(self, file_path):
        """Thread-safe version of _play_audio for UI callbacks"""
        if self.main_loop and not self.shutdown_event.is_set():
            future = asyncio.run_coroutine_threadsafe(
                self._play_audio(file_path), 
                self.main_loop
            )
            try:
                # Wait for completion with timeout
                future.result(timeout=30)
            except Exception as e:
                print(f"Error in thread-safe audio playback: {e}")
        
    def _save_config(self):
        """Save configuration to file"""
        with open('config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)
        
    def _show_status(self):
        """Show status information"""
        print("Showing status information...")
        continuous_status = "Active" if self.continuous_mode else "Inactive"
        timing_info = ""
        if self.continuous_mode:
            timing_info = f" (exits after {self.continuous_mode_duration}s of silence)"
        
        audio_status = "Playing" if self.is_playing_audio else "Idle"
        audio_engine = "pygame" if self.pygame_available else "playsound" if PLAYSOUND_AVAILABLE else "system default"
        
        # Get memory statistics
        memory_stats = {}
        if self.memory_manager:
            memory_stats = self.memory_manager.get_memory_stats()
        
        status = {
            "Listening status": "Running" if self.is_running else "Stopped",
            "Current model": self.llm_client.get_model_info()["name"] if self.llm_client else "Not initialized",
            "Current voice": self.tts_engine.get_current_settings()["voice"] if self.tts_engine else "Not initialized",
            "Continuous mode": f"{continuous_status}{timing_info}",
            "Wake words required": "No" if self.continuous_mode else "Yes",
            "Audio playback": f"{audio_status} (using {audio_engine})",
            "Audio settings": f"Max speech: 15s, Silence between sentences: 2s",
            "Interruption": "Enabled - speak to interrupt AI (safe mode)",
            "Memory system": {
                "Total memories": memory_stats.get('total_memories', 0),
                "Conversation messages": memory_stats.get('conversation_messages', 0),
                "Long-term summaries": memory_stats.get('long_term_summaries', 0),
                "Important facts": memory_stats.get('important_facts', 0),
                "User interests": memory_stats.get('user_interests', 0)
            }
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
        
        # Stop any ongoing audio playback safely
        if self.is_playing_audio:
            print("Stopping audio playback before exit...")
            self._interrupt_audio_playback()
            # Brief wait for cleanup
            time.sleep(0.2)
        
        # Cleanup pygame if it was initialized
        if self.pygame_available:
            try:
                pygame.mixer.quit()
                print("Pygame mixer cleaned up")
            except Exception as e:
                print(f"Error cleaning up pygame: {e}")
        
        # Clean up continuous mode timer
        self._reset_continuous_mode()
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Signal the async task processor to stop
        if self.async_task_queue:
            self.async_task_queue.put(None)
        
        # Stop tray icon (this will exit the main thread loop)
        if self.tray_icon:
            self.tray_icon.stop()
        
        # Wait for async thread to finish
        if self.async_thread and self.async_thread.is_alive():
            print("Waiting for async thread to finish...")
            self.async_thread.join(timeout=5)
            if self.async_thread.is_alive():
                print("Warning: Async thread did not finish within timeout")
        
    def _setup_hotkeys(self):
        """Set up global hotkeys"""
        print("Setting up global hotkeys...")
        try:
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
        except Exception as e:
            print(f"Error setting up hotkeys: {e}")
        
    def _toggle_listening(self):
        """Toggle listening status"""
        print("Toggling listening status...")
        if self.is_running:
            print("Stopping audio listening...")
            if self.audio_listener:
                self.audio_listener.stop()
            if self.stt_engine:
                self.stt_engine.stop()
        else:
            print("Starting audio listening...")
            if self.audio_listener and self.stt_engine:
                self.audio_listener.start(self._on_audio_data)
                self.stt_engine.start(self._on_stt_result)
        self.is_running = not self.is_running
        
    def _switch_voice(self):
        """Switch voice"""
        print("Switching voice...")
        if self.tts_engine:
            try:
                voices = list(self.tts_engine.get_available_voices().keys())
                current_index = voices.index(self.tts_engine.voice)
                next_index = (current_index + 1) % len(voices)
                self.tts_engine.set_voice(voices[next_index])
                print(f"Voice switched: {self.tts_engine.voice}->{voices[next_index]}")
            except Exception as e:
                print(f"Error switching voice: {e}")
        
    def _adjust_speed(self):
        """Adjust speech rate"""
        if self.tts_engine:
            try:
                current_rate = self.tts_engine.rate
                new_rate = current_rate + 0.1 if current_rate < 1.0 else 0.1
                print(f"Adjusting speech rate: {current_rate}->{new_rate}")
                self.tts_engine.set_rate(new_rate)
            except Exception as e:
                print(f"Error adjusting speed: {e}")
        
    def _on_audio_data(self, audio_data):
        """Process audio data"""
        # If in continuous mode, reset the timer since we detected audio activity
        if self.continuous_mode:
            self._reset_continuous_timer()
        
        if self.stt_engine:
            self.stt_engine.process_audio(audio_data)
        
    def _on_stt_result(self, text: str):
        """Process speech recognition result"""
        print(f"Original STT result: '{text}'") 
        
        # Check if AI is currently speaking and interrupt if necessary
        if self.is_playing_audio:
            print("User interrupted AI - stopping current speech")
            self._interrupt_audio_playback()
            
            # Brief wait for pygame to stop (much faster than before)
            time.sleep(0.1)
            
            # Enter continuous mode immediately since user is interrupting
            self.continuous_mode = True
            if self.continuous_timer:
                self.continuous_timer.cancel()
                self.continuous_timer = None
            
            print("Previous audio stopped, ready to process new input")
        
        # Clear any previous interruption flag for new processing
        self.processing_interrupted = False
        
        cleaned_stt_text = text.lower() # Convert to lowercase first
        
        # Remove specific known prefixes like "- "
        if cleaned_stt_text.startswith("- "):
            cleaned_stt_text = cleaned_stt_text[2:]
        
        # Remove all punctuation (characters that are not word characters or whitespace)
        # \w matches any word character (equal to [a-zA-Z0-9_])
        # \s matches any whitespace character (equal to [ \t\n\r\f\v])
        # [^\w\s] matches any character that is NOT a word character and NOT whitespace (i.e., punctuation)
        cleaned_stt_text = re.sub(r'[^\w\s]', '', cleaned_stt_text) 
        # Normalize multiple spaces to single space and strip leading/trailing spaces
        cleaned_stt_text = re.sub(r'\s+', ' ', cleaned_stt_text).strip()
        
        # Remove blank audio markers that interfere with normal conversation
        blank_audio_patterns = ['blank_audio', 'blank audio', 'silence', 'no audio', 'quiet']
        for pattern in blank_audio_patterns:
            cleaned_stt_text = cleaned_stt_text.replace(pattern, '').strip()
        # Clean up any multiple spaces again after removal
        cleaned_stt_text = re.sub(r'\s+', ' ', cleaned_stt_text).strip()

        print(f"Cleaned STT result for matching: '{cleaned_stt_text}'")
        
        # Filter out blank/empty audio inputs
        # Only ignore if the entire cleaned text is one of these patterns, not if it contains them
        blank_patterns = ['blank_audio', 'blank audio', '', 'silence', 'no audio', 'quiet']
        if cleaned_stt_text in blank_patterns or len(cleaned_stt_text.strip()) == 0:
            print("Ignoring blank or empty audio input")
            return
        
        # Filter out very short meaningless inputs
        if len(cleaned_stt_text.strip()) < 2:
            print("Ignoring very short input (likely noise)")
            return
        
        command_to_process = None

        # Check if we're in continuous conversation mode
        if self.continuous_mode:
            print("Continuous conversation mode active - processing without wake word")
            command_to_process = cleaned_stt_text
            if not command_to_process:
                print("STT result is empty after cleaning, skipping processing.")
                return
            # Continuous mode timer will be paused during processing
        
        # Normal wake word processing
        elif self.wake_words_enabled and self.lowercase_wake_words:
            found_wake_word = False
            for wake_word in self.lowercase_wake_words:
                # Ensure wake_word itself doesn't have extra spaces (already handled during loading)
                if cleaned_stt_text.startswith(wake_word):
                    # Extract the part after the wake word
                    command_part_raw = cleaned_stt_text[len(wake_word):].strip()
                    
                    # Further clean the command part: e.g., remove a leading comma if STT adds it.
                    # This regex approach is more robust if multiple punctuation types could appear.
                    # For now, a simple startswith check for comma is fine as per previous discussion.
                    if command_part_raw.startswith(','):
                         command_part_raw = command_part_raw[1:].strip()
                    
                    command_part_clean = command_part_raw # command_part_raw should be fairly clean now

                    print(f"Wake word '{wake_word}' detected. Command: '{command_part_clean}'")
                    if command_part_clean: 
                        command_to_process = command_part_clean
                    else:
                        # This case means only the wake word was said, e.g., "Hey Jarvis"
                        print("Wake word detected, but no command followed.")
                        # Optionally: play a sound, enter a temporary "listening for command" state
                    found_wake_word = True
                    break 
            
            if not found_wake_word:
                print(f"No wake word detected in: '{cleaned_stt_text}' (Original STT: '{text}')")
                return # Do not process further if wake word is required and not found
        
        else: # Wake word not enabled, or no wake words defined, process the (cleaned) text directly
            command_to_process = cleaned_stt_text 
            if not command_to_process: # If STT gives an empty string even after cleaning
                print("STT result is empty after cleaning, skipping processing.")
                return

        if command_to_process:
            print(f"Sending to LLM: '{command_to_process}'")
            # Put task into queue for async processing
            if not self.shutdown_event.is_set():
                self.async_task_queue.put(self._process_text(command_to_process))
            else:
                print("Shutdown in progress, ignoring command.")
        
    async def _play_audio(self, file_path):
        """Play audio file with pygame for easy interruption"""
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return
            
        with self.audio_interruption_lock:
            # Stop any previous audio first
            if self.is_playing_audio:
                print("Stopping previous audio before starting new one...")
                self._stop_pygame_audio()
                
            self.is_playing_audio = True
            self.should_interrupt_audio = False
        
        try:
            print("Playing audio...")
            
            if self.pygame_available:
                print("Using: pygame mixer")
                await self._play_audio_with_pygame(file_path)
            elif PLAYSOUND_AVAILABLE:
                print("Using: playsound library (fallback)")
                await self._play_audio_with_playsound(file_path)
            else:
                print("Using: system default player (fallback)")
                await self._play_audio_system_default(file_path)
                    
        except Exception as e:
            if not self.should_interrupt_audio:
                print(f"Playback failed: {e}")
        finally:
            with self.audio_interruption_lock:
                self.is_playing_audio = False
                self.should_interrupt_audio = False
    
    async def _play_audio_with_pygame(self, file_path):
        """Play audio using pygame mixer with interruption support"""
        try:
            # Load and play the audio file
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # Monitor playback with interruption checks
            while pygame.mixer.music.get_busy():
                if self.should_interrupt_audio:
                    print("Audio playback interrupted by user")
                    pygame.mixer.music.stop()
                    return
                
                # Check every 50ms for responsive interruption
                await asyncio.sleep(0.05)
                
            print("Audio playback completed normally")
            
        except Exception as e:
            print(f"Pygame audio playback error: {e}")
            # Fallback to playsound if pygame fails
            if PLAYSOUND_AVAILABLE:
                print("Falling back to playsound...")
                await self._play_audio_with_playsound(file_path)
    
    async def _play_audio_with_playsound(self, file_path):
        """Fallback audio playback with playsound"""
        loop = asyncio.get_event_loop()
        
        def play_audio_in_thread():
            try:
                if self.should_interrupt_audio:
                    return
                playsound(file_path)
            except Exception as e:
                if not self.should_interrupt_audio:
                    print(f"Playsound error: {e}")
        
        # Start playback in executor
        future = loop.run_in_executor(None, play_audio_in_thread)
        
        # Monitor for interruption
        while not future.done():
            if self.should_interrupt_audio:
                print("Attempting to cancel playsound playback")
                future.cancel()
                return
            await asyncio.sleep(0.1)
        
        try:
            await future
        except asyncio.CancelledError:
            print("Playsound playback cancelled")
    
    async def _play_audio_system_default(self, file_path):
        """Play audio using system default player"""
        if os.name == 'nt':  # Windows
            os.startfile(file_path)
        else:  # Linux/Mac
            subprocess.run(['xdg-open', file_path])
    
    def _stop_pygame_audio(self):
        """Stop pygame audio playback"""
        if self.pygame_available:
            try:
                pygame.mixer.music.stop()
                print("Pygame audio stopped")
            except Exception as e:
                print(f"Error stopping pygame audio: {e}")
                
    def _interrupt_audio_playback(self):
        """Interrupt current audio playback safely"""
        with self.audio_interruption_lock:
            if self.is_playing_audio:
                print("Interrupting AI speech...")
                self.should_interrupt_audio = True
                self.processing_interrupted = True
                self._stop_pygame_audio()
        
    async def _process_text(self, text):
        """Process text and generate response"""
        print("Processing text...")
        
        # Pause continuous mode timer during AI processing to avoid expiring while AI speaks
        if self.continuous_mode and self.continuous_timer:
            self.continuous_timer.cancel()
            self.continuous_timer = None
        
        # Check if processing was interrupted before starting
        if self.processing_interrupted:
            print("Processing skipped due to interruption")
            return
        
        # Check if LLM client is available
        if not self.llm_client:
            print("Error: LLM client is not initialized")
            return
            
        try:
            # Build enhanced prompt with memory context
            if self.memory_manager:
                enhanced_prompt = await self.memory_manager.build_enhanced_prompt(text)
                print(f"Using memory-enhanced prompt (length: {len(enhanced_prompt)} chars)")
            else:
                enhanced_prompt = text
                print("Using simple prompt (memory manager not available)")
            
            # Call LLM to generate response
            print(f"Calling LLM with enhanced prompt")
            response = await self.llm_client.generate(enhanced_prompt)
            
            # Check for interruption after LLM call
            if self.processing_interrupted:
                print("Processing interrupted after LLM response")
                return
            
            if response:
                print(f"LLM response: {response}")
                
                # Check if TTS engine is available
                if not self.tts_engine:
                    print("Error: TTS engine is not initialized")
                    return
                    
                # Synthesize speech (fast step)
                try:
                    audio_path = await self.tts_engine.speak(response)
                    
                    # Final interruption check before audio/memory operations
                    if self.processing_interrupted or self.should_interrupt_audio:
                        print("Text processing interrupted before audio playback")
                        return
                    
                    if audio_path:
                        print(f"Speech synthesis complete, saved to: {audio_path}")
                        
                        # Start memory storage in parallel with audio playback to reduce response time
                        memory_task = None
                        if self.memory_manager:
                            print("Starting memory storage in background...")
                            memory_task = asyncio.create_task(
                                self._store_memory_in_background(text, response)
                            )
                        
                        # Play audio (this is the main delay for user experience)
                        await self._play_audio(audio_path)
                        
                        # Wait for memory storage to complete (if it's still running)
                        if memory_task and not memory_task.done():
                            try:
                                await memory_task
                                print("Background memory storage completed")
                            except Exception as e:
                                print(f"Error in background memory storage: {e}")
                        
                        # Only enable continuous conversation mode if not interrupted
                        if not self.processing_interrupted and not self.should_interrupt_audio:
                            print("AI response completed, enabling continuous conversation mode...")
                            self._enable_continuous_mode()
                        else:
                            print("Skipping continuous mode activation due to interruption")
                        
                    else:
                        print("Speech synthesis failed: no audio file generated")
                except Exception as e:
                    print(f"Error during speech synthesis: {e}")
            else:
                print("LLM returned empty response")
                
        except Exception as e:
            print(f"Error processing text: {e}")
            # Additional diagnostic info
            if "Session is closed" in str(e):
                print("Detected session closure error - this should be fixed with the new OllamaClient implementation")
            elif "Connection" in str(e):
                print("Connection error - check if Ollama server is running and accessible")
            else:
                print(f"Unexpected error type: {type(e).__name__}")
    
    async def _store_memory_in_background(self, user_input: str, ai_response: str):
        """Store interaction in memory as a background task"""
        try:
            await self.memory_manager.process_interaction(user_input, ai_response)
            print("Interaction stored in memory")
        except Exception as e:
            print(f"Error storing interaction in memory: {e}")

    def _enable_continuous_mode(self):
        """Enable continuous conversation mode for a limited time"""
        print(f"Entering continuous conversation mode...")
        print("You can now speak directly without using wake words!")
        self.continuous_mode = True
        
        # Cancel any existing timer
        if self.continuous_timer:
            self.continuous_timer.cancel()
        
        # Start timer for silence detection (will be reset when audio is detected)
        self._reset_continuous_timer()
    
    def _reset_continuous_timer(self):
        """Reset the continuous mode timer (called when audio activity is detected)"""
        if not self.continuous_mode:
            return
            
        # Cancel existing timer
        if self.continuous_timer:
            self.continuous_timer.cancel()
        
        # Start new timer - only exit if there's silence for the full duration
        self.continuous_timer = threading.Timer(
            self.continuous_mode_duration, 
            self._disable_continuous_mode
        )
        self.continuous_timer.start()
    
    def _disable_continuous_mode(self):
        """Disable continuous conversation mode"""
        print("Continuous conversation mode expired (10 seconds of silence detected)")
        print("Wake word required for next interaction")
        self.continuous_mode = False
        if self.continuous_timer:
            self.continuous_timer.cancel()
            self.continuous_timer = None
    
    def _reset_continuous_mode(self):
        """Reset continuous mode timer (called when new conversation starts)"""
        if self.continuous_timer:
            self.continuous_timer.cancel()
            self.continuous_timer = None
        self.continuous_mode = False

    async def _async_task_processor(self):
        """Process asynchronous tasks from a queue."""
        print("Async task processor loop started.")
        while not self.shutdown_event.is_set():
            try:
                # Non-blocking get with timeout
                try:
                    coro = self.async_task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                if coro is None:  # Sentinel value to stop
                    print("Async task processor received stop signal.")
                    break
                
                # Execute the coroutine
                await coro
                
            except Exception as e:
                print(f"Error in async task processor: {e}")
        print("Async task processor loop finished.")

    def run(self):
        """Run assistant (main entry point)"""
        print("Starting Jarvis Assistant...")
        
        # Start the background asyncio thread
        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.async_thread.start()
        
        # Wait for the async loop to be ready
        print("Waiting for async loop to be ready...")
        self.loop_ready_event.wait(timeout=10)
        if not self.loop_ready_event.is_set():
            print("Error: Async loop failed to start within timeout")
            return
        
        # Wait a bit for components to initialize
        time.sleep(2)
        
        # Initialize UI components in main thread
        self._init_ui_components()
        
        # Set up hotkeys in main thread
        self._setup_hotkeys()
        
        print("Starting system tray (this will block the main thread)...")
        
        # Run the tray icon (this blocks the main thread)
        try:
            self.tray_icon.run()
        except KeyboardInterrupt:
            print("Keyboard interrupt received")
        except Exception as e:
            print(f"Error in tray icon: {e}")
        finally:
            print("Tray icon stopped, cleaning up...")
            self._quit()

if __name__ == "__main__":
    assistant = JarvisAssistant()
    assistant.run() 