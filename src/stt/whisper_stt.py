import subprocess
import numpy as np
from typing import Optional, Callable
import threading
import queue
import time
import os
import tempfile
import wave
import re

class WhisperSTT:
    def __init__(self, config: dict):
        self.config = config
        self.is_processing = False
        self.audio_queue = queue.Queue()
        self.callback = None
        
        # set whisper.exe path
        self.whisper_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                       'whisper_bin', 'main.exe')
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     'models', f"{self.config['stt']['model']}.bin")
        
        if not os.path.exists(self.whisper_path):
            raise FileNotFoundError(f"whisper executable file not found: {self.whisper_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model file not found: {self.model_path}")
            
    def start(self, callback: Optional[Callable] = None):
        """start stt"""
        if self.is_processing:
            return
            
        self.callback = callback
        self.is_processing = True
        
        # start processing thread
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()

        print("stt started")
        
    def stop(self):
        """stop stt"""
        self.is_processing = False
        
    def process_audio(self, audio_data: np.ndarray):
        """process audio data"""
        # print("adding audio data...")
        # print("audio data length:", len(audio_data))
        # print("audio data content:", audio_data)
        self.audio_queue.put(audio_data)
        
    def _save_audio_to_wav(self, audio_data: np.ndarray) -> str:
        """save audio data to temporary wav file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(self.config['audio']['channels'])
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.config['audio']['sample_rate'])
                wav_file.writeframes(audio_data.tobytes())
            return temp_file.name
            
    def _process_loop(self):
        """audio processing loop"""
        while self.is_processing:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    
                    # save audio to temporary file
                    wav_path = self._save_audio_to_wav(audio_data)
                    try:
                        # call whisper.cpp
                        result = subprocess.run([
                            self.whisper_path,
                            '--model', self.model_path,
                            '--language', self.config['stt']['language'],
                            '-otxt',  # output as text
                            '-f', wav_path  # input file
                        ], capture_output=True, text=True)
                        # print("whisper.cpp call completed")
                        
                        if result.returncode == 0 and result.stdout:
                            print('stt result:', result.stdout)
                            text = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s*', '', result.stdout)
                            if self.callback:
                                self.callback(text.strip())
                                
                    finally:
                        # clean temporary file
                        try:
                            os.unlink(wav_path)
                        except:
                            pass
                            
            except Exception as e:
                print(f"stt error: {e}")
                time.sleep(0.1)
                
    def get_available_models(self):
        """get available whisper model list"""
        return [
            "tiny.en",
            "base.en",
            "small.en",
            "medium.en",
            "large"
        ] 