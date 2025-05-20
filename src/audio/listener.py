import sounddevice as sd
import numpy as np
import webrtcvad
import threading
import queue
import time
from typing import Optional, Callable

class AudioListener:
    def __init__(self, config: dict):
        self.config = config
        self.vad = webrtcvad.Vad(3)  # set VAD sensitivity
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.callback = None
        assert config['audio']['frame_ms'] in (10, 20, 30), "frame_ms must be 10/20/30 ms"
        assert config['audio']['sample_rate'] in (8000, 16000, 32000, 48000), "sample_rate must be 8000/16000/32000/48000"
        
        # audio parameters
        self.sample_rate = config['audio']['sample_rate']
        self.channels = config['audio']['channels']
        self.chunk_size = int(self.sample_rate * config['audio']['frame_ms'] / 1000) 
        self.device_index = config['audio']['device_index']
        
        # speech detection parameters
        self.min_silence_frames = int(0.5 * 1000 / config['audio']['frame_ms'])  # 0.5 seconds silence set as end of speech
        self.max_speech_frames = int(10 * 1000 / config['audio']['frame_ms'])  # max 10 seconds speech
        
        # buffer
        self.audio_buffer = []
        self.silence_count = 0
        self.speech_count = 0
        self.is_speech_active = False
        
    def start(self, callback: Optional[Callable] = None):
        """Start audio listening"""
        if self.is_listening:
            return
            
        self.callback = callback
        self.is_listening = True
        
        # open audio stream
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype="int16",
            channels=self.channels,
            # device=self.device_index,
            callback=self._audio_callback
        )
        
        # explicitly start stream
        self.stream = self.stream.__enter__()
        
        # start listening thread
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        print("start listening...")
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback function"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(bytes(indata))
        
    def stop(self):
        """Stop audio listening"""
        self.is_listening = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
    def _listen_loop(self):
        """Audio listening loop"""
        while self.is_listening:
            try:
                if not self.audio_queue.empty():
                    frame = self.audio_queue.get()
                    audio_data = np.frombuffer(frame, dtype=np.int16)
                    
                    
                    try:
                        speech = self.vad.is_speech(frame, self.sample_rate)
                    except Exception as e:
                        print(f"VAD error: {e}")
                        speech = False
                 
                    if speech:
                       
                        if not self.is_speech_active:

                            self.is_speech_active = True
                            self.audio_buffer = []  
                           
                        
                    
                        self.audio_buffer.append(audio_data)
                        self.speech_count += 1
                        self.silence_count = 0
                        
                        if self.speech_count >= self.max_speech_frames:
                            if self.callback and self.audio_buffer:
                                combined_audio = np.concatenate(self.audio_buffer)
                                print(f"Speech too long, forced processing, data length: {len(combined_audio)}")
                                self.callback(combined_audio)
                            self.audio_buffer = []
                            self.speech_count = 0
                            self.is_speech_active = False
                    else:
                       
                        if self.is_speech_active:
                           
                            self.silence_count += 1
                            self.audio_buffer.append(audio_data)  
                            
                            if self.silence_count >= self.min_silence_frames:
                                if self.callback and self.audio_buffer:
                                    combined_audio = np.concatenate(self.audio_buffer)
                                    print(f"Detected end of speech, data length: {len(combined_audio)}")
                                    self.callback(combined_audio)
                                self.audio_buffer = []
                                self.silence_count = 0
                                self.speech_count = 0
                                self.is_speech_active = False
            except Exception as e:
                print(f"Audio listening error: {e}")
                time.sleep(0.1)
                
    def get_audio_devices(self):
        """Get available audio input devices"""
        devices = sd.query_devices()
        return [{'index': i, 'name': device['name'], 'channels': device['max_input_channels']} for i, device in enumerate(devices) if device['max_input_channels'] > 0] 