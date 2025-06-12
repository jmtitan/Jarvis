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
        audio_config = config['audio'] # Cache for convenience
        
        # VAD setup (allow sensitivity to be configured or default to 3)
        self.vad = webrtcvad.Vad(audio_config.get('vad_sensitivity', 3))
        
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.callback = None
        
        # Validate core audio parameters
        assert audio_config['frame_ms'] in (10, 20, 30), "frame_ms must be 10, 20, or 30 ms"
        assert audio_config['sample_rate'] in (8000, 16000, 32000, 48000), "sample_rate must be 8000, 16000, 32000, or 48000 Hz"
        
        # Audio parameters
        self.sample_rate = audio_config['sample_rate']
        self.channels = audio_config['channels']
        self.frame_duration_ms = audio_config['frame_ms']
        # Calculate chunk_size in samples (bytes will be chunk_size * 2 for int16)
        self.chunk_size = int(self.sample_rate * self.frame_duration_ms / 1000) 
        self.device_index = audio_config.get('device_index') # Use .get for optional device_index

        # Speech detection parameters from config, with defaults
        self.speech_min_duration_ms = audio_config.get('speech_min_duration_ms', 200)
        self.speech_max_silence_ms = audio_config.get('speech_max_silence_ms', 700)
        self.max_speech_duration_s = audio_config.get('max_speech_duration_s', 10)

        # Convert ms parameters to frame counts
        self.min_speech_frames = int(self.speech_min_duration_ms / self.frame_duration_ms)
        self.min_silence_frames_for_break = int(self.speech_max_silence_ms / self.frame_duration_ms)
        self.max_speech_frames_before_force_break = int(self.max_speech_duration_s * 1000 / self.frame_duration_ms)  

        # Buffers and state variables
        self.audio_buffer = [] # Stores audio frames (as np.ndarray) for the current utterance
        self.current_silence_frames = 0
        self.current_speech_frames = 0 # Number of speech frames VAD identified in current utterance
        self.frames_in_buffer = 0 # Total frames (speech + silence) in audio_buffer
        self.is_speech_active = False # True if VAD has detected speech and we are accumulating frames
        
    def start(self, callback: Optional[Callable] = None):
        """Start audio listening"""
        if self.is_listening:
            print("Listener already started.")
            return
            
        self.callback = callback
        
        # Reset state variables
        self.audio_buffer = []
        self.current_silence_frames = 0
        self.current_speech_frames = 0
        self.frames_in_buffer = 0
        self.is_speech_active = False
        while not self.audio_queue.empty(): # Clear queue if anything is left from previous run
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            # Open audio stream
            self.stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size, # samples per callback
                dtype="int16", # Whisper expects 16-bit PCM
                channels=self.channels,
                device=self.device_index, # Can be None to use default device
                callback=self._audio_callback
            )
        except Exception as e:
            print(f"Error opening audio stream with device_index '{self.device_index}': {e}")
            try:
                print("Attempting to use default audio device.")
                self.stream = sd.RawInputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    dtype="int16",
                    channels=self.channels,
                    device=None, # Explicitly use default
                    callback=self._audio_callback
                )
                self.device_index = None # Update to reflect default device usage
                print(f"Successfully opened default audio device.")
            except Exception as e_default:
                print(f"Failed to open default audio device: {e_default}")
                self.is_listening = False # Ensure we don't proceed if stream fails
                return

        # Explicitly start stream using its context management
        self.stream_context = self.stream.__enter__() 
        
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True # Allow main program to exit even if thread is running
        self.listen_thread.start()
        
        device_info = f"Device: {sd.query_devices(self.device_index)['name'] if self.device_index is not None else 'Default'}"
        print(f"AudioListener started. {device_info}, Rate: {self.sample_rate}, Chunk: {self.chunk_size} samples ({self.frame_duration_ms}ms per frame)")
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback function from sounddevice"""
        if status:
            # Log an actual warning if it's an input overflow/underflow
            if status.input_underflow:
                print(f"Warning: Input underflow detected by sounddevice: {status}", file=sys.stderr)
            elif status.input_overflow:
                print(f"Warning: Input overflow detected by sounddevice: {status}", file=sys.stderr)
            else:
                print(f"Audio callback status: {status}")
        
        # indata is a CFFI buffer. Convert to bytes for VAD and queueing.
        # VAD expects bytes.
        self.audio_queue.put(bytes(indata)) 
        
    def stop(self):
        """Stop audio listening"""
        if not self.is_listening:
            # print("Listener already stopped.") # Can be noisy if called multiple times
            return

        print("Attempting to stop AudioListener...")
        self.is_listening = False # Signal the listening loop to terminate
        
        # Wait for the listening thread to finish
        if hasattr(self, 'listen_thread') and self.listen_thread.is_alive():
            print("Joining listen_thread...")
            self.listen_thread.join(timeout=1.0) # Wait up to 1 second
            if self.listen_thread.is_alive():
                print("Warning: Listen thread did not terminate gracefully within timeout.")

        # Properly close the audio stream using its context manager's __exit__
        if hasattr(self, 'stream_context'):
            try:
                self.stream_context.__exit__(None, None, None)
                print("Audio stream stopped and closed via context.")
            except Exception as e:
                print(f"Error closing audio stream via context: {e}")
        elif hasattr(self, 'stream'): # Fallback if stream_context wasn't set (should not happen)
             try:
                self.stream.stop()
                self.stream.close()
                print("Audio stream stopped and closed (direct method).")
             except Exception as e:
                print(f"Error closing audio stream (direct method): {e}")
        
        # Clean up queue and buffers
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.audio_buffer = []
        self.current_speech_frames = 0
        self.current_silence_frames = 0
        self.frames_in_buffer = 0
        self.is_speech_active = False
        print("AudioListener stopped.")
        
    def _listen_loop(self):
        """Audio listening loop with improved buffering and speech segment detection."""
        print("Listen loop started.")
        byte_length_per_frame = self.chunk_size * self.channels * 2 # 2 bytes per int16 sample

        while self.is_listening:
            try:
                # Timeout allows checking self.is_listening periodically
                frame_bytes = self.audio_queue.get(timeout=0.1) 
                
                if len(frame_bytes) != byte_length_per_frame:
                    # This could happen if the audio stream provides data in different chunk sizes
                    # than expected, or if there's partial data.
                    # print(f"Warning: Received frame with unexpected byte length. Got {len(frame_bytes)}, expected {byte_length_per_frame}. Skipping frame.")
                    continue

                is_speech_current_frame = False
                try:
                    # VAD expects bytes of PCM data.
                    # Ensure sample_rate is correct for VAD.
                    is_speech_current_frame = self.vad.is_speech(frame_bytes, self.sample_rate)
                except Exception as e:
                    # This error ("Error while processing frame") can occur if frame_bytes is not a valid PCM audio frame
                    # of 10, 20, or 30 ms for the VAD.
                    # print(f"VAD processing error: {e}. Frame byte length: {len(frame_bytes)}. This might indicate an issue with audio chunking or format.")
                    # Treat as non-speech to avoid breaking the loop.
                    pass 

                # Convert frame to numpy array for buffering, regardless of speech content for now
                # This ensures that if we decide to send audio, we have the full segment.
                audio_frame_np = np.frombuffer(frame_bytes, dtype=np.int16)

                if self.is_speech_active:
                    # We are currently in a speech segment
                    self.audio_buffer.append(audio_frame_np)
                    self.frames_in_buffer += 1

                    if is_speech_current_frame:
                        self.current_speech_frames += 1
                        self.current_silence_frames = 0 # Reset silence counter on speech
                    else: # Silence frame during an active speech segment
                        self.current_silence_frames += 1

                    # Check for end of speech due to silence
                    if self.current_silence_frames >= self.min_silence_frames_for_break:
                        # print(f"End of speech detected due to silence ({self.current_silence_frames * self.frame_duration_ms}ms).")
                        if self.current_speech_frames >= self.min_speech_frames:
                            # print(f"Speech duration ({self.current_speech_frames * self.frame_duration_ms}ms) is sufficient.")
                            if self.callback and self.audio_buffer:
                                combined_audio = np.concatenate(self.audio_buffer)
                                # print(f"Processing utterance. Total frames: {self.frames_in_buffer}, Speech frames: {self.current_speech_frames}. Audio length: {len(combined_audio)} samples.")
                                self.callback(combined_audio)
                        # else:
                            # print(f"Speech too short ({self.current_speech_frames * self.frame_duration_ms}ms), discarding. Min required: {self.speech_min_duration_ms}ms.")
                        
                        # Reset for next utterance
                        self.is_speech_active = False
                        self.audio_buffer = []
                        self.current_speech_frames = 0
                        self.current_silence_frames = 0
                        self.frames_in_buffer = 0
                    
                    # Check for forced break due to maximum speech length
                    elif self.frames_in_buffer >= self.max_speech_frames_before_force_break:
                        # print(f"Max speech duration reached ({self.frames_in_buffer * self.frame_duration_ms}ms), forcing processing.")
                        if self.callback and self.audio_buffer: # Ensure there's something to send
                            combined_audio = np.concatenate(self.audio_buffer)
                            # print(f"Force processing max length. Total frames: {self.frames_in_buffer}, Speech frames: {self.current_speech_frames}. Audio length: {len(combined_audio)} samples.")
                            self.callback(combined_audio)
                        
                        # Reset
                        self.is_speech_active = False
                        self.audio_buffer = []
                        self.current_speech_frames = 0
                        self.current_silence_frames = 0
                        self.frames_in_buffer = 0

                else: # Not self.is_speech_active
                    if is_speech_current_frame:
                        # Start of a new speech segment
                        # print("Speech start detected.")
                        self.is_speech_active = True
                        self.audio_buffer = [audio_frame_np] # Start new buffer
                        self.current_speech_frames = 1
                        self.frames_in_buffer = 1
                        self.current_silence_frames = 0 
                    # else:
                        # Silence continues, and no speech was active. Do nothing.
                        # pass
            
            except queue.Empty:
                # Queue was empty, means no new audio data.
                # This is expected if timeout is used in queue.get()
                if not self.is_listening:
                    break # Exit if stop() was called during timeout
                # If speech was active and queue is empty for a while, it might also mean end of speech.
                # This case is implicitly handled if VAD stops sending speech frames, leading to silence count increase.
                continue 
            except Exception as e:
                print(f"Critical error in listen loop: {e}")
                import traceback
                traceback.print_exc()
                # Consider if loop should break or try to recover. For now, it continues.
                if not self.is_listening: 
                    break
                time.sleep(0.05) # Brief pause to prevent tight loop on persistent error
        
        print("Listen loop ended.")
                
    def get_audio_devices(self):
        """Get available audio input devices"""
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                # device['max_input_channels'] might be a string in some sounddevice versions/systems
                try:
                    max_input_channels = int(device['max_input_channels'])
                except ValueError:
                    max_input_channels = 0
                
                if max_input_channels > 0:
                    input_devices.append({
                        'index': i, 
                        'name': device['name'], 
                        'channels': max_input_channels,
                        'hostapi_name': sd.query_hostapis(device['hostapi'])['name'] # More detailed device info
                    })
            return input_devices
        except Exception as e:
            print(f"Error querying audio devices: {e}")
            return []

# Example of how to add sys import if needed for stderr logging, but it's usually available
# import sys