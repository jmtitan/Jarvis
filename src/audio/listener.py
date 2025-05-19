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
        self.vad = webrtcvad.Vad(2)  # 设置 VAD 灵敏度
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.callback = None
        assert config['audio']['frame_ms'] in (10, 20, 30), "frame_ms must be 10/20/30 ms"
        assert config['audio']['sample_rate'] in (8000, 16000, 32000, 48000), "sample_rate must be 8000/16000/32000/48000"
        
        # 音频参数
        self.sample_rate = config['audio']['sample_rate']
        self.channels = config['audio']['channels']
        self.chunk_size = int(self.sample_rate * config['audio']['frame_ms'] / 1000) 
        self.device_index = config['audio']['device_index']
        
        # 语音检测参数
        self.min_silence_frames = int(0.5 * 1000 / config['audio']['frame_ms'])  # 0.5秒静音判定为语音结束
        self.max_speech_frames = int(10 * 1000 / config['audio']['frame_ms'])  # 最大收集10秒语音
        
        # 缓冲区
        self.audio_buffer = []
        self.silence_count = 0
        self.speech_count = 0
        self.is_speech_active = False
        
    def start(self, callback: Optional[Callable] = None):
        """启动音频监听"""
        if self.is_listening:
            return
            
        self.callback = callback
        self.is_listening = True
        
        # 打开音频流
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype="int16",
            channels=self.channels,
            # device=self.device_index,
            callback=self._audio_callback
        )
        
        # 显式启动流
        self.stream = self.stream.__enter__()
        
        # 启动监听线程
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        print("开始音频监听...")
        
    def _audio_callback(self, indata, frames, time_info, status):
        """音频回调函数"""
        if status:
            print(f"音频状态: {status}")
        self.audio_queue.put(bytes(indata))
        
    def stop(self):
        """停止音频监听"""
        self.is_listening = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
    def _listen_loop(self):
        """音频监听循环"""
        while self.is_listening:
            try:
                if not self.audio_queue.empty():
                    frame = self.audio_queue.get()
                    audio_data = np.frombuffer(frame, dtype=np.int16)
                    
                    # 检测是否含有语音
                    try:
                        speech = self.vad.is_speech(frame, self.sample_rate)
                    except Exception as e:
                        print(f"VAD 检测错误: {e}")
                        speech = False
                    
                    # 语音状态管理
                    if speech:
                        # 检测到语音
                        if not self.is_speech_active:
                            # 语音开始
                            self.is_speech_active = True
                            self.audio_buffer = []  # 清空缓冲区，准备收集新的语音
                            print("检测到语音开始")
                        
                        # 添加到缓冲区
                        self.audio_buffer.append(audio_data)
                        self.speech_count += 1
                        self.silence_count = 0
                        
                        # 如果语音太长，强制处理当前缓冲区
                        if self.speech_count >= self.max_speech_frames:
                            if self.callback and self.audio_buffer:
                                combined_audio = np.concatenate(self.audio_buffer)
                                print(f"语音太长，强制处理，数据长度: {len(combined_audio)}")
                                self.callback(combined_audio)
                            self.audio_buffer = []
                            self.speech_count = 0
                            self.is_speech_active = False
                    else:
                        # 未检测到语音
                        if self.is_speech_active:
                            # 之前是语音状态，现在是静音
                            self.silence_count += 1
                            self.audio_buffer.append(audio_data)  # 继续添加一小段静音
                            
                            # 如果静音足够长，认为语音结束
                            if self.silence_count >= self.min_silence_frames:
                                if self.callback and self.audio_buffer:
                                    combined_audio = np.concatenate(self.audio_buffer)
                                    print(f"检测到语音结束，数据长度: {len(combined_audio)}")
                                    self.callback(combined_audio)
                                self.audio_buffer = []
                                self.silence_count = 0
                                self.speech_count = 0
                                self.is_speech_active = False
            except Exception as e:
                print(f"音频监听错误: {e}")
                time.sleep(0.1)
                
    def get_audio_devices(self):
        """获取可用的音频输入设备"""
        devices = sd.query_devices()
        return [{'index': i, 'name': device['name'], 'channels': device['max_input_channels']} for i, device in enumerate(devices) if device['max_input_channels'] > 0] 