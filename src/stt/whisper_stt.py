import subprocess
import numpy as np
from typing import Optional, Callable
import threading
import queue
import time
import os
import tempfile
import wave

class WhisperSTT:
    def __init__(self, config: dict):
        self.config = config
        self.is_processing = False
        self.audio_queue = queue.Queue()
        self.callback = None
        
        # 设置 whisper.exe 路径
        self.whisper_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                       'whisper_bin', 'main.exe')
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     'models', f"{self.config['stt']['model']}.bin")
        
        if not os.path.exists(self.whisper_path):
            raise FileNotFoundError(f"找不到 whisper 可执行文件: {self.whisper_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到模型文件: {self.model_path}")
            
    def start(self, callback: Optional[Callable] = None):
        """启动语音识别"""
        if self.is_processing:
            return
            
        self.callback = callback
        self.is_processing = True
        
        # 启动处理线程
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()

        print("语音识别启动完成")
        
    def stop(self):
        """停止语音识别"""
        self.is_processing = False
        
    def process_audio(self, audio_data: np.ndarray):
        """处理音频数据"""
        print("添加音频数据...")
        # print("音频数据长度:", len(audio_data))
        # print("音频数据内容:", audio_data)
        self.audio_queue.put(audio_data)
        
    def _save_audio_to_wav(self, audio_data: np.ndarray) -> str:
        """将音频数据保存为临时 WAV 文件"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(self.config['audio']['channels'])
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.config['audio']['sample_rate'])
                wav_file.writeframes(audio_data.tobytes())
            return temp_file.name
            
    def _process_loop(self):
        """音频处理循环"""
        while self.is_processing:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    
                    # 保存音频到临时文件
                    wav_path = self._save_audio_to_wav(audio_data)
                    try:
                        # 调用 whisper.cpp
                        result = subprocess.run([
                            self.whisper_path,
                            '--model', self.model_path,
                            '--language', self.config['stt']['language'],
                            '-otxt',  # 输出为文本
                            '-f', wav_path  # 输入文件
                        ], capture_output=True, text=True)
                        print("whisper.cpp 调用完成")
                        print('result.stdout:', result.stdout)
                        if result.returncode == 0 and result.stdout:
                            if self.callback:
                                self.callback(result.stdout.strip())
                                
                    finally:
                        # 清理临时文件
                        try:
                            os.unlink(wav_path)
                        except:
                            pass
                            
            except Exception as e:
                print(f"语音识别错误: {e}")
                time.sleep(0.1)
                
    def get_available_models(self):
        """获取可用的 Whisper 模型列表"""
        return [
            "tiny.en",
            "base.en",
            "small.en",
            "medium.en",
            "large"
        ] 