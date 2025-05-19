import edge_tts
import asyncio
import tempfile
import os
from typing import Optional, Dict, Any
import json

class VoiceSynthesizer:
    def __init__(self, config: dict):
        self.config = config
        self.engine = config['tts']['default_engine']
        self.voice = config['tts']['default_voice']
        self.rate = config['tts']['rate']
        self.volume = config['tts']['volume']
        self.available_voices = {}
        
    async def initialize(self):
        """初始化语音合成器"""
        if self.engine == "edge":
            await self._load_edge_voices()
            
    async def _load_edge_voices(self):
        """加载 Edge TTS 可用声音列表"""
        try:
            voices = await edge_tts.list_voices()
            self.available_voices = {
                voice["ShortName"]: voice
                for voice in voices
                if voice["Gender"] == "Male"  # 默认只加载男声
            }
        except Exception as e:
            print(f"加载声音列表失败: {e}")
            
    async def speak(self, text: str) -> Optional[str]:
        """合成语音并返回音频文件路径"""
        if not text:
            return None
            
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_path = temp_file.name
                
            # 合成语音
            if self.engine == "edge":
                # 根据 edge-tts 的要求，速率应该是带符号的整数百分比
                # 例如："+0%", "+10%", "-10%"
                rate_str = f"{int(self.rate*100):+d}%"
                # 音量也需要是带符号的整数百分比
                volume_str = f"{int(self.volume*100):+d}%"
                communicate = edge_tts.Communicate(
                    text,
                    self.voice,
                    rate=rate_str,
                    volume=volume_str
                )
                await communicate.save(temp_path)
                
            return temp_path
            
        except Exception as e:
            print(f"语音合成失败: {e}")
            return None
            
    def get_available_voices(self) -> Dict[str, Any]:
        """获取可用的声音列表"""
        return self.available_voices
        
    def set_voice(self, voice_id: str):
        """设置声音"""
        if voice_id in self.available_voices:
            self.voice = voice_id
            
    def set_rate(self, rate: float):
        """设置语速"""
        self.rate = rate
        
    def set_volume(self, volume: float):
        """设置音量"""
        self.volume = volume
        
    def get_current_settings(self) -> Dict[str, Any]:
        """获取当前设置"""
        return {
            "engine": self.engine,
            "voice": self.voice,
            "rate": self.rate,
            "volume": self.volume
        } 