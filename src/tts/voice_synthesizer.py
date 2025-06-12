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
        """Initialize voice synthesizer"""
        if self.engine == "edge":
            await self._load_edge_voices()
            
    async def _load_edge_voices(self):
        """Load available Edge TTS voices"""
        try:
            voices = await edge_tts.list_voices()
            
            # Add recommended female voices (British English Neural voices)
            recommended_female_voices = {
                "en-GB-LibbyNeural": "Clear, natural, gentle - most commonly used",
                "en-GB-SoniaNeural": "Younger, lively - suitable for casual conversation", 
                "en-GB-MaisieNeural": "Sweet voice, soft tone, approachable",
                "en-GB-HollieNeural": "Warm, expressive - suitable for formal contexts",
                "en-GB-LunaNeural": "Neutral formal, clear expression, mature"
            }
            
            # Load all available voices, prioritizing recommended female voices
            self.available_voices = {}
            
            for voice in voices:
                voice_name = voice["ShortName"]
                
                # Priority add recommended female voices
                if voice_name in recommended_female_voices:
                    voice_info = voice.copy()
                    voice_info["Description"] = recommended_female_voices[voice_name]
                    voice_info["Recommended"] = True
                    self.available_voices[voice_name] = voice_info
                    print(f"Added recommended female voice: {voice_name} - {recommended_female_voices[voice_name]}")
                
                # Add other English voices (both male and female)
                elif voice_name.startswith(("en-US", "en-GB", "en-AU", "en-CA")):
                    voice_info = voice.copy()
                    voice_info["Recommended"] = False
                    self.available_voices[voice_name] = voice_info
            
            print(f"Loaded {len(self.available_voices)} voices, including {len(recommended_female_voices)} recommended female voices")
            
        except Exception as e:
            print(f"Failed to load voice list: {e}")
            # If loading fails, provide fallback recommended female voice list
            self.available_voices = {
                "en-GB-LibbyNeural": {
                    "ShortName": "en-GB-LibbyNeural",
                    "Gender": "Female", 
                    "Locale": "en-GB",
                    "Description": "Clear, natural, gentle - most commonly used",
                    "Recommended": True
                },
                "en-GB-SoniaNeural": {
                    "ShortName": "en-GB-SoniaNeural", 
                    "Gender": "Female",
                    "Locale": "en-GB", 
                    "Description": "Younger, lively - suitable for casual conversation",
                    "Recommended": True
                },
                "en-GB-MaisieNeural": {
                    "ShortName": "en-GB-MaisieNeural",
                    "Gender": "Female",
                    "Locale": "en-GB",
                    "Description": "Sweet voice, soft tone, approachable", 
                    "Recommended": True
                },
                "en-GB-HollieNeural": {
                    "ShortName": "en-GB-HollieNeural",
                    "Gender": "Female",
                    "Locale": "en-GB", 
                    "Description": "Warm, expressive - suitable for formal contexts",
                    "Recommended": True
                },
                "en-GB-LunaNeural": {
                    "ShortName": "en-GB-LunaNeural",
                    "Gender": "Female", 
                    "Locale": "en-GB",
                    "Description": "Neutral formal, clear expression, mature",
                    "Recommended": True
                }
            }
            
    async def speak(self, text: str) -> Optional[str]:
        """Synthesize speech and return audio file path"""
        if not text:
            return None
            
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_path = temp_file.name
                
            # Synthesize speech
            if self.engine == "edge":
                # According to edge-tts requirements, rate should be signed integer percentage
                # For example: "+0%", "+10%", "-10%"
                rate_str = f"{int(self.rate*100):+d}%"
                # Volume also needs to be signed integer percentage
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
            print(f"Speech synthesis failed: {e}")
            return None
            
    def get_available_voices(self) -> Dict[str, Any]:
        """Get available voice list"""
        return self.available_voices
    
    def get_recommended_female_voices(self) -> Dict[str, Any]:
        """Get recommended female voice list"""
        return {
            name: info for name, info in self.available_voices.items() 
            if info.get("Recommended", False) and info.get("Gender") == "Female"
        }
    
    def list_voices_by_category(self) -> Dict[str, Dict[str, Any]]:
        """Display voices grouped by category"""
        categories = {
            "Recommended Female": {},
            "British English": {},
            "American English": {},
            "Other English": {}
        }
        
        for name, info in self.available_voices.items():
            if info.get("Recommended", False):
                categories["Recommended Female"][name] = info
            elif name.startswith("en-GB"):
                categories["British English"][name] = info
            elif name.startswith("en-US"):
                categories["American English"][name] = info
            else:
                categories["Other English"][name] = info
        
        return categories
        
    def set_voice(self, voice_id: str):
        """Set voice"""
        if voice_id in self.available_voices:
            self.voice = voice_id
            
    def set_rate(self, rate: float):
        """Set speech rate"""
        self.rate = rate
        
    def set_volume(self, volume: float):
        """Set volume"""
        self.volume = volume
        
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        return {
            "engine": self.engine,
            "voice": self.voice,
            "rate": self.rate,
            "volume": self.volume
        } 