#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jarvis Female Voice Demo Script
Demonstrates all recommended British English Neural female voices
"""

import asyncio
from src.tts.voice_synthesizer import VoiceSynthesizer
import yaml
import os
import time

# Demo text
DEMO_TEXT = "Hello! I'm one of Jarvis's recommended female voices. I speak with a clear, natural British accent."

async def demo_female_voices():
    """Demonstrate all recommended female voices"""
    print("🎤 Jarvis Female Voice Demo")
    print("=" * 50)
    
    # Load configuration
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize voice synthesizer
    tts = VoiceSynthesizer(config)
    await tts.initialize()
    
    # Get recommended female voices
    recommended_voices = tts.get_recommended_female_voices()
    
    print(f"📊 Found {len(recommended_voices)} recommended female voices:")
    for name, info in recommended_voices.items():
        print(f"  ✅ {name}: {info.get('Description', 'No description')}")
    
    print("\n🎵 Starting voice demo...")
    print("=" * 50)
    
    for i, (voice_name, voice_info) in enumerate(recommended_voices.items(), 1):
        print(f"\n{i}. Demonstrating: {voice_name}")
        print(f"   Description: {voice_info.get('Description', 'No description')}")
        
        # Set current voice
        tts.set_voice(voice_name)
        
        # Generate demo text
        demo_text = f"Hello! I'm {voice_name.replace('Neural', '').replace('en-GB-', '')}. {voice_info.get('Description', '')}"
        
        # Synthesize speech
        audio_path = await tts.speak(demo_text)
        
        if audio_path and os.path.exists(audio_path):
            print(f"   ▶️  Playing audio: {audio_path}")
            
            # Use pygame to play audio
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                pygame.mixer.quit()
                
            except Exception as e:
                print(f"   ❌ Playback failed: {e}")
                
            # Clean up temporary file
            try:
                os.unlink(audio_path)
            except:
                pass
                
        else:
            print(f"   ❌ Speech synthesis failed")
        
        # Pause briefly for clarity
        if i < len(recommended_voices):
            print("   ⏳ Waiting 2 seconds...")
            time.sleep(2)
    
    print("\n🎉 Demo complete!")
    print("💡 You can set your preferred voice in config.yaml under tts.default_voice")

if __name__ == "__main__":
    try:
        asyncio.run(demo_female_voices())
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user") 