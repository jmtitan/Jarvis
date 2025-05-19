import asyncio
import yaml
import os
import sys
from pathlib import Path
import subprocess

# 添加项目根目录到 Python 路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.tts.voice_synthesizer import VoiceSynthesizer

# 导入playsound库
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
    print("找到 playsound 库")
except ImportError:
    PLAYSOUND_AVAILABLE = False
    print("playsound 不可用，请使用 pip install playsound==1.2.2 安装")
    sys.exit(1)  # 如果playsound不可用，则退出程序

async def play_audio(file_path):
    """播放音频文件的函数"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
        
    try:
        print("正在播放语音...")
        # 使用playsound播放音频
        print("使用方式: playsound 库播放音频")
        playsound(file_path)
                
        # 等待足够的时间让音频播放完成
        await asyncio.sleep(1)  # playsound是阻塞的，所以不需要长时间等待
    except Exception as e:
        print(f"播放失败: {e}")

async def test_voice_synthesizer():
    # 加载配置文件
    config_path = root_dir / 'config.yaml'
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，请确保配置文件存在。")
        return
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化语音合成器
    print("初始化语音合成器...")
    voice_synthesizer = VoiceSynthesizer(config)
    await voice_synthesizer.initialize()
    
    # 打印当前设置
    settings = voice_synthesizer.get_current_settings()
    print("\n当前 TTS 配置:")
    print(f"引擎: {settings['engine']}")
    print(f"声音: {settings['voice']}")
    print(f"语速: {settings['rate']}")
    print(f"音量: {settings['volume']}")
    
    # 获取可用声音列表
    voices = voice_synthesizer.get_available_voices()
    print(f"\n可用声音数量: {len(voices)}")
    print("部分可用声音:")
    count = 0
    for voice_id, voice_info in voices.items():
        print(f"  {voice_id}: {voice_info.get('FriendlyName', 'Unknown')}")
        count += 1
        if count >= 5:  # 只打印前5个声音
            break
    
    # 测试不同文本的合成
    test_texts = [
        # "你好，我是你的智能语音助手。",
        "Hello, I am your intelligent voice assistant.",
        # "今天的天气怎么样？",
        "What is the weather like today?"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n测试合成 {i+1}: {text}")
        try:
            audio_path = await voice_synthesizer.speak(text)
            if audio_path:
                print(f"语音合成成功，文件保存在: {audio_path}")
                print(f"文件类型: {os.path.splitext(audio_path)[1]}")  # 显示文件扩展名
                
                # 播放合成的语音
                await play_audio(audio_path)
            else:
                print("语音合成失败!")
        except Exception as e:
            print(f"语音合成异常: {e}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    try:
        asyncio.run(test_voice_synthesizer())
    except Exception as e:
        print(f"主程序发生错误: {e}") 