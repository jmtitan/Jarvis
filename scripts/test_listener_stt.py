import asyncio
import yaml
import os
import time
import numpy as np
import sounddevice as sd
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from src.audio.listener import AudioListener
from src.stt.whisper_stt import WhisperSTT

async def test_listener_stt():
    # 加载配置文件
    config_path = root_dir / 'config.yaml'
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，请确保配置文件存在。")
        return
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确保配置文件包含必要的参数
    if 'frame_ms' not in config['audio']:
        print("配置文件中缺少 frame_ms 参数，正在添加默认值 30ms")
        config['audio']['frame_ms'] = 30
    
    # 初始化 AudioListener
    print("初始化 AudioListener...")
    audio_listener = AudioListener(config)
    
    # 获取可用的音频输入设备
    devices = audio_listener.get_audio_devices()
    print("可用的音频输入设备:")
    for device in devices:
        print(f"设备索引: {device['index']}, 名称: {device['name']}, 通道数: {device['channels']}")
    
    # 打印当前配置
    print("\n当前音频配置:")
    print(f"采样率: {audio_listener.sample_rate}")
    print(f"通道数: {audio_listener.channels}")
    print(f"块大小: {audio_listener.chunk_size}")
    print(f"帧长度: {config['audio']['frame_ms']}ms")
    print(f"设备索引: {audio_listener.device_index}")
    
    # 初始化 WhisperSTT
    print("初始化 WhisperSTT...")
    stt = WhisperSTT(config)
    
    # 定义回调函数，用于处理音频数据
    def on_audio_data(audio_data):
        # print("收到音频数据，数据长度:", len(audio_data))
        # 计算音频数据的均方根值，用于观察音量
        # rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32) / 32768.0)))
        # print(f"音频音量: {rms:.6f}")
        stt.process_audio(audio_data)
    
    # 定义回调函数，用于处理识别结果
    def on_stt_result(result):
        print("识别结果:", result)
    
    try:
        # 启动音频监听
        print("开始音频监听...")
        audio_listener.start(on_audio_data)
        
        # 启动语音识别
        print("开始语音识别...")
        stt.start(on_stt_result)
        
        # 保持程序运行，直到用户手动停止
        print("音频监听和语音识别已启动，按 Ctrl+C 停止...")
        while True:
            await asyncio.sleep(1)
            print("程序正在运行...")
    except KeyboardInterrupt:
        print("收到键盘中断，停止音频监听和语音识别...")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保停止监听和语音识别
        print("停止音频监听和语音识别...")
        audio_listener.stop()
        stt.stop()
        print("音频监听和语音识别已停止")

if __name__ == "__main__":
    try:
        asyncio.run(test_listener_stt())
    except Exception as e:
        print(f"主程序发生错误: {e}") 