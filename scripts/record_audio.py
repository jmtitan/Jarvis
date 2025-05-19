import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000  # 采样率
seconds = 5  # 录音时长

print("开始录音，请说话...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()  # 等待录音结束
write('models/test.wav', fs, recording)
print("录音结束，文件已保存为 models/test.wav") 