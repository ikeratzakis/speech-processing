"""
Downsample a wav file.
"""
import librosa

new_rate = 8000
y, s = librosa.load('../data/chapter_10/test_16k.wav', sr=new_rate)
librosa.output.write_wav('test_8k.wav', y, sr=new_rate)
