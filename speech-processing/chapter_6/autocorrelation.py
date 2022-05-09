"""
6.20 - Calculate autocorrelation
"""
from scipy.io import wavfile
import librosa
import numpy as np

audio_file = wavfile.read('../data/chapter_10/test_16k.wav')
audio = list(audio_file[1] / 10)
k = 10

# Librosa autocorrelation implementation
acorr = librosa.core.autocorrelate(np.array(audio), max_size=len(audio))
print(acorr)
# Modified autocorrelation
modified_corr = np.correlate(audio, audio, 'full').tolist()
print(modified_corr)
