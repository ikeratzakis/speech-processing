"""
3.21 - Remove 60Hz noise by applying a rolling mean filter.
"""
import numpy as np
from scipy.io import wavfile

audio_file = wavfile.read('../data/chapter_10/test_16k.wav')
audio = audio_file[1]
win_length = 5
filtered_audio = np.convolve(audio, np.ones(win_length), 'valid') / win_length
wavfile.write('mean_16k.wav', rate=16000, data=filtered_audio)
