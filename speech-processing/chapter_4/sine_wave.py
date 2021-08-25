"""
Generate 1 kHz sine wave.
"""
import numpy as np
from scipy.io import wavfile

t = 3  # in seconds
f = 1000  # The signal frequency
fs = 16000
samples = np.linspace(0, t, int(fs * t), endpoint=False)
signal = np.sin(2 * np.pi * f * samples)
signal *= 32767
signal = np.int16(signal)
wavfile.write('../sine.wav', rate=fs, data=signal)
