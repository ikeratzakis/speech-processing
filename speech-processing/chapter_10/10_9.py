import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from scipy.signal import stft

# Read and convert sampling rate
print('Reading audio file and converting sample rate to 8000...')
gender = 'male'
new_fs = 10000
signal, fs = librosa.load('s6.wav', sr=new_fs)

# Filter signal with a simple 1- z^(-1) FIR filter and write filtered signal to file
filtered_signal = filtfilt(b=[1, -1], a=1, x=signal)
sf.write('filtered_s6.wav', data=filtered_signal, samplerate=10000)

# Parameters for cepstral analysis
nfft = 4000  # STFT size
window_length = 400
step_length = 100
pthr1 = 4
nlow = 40
nhigh = 167

# Calculate cepstrum based on stft
f, t, zxxx = stft(filtered_signal, nperseg=window_length, noverlap=window_length - step_length, nfft=nfft)

