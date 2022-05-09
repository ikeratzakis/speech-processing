"""
3.19 - Apply low-pass butterworth filter to speech signal and save it.
"""
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

audio_file = wavfile.read('../data/chapter_10/test_16k.wav')
audio = audio_file[1]

fs = 16000
cutoff = 3200
nyq = 0.5 * fs
N = 6
fc = cutoff / nyq
b, a = signal.butter(N, fc)
filtered_audio = signal.lfilter(b, a, audio, axis=0)
print('Original audio samples: ', audio)
print('Filtered audio samples: ',filtered_audio)
wavfile.write('../filtered_test_16k.wav', rate=16000, data=filtered_audio / 1000)

plt.plot(audio)
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.show()
