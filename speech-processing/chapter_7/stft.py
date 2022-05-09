"""7.24-STFT """
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

# Read file and calculate window length (in ms it's 40)
sample_rate, signal = wavfile.read('s5.wav')
time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
window_length = 40 * 8  # 8000 hz sampling rate * 40 ms per frame=> 320 samples for each frame
# Perform stft
f, t, Zxx = stft(signal[7000:], window='hamming', noverlap=None, fs=sample_rate, nperseg=320)

mult_signal = []

# Loop and calculate the signal multipled by hamming window
for i in range(7000, len(signal), window_length):
    if len(signal[i:i + window_length]) != 320:
        break
    mult_signal.append((signal[i:i + window_length] * np.hamming(320)))
print(len(mult_signal))
# Plot required figures
fig, axs = plt.subplots(3, 1)
axs[0].plot(time, signal)
axs[0].set_title('Original waveform')
axs[0].set_ylabel('Amplitude')
axs[0].set_xlabel('Time [sec]')
axs[1].plot(mult_signal)
axs[1].set_title('Signal multipled by 320 points Hamming window')
axs[1].set_ylabel('Filtered amplitude')
axs[2].pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(f), shading='gouraud')
axs[2].set_title('STFT Magnitude')
axs[2].set_ylabel('Frequency [Hz]')
axs[2].set_xlabel('Time [sec]')
plt.show()
