"""7.25-RECTANGULAR AND HAMMING WINDOW APPLIED """
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows
from scipy.io import wavfile
from scipy.signal import stft

# Read file and calculate window length (in ms it's 40)
sample_rate, signal = wavfile.read('s5.wav')
time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
window_length = 20 * 8  # 8000 hz sampling rate * 40 ms per frame=> 320 samples for each frame
# Perform stft
f1, t1, Zxx1 = stft(signal[7000:], window='hamming', noverlap=None, fs=sample_rate, nperseg=window_length)
f2, t2, Zxx2 = stft(signal[7000:], window='boxcar', noverlap=None, fs=sample_rate, nperseg=window_length)
rect_signal = []
hamm_signal = []
# Loop and calculate the signal multipled by respective window
for i in range(7000, len(signal), window_length):
    if len(signal[i:i + window_length]) != window_length:
        break
    hamm_signal.append((signal[i:i + window_length] * np.hamming(window_length)))
    rect_signal.append((signal[i:i + window_length] * scipy.signal.windows.boxcar(window_length)))

fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(time, signal)
axs[0, 0].set_title('Original waveform')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].set_xlabel('Time [sec]')
axs[1, 0].plot(rect_signal)
axs[1, 0].set_title('Signal multipled by 160 points rectangular window')
axs[1, 0].set_ylabel('Filtered amplitude')
axs[2, 0].pcolormesh(t2, f2, np.abs(Zxx2), vmin=0, vmax=np.max(f2), shading='gouraud')
axs[2, 0].set_title('STFT Magnitude')
axs[2, 0].set_ylabel('Frequency [Hz]')
axs[2, 0].set_xlabel('Time [sec]')
axs[0, 1].plot(time, signal)
axs[0, 1].set_title('Original waveform')
axs[0, 1].set_ylabel('Amplitude')
axs[0, 1].set_xlabel('Time [sec]')
axs[1, 1].plot(hamm_signal)
axs[1, 1].set_title('Signal multipled by 160 points Hamming window')
axs[1, 1].set_ylabel('Filtered amplitude')
axs[2, 1].pcolormesh(t1, f1, np.abs(Zxx1), vmin=0, vmax=np.max(f1), shading='gouraud')
axs[2, 1].set_title('STFT Magnitude')
axs[2, 1].set_ylabel('Frequency [Hz]')
axs[2, 1].set_xlabel('Time [sec]')
plt.show()
