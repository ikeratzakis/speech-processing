"""FILTER AN AUDIO SIGNAL AND DEMONSTRATE THE EFFECTS OF FILTERING ON THE ZERO CROSSING RATE"""
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from short_time_analysis import zero_crossing_rate
from scipy.signal import butter, sosfilt, medfilt

# Read audio file and format time properly
signal, fs = sf.read('../data/chapter_10/test_16k.wav')
time = np.linspace(0, len(signal) / fs, num=len(signal))
# 10 ms frame duration and 5 ms step length , calculate zero crossing rate on original signal
window_length = int(10 * fs / 1000)
step_length = int(5 * fs / 1000)
print('Sample rate:', fs, 'Signal data:', signal)
print('Window length:', window_length, 'Step length:', step_length)
zcr = zero_crossing_rate(signal, window_length, step_length)

# Apply lowpass filter to signal and calculate zero crossing rate on filtered signal
lowpass_order = 41
nyq = 0.5 * fs
print(nyq)
cutoff = 1600  # Hz
sos = butter(N=lowpass_order, Wn=cutoff / nyq, btype='low', output='sos')
lowpass_signal = sosfilt(sos, signal)
zcr_lowpass = zero_crossing_rate(lowpass_signal, window_length, step_length)

# Apply double median filter and calculate zcr again
medfilt_signal = medfilt(signal, kernel_size=7)
medfilt_signal = medfilt(medfilt_signal, kernel_size=5)
zcr_medfilt = zero_crossing_rate(medfilt_signal, window_length, step_length)

# Plot results
fig, axes = plt.subplots(1, 2)
axes[0].plot(time, signal)
axes[0].set_title('Signal')
axes[0].set_xlabel('Seconds')
axes[1].plot(zcr, '-b', label='Unfiltered')
axes[1].set_title('Zero crossing rates')
axes[1].plot(zcr_lowpass, '-r', label='Lowpass')
axes[1].plot(zcr_medfilt, '-g', label='Medfilt')
axes[1].set_xlabel('Frame number')
axes[1].legend(loc='upper left')
plt.show()
