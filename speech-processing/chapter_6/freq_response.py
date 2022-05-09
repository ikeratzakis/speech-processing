"""
6.16 - Calculate frequency response for typical short term windows for speech processing: Rectangular, triangular, Hann,
Hamming, Blackman. The process is similar for all windows, only hamming is displayed here. 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, fftshift


L = 51  # Window length
# Calculate windows
hamm = np.hamming(L)
black = np.blackman(L)
tri = signal.triang(L)
hann = np.hanning(L)
box = signal.boxcar(L)

# Calculate frequency responses
hamm_fft = fft(hamm, 2048) / 25.5
hamm_mag = np.abs(fftshift(hamm_fft))
hamm_freq = np.linspace(-0.5, 0.5, len(hamm_fft))
hamm_response = 20 * np.log10(hamm_mag)
hamm_response = np.clip(hamm_response, -100, 100)

plt.subplot(2, 1, 1)
plt.title('Hamming window')
plt.plot(hamm)
plt.ylabel('Amplitude')
plt.xlabel('Sample')

plt.subplot(2, 1, 2)
plt.plot(hamm_freq, hamm_response)
plt.axis('tight')
plt.ylabel('Normalized magnitude [dB]')
plt.xlabel('Normalized frequency')
plt.show()
