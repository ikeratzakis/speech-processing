"""Annotate words on signal waveform"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read

# Read wav file and convert samples to time
data = read('we were away a year ago_lrr.wav')
signal = data[1]
fs = data[0]
time = np.linspace(0, len(signal) / fs, num=len(signal))
# Plot and annotate the words
x_positions = [0, 6100, 10600, 17000, 18000, 24000, 30300]
for x in x_positions:
    plt.axvline(x)
plt.title('We were away a year ago')
plt.text(3500, 12500, 'We')
plt.text(7500, 15000, 'were')
plt.text(13500, 23000, 'away')
plt.text(17500, 8000, 'a')
plt.text(20000, 13000, 'year')
plt.text(26000, 13000, 'ago')
plt.plot(signal)
plt.show()
