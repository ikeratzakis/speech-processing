"""10.8-FUNDAMENTAL FREQUENCY CALCULATOR BASED ON HARMONIC PRODUCT SPECTRUM. Uses some ideas and code (modified) based on
https://gist.github.com/fasiha/957035272009eb1c9eb370936a6af2eb"""
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy.signal import filtfilt
from scipy.signal.windows import hamming

# Read and convert sampling rate
print('Reading audio file and converting sample rate to 8000...')
gender = 'male'
new_fs = 10000
signal, fs = librosa.load('../data/chapter_10/ah_lrr.wav', sr=new_fs)

# Filter signal with a simple 1- z^(-1) FIR filter and write filtered signal to file
filtered_signal = filtfilt(b=[1, -1], a=1, x=signal)
sf.write('filtered_ah_lrr.wav', data=filtered_signal, samplerate=10000)

# Parameters for Fourier analysis
nfft = 4000  # STFT size
window_length = 400
step_length = 100
K = 10
Coffset = 275


def hps(x, num_prod, nfft, fs):
    # Evaluate FFT. f is the frequencies corresponding to the spectrum xf
    f = np.arange(nfft) / nfft
    xf = fft.fft(x, nfft)
    # Keep magnitude of spectrum at specific frequencies
    xf = np.abs(xf[f < 0.5])
    f = f[f < 0.5]
    N = f.size
    # Downsample-multiply
    smallest_length = int(np.ceil(N / num_prod))
    y = xf[:smallest_length].copy()
    for i in range(2, num_prod + 1):
        y *= xf[::i][:smallest_length]
    f = f[:smallest_length] * fs
    return y, f


# Calculate hps
Nfft = int(4 * 2 ** np.ceil(np.log2(window_length)))

plt.plot(signal), plt.xlabel('Time'), plt.ylabel('Ampltitude'), plt.title('Signal filtering')
plt.plot(filtered_signal)
plt.show()
