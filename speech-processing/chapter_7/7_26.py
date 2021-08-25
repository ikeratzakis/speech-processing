"""WIDE AND NARROW BAND SPECTROGRAM, 7.27 is almost the same"""
import matplotlib.pyplot as plt  # Drawing
import librosa.core as  lc  # Calculate the use of stft
import numpy as  np  # Used some of the tool functions
import librosa.display  # Picture sound spectrogram
from scipy.io import wavfile  # Get the sampling rate of the wav file

path = "s5.wav"
fs, y_ = wavfile.read(path)  # Read the sample rate of the file
fs = fs
n_fft = 1024  # FFTLength
y, sr = librosa.load(path, sr=fs)

# Get broadband spectrogram
mag = np.abs(lc.stft(y, n_fft=n_fft, hop_length=10, win_length=40,
                     window='hamming'))  # Perform short-time Fourier transform and get the amplitude
D = librosa.amplitude_to_db(mag, ref=np.max)  # Amplitude converted to db unit
librosa.display.specshow(D, sr=fs, hop_length=10, x_axis='s', y_axis='linear')  #
plt.figure()
#plt.colorbar(format='%+2.0f dB')
plt.title('broadband spectrogram')
plt.savefig('broader.png')
plt.show()

# Get narrowband spectrogram
mag1 = np.abs(lc.stft(y, n_fft=n_fft, hop_length=100, win_length=400, window='hamming'))
mag1_log = 20 * np.log(mag1)
D1 = librosa.amplitude_to_db(mag1, ref=np.max)
librosa.display.specshow(D1, sr=fs, hop_length=100, x_axis='s', y_axis='linear')
plt.figure()
#plt.colorbar(format='%+2.0f dB')
plt.title('narrowband spectrogram')
plt.savefig('narrowband.png')
plt.show()
