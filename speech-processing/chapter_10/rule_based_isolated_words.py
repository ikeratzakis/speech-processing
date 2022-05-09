"""10.4-RULE BASED ISOLATED WORD SPEECH DETECTOR"""
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import butter, sosfilt
from scipy.io import wavfile
from short_time_analysis import zero_crossing_rate, short_time_energy

# Read and convert sampling rate
print('Reading audio file and converting sample rate to 8000...')
new_fs = 8000
signal, fs = librosa.load('../data/chapter_10/6A.waV', sr=new_fs)

# Apply bandpass filter to signal - Butterworth method
nyq = 0.5 * fs
filter_order = 20
lowcut = 100
highcut = 4000
low = lowcut / nyq  # Normalized frequencies
high = highcut / nyq
sos = butter(N=filter_order, Wn=[low, 0.99 * high], btype='band', output='sos')  # 0.99*high to avoid filter instability
filtered_signal = sosfilt(sos, signal)
wavfile.write('filtered_6a.wav', rate=fs, data=filtered_signal)

# Perform short time analysis (log energy, zero crossing rate)
window_length = int(40 * fs / 1000)
step_length = int(10 * fs / 1000)
energy = short_time_energy(filtered_signal, window_length, step_length)
zcr = zero_crossing_rate(filtered_signal, window_length, step_length)
zcr = [100 * x for x in zcr]
log_energy = [10 * np.log10(x) for x in energy]
log_energy -= max(log_energy)
# Necessary parameters
eavg = np.mean(log_energy[:10])
esig = np.std(log_energy[:10])
zcavg = np.mean(zcr[:10])
zcsig = np.std(zcr[:10])

# Algorithm for speech detection, first estimate some parameters based on the ones that were just calculated.
IF = 45  # Adjusted for non-hamming window, standard zero crossing rate threshold
IZCT = max(IF, zcavg + 3 * zcsig)  # Zero crossing rate threshold, varies
IMX = max(log_energy)
ITU = IMX - 20  # High threshold for log energy
ITL = max(eavg + 3 * esig, ITU - 10)  # Low threshold for log energy

B1 = None  # B1, B2, E1, E2 are the frames that estimate the start or stop of the speech region inside the signal.
B2 = None
E1 = None
E2 = None
for index, data in enumerate(log_energy):
    if data > ITL:
        # Found B1 frame
        B1 = index
        break

for index, data in enumerate(log_energy[::-1]):
    if data > ITL:
        # Found E1 frame
        E1 = len(log_energy) - index
        break

frame_counter = 0  # Counts number of consecutive frames in which zcr is above IZCT. First we search backwards (B1)
for data in log_energy[B1:B1 - 25:-1]:
    index, = np.where(log_energy == data)
    if zcr[index[0]] > IZCT:
        frame_counter += 1
    else:
        frame_counter = 0
    if frame_counter == 4:
        # Found B2 frame
        B2 = index
        break

frame_counter = 0
# Then a forward search based on E1
for data in log_energy[E1:E1 + 25]:
    index, = np.where(log_energy == data)
    if zcr[index[0]] > IZCT:
        frame_counter += 1
    else:
        frame_counter = 0
    if frame_counter == 4:
        # Found E2 frame
        E2 = index
        break

# Plot results
fig, axes = plt.subplots(2, 1)
axes[0].plot(log_energy)
if B1 is not None:
    axes[0].axvline(x=B1, color='red', linestyle='--')
    axes[1].axvline(x=B1, color='red', linestyle='--')
    axes[0].text(x=B1, y=max(log_energy) + 5, s='B1')
if E1 is not None:
    axes[0].axvline(x=E1, color='red', linestyle='--')
    axes[1].axvline(x=E1, color='red', linestyle='--')
    axes[0].text(x=E1, y=max(log_energy) + 5, s='E1')
axes[0].set_title('Log energy')
axes[1].plot(zcr)
if B2 is not None:
    axes[0].axvline(x=B2, color='green', linestyle='--')
    axes[1].axvline(x=B2, color='green', linestyle='--')
    axes[0].text(x=B2, y=max(log_energy) + 5, s='B2')
if E2 is not None:
    axes[0].axvline(x=E2, color='green', linestyle='--')
    axes[1].axvline(x=E2, color='green', linestyle='--')
    axes[0].text(x=E2, y=max(log_energy) + 5, s='E2')
axes[1].set_title('Zero crossing rate')
axes[1].set_xlabel('Frame number')
plt.show()
