"""
6.17 - Calculate short time measures (waveform, energy, meter, zero crossing rate) and plot them.
Also solves 6.18: plot for different window lengths
"""
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate the short time energy for given windowLength and step.
def short_time_energy(signal, window_length, step, magnitude):
    cur_pos = 0
    l = len(signal)
    energy = []
    # frameCounter = 1
    while cur_pos + window_length - 1 <= l:
        window = signal[cur_pos:cur_pos + window_length - 1]
        window_energy = (1 / window_length) * sum(
            [x ** 2 for x in window] if magnitude is False else [x for x in window])
        energy.append(window_energy)
        cur_pos += step
        # frameCounter +=1
    # print(len(energy))

    return energy


# Function to calculate the short time zero crossing rate for given windowLength and step.

def short_time_zcr(signal, window_length, step):
    cur_pos = 0
    l = len(signal)
    zcr = []
    # frameCounter = 1
    while cur_pos + window_length - 1 <= l:
        window = signal[cur_pos:cur_pos + window_length - 1]

        temp = 0
        for i in range(1, window_length - 1):
            if window[i] >= 0 > window[i - 1]:
                temp += 2
            if window[i] < 0 <= window[i - 1]:
                temp += 2
        window_zcr = (1 / (2 * window_length)) * temp
        zcr.append(window_zcr)
        cur_pos += step
        # frameCounter +=1
    return zcr


def step_length(sample_rate):
    # windowLength = int((len(inputSignal))/((len(inputSignal)/sampleRate)*100))
    step = int(10 * sample_rate / 1000)
    return step


def main():
    audio_file = wavfile.read('../data/chapter_10/test_16k.wav')
    audio = list(audio_file[1] / 10)
    sample_rate = 16000
    # Calculate measurements
    window_length = int((len(audio)) / ((len(audio) / sample_rate) * 100))
    step = step_length(sample_rate=sample_rate)
    energy = short_time_energy(audio, window_length=window_length, step=step, magnitude=False)
    magnitude = short_time_energy(audio, window_length=window_length, step=step, magnitude=True)
    zcr = short_time_zcr(audio, window_length=window_length, step=step)
    norm_energy = [10 * np.log10(x) for x in energy]
    norm_magnitude = [10 * np.log10(abs(x)) for x in magnitude]
    norm_zcr = [100 * x for x in zcr]


    plt.subplot(221)
    plt.title('Waveform')
    plt.plot(audio)
    plt.xlabel('Sample')

    plt.subplot(222)
    plt.plot(energy)
    plt.title('Εnergy')
    plt.xlabel('Frame number')
    plt.ylabel('dB')

    plt.subplot(223)
    plt.plot(norm_magnitude)
    plt.title('Magnitude')
    plt.ylabel('dB')
    plt.xlabel('Frame number')

    plt.subplot(224)
    plt.plot(norm_zcr)
    plt.title('Zero crossing rate')
    plt.xlabel('Frame number')
    plt.ylabel('ZCR per 10ms')
    plt.show()

    # 6.18 starts here
    lengths = [51, 101, 201, 401]
    energy_list = []
    zcr_list = []
    magnitude_list = []
    for l in lengths:
        energy_list.append(short_time_energy(signal=audio, window_length=l, step=step, magnitude=False))
        zcr_list.append(short_time_zcr(signal=audio, window_length=l, step=step))
        magnitude_list.append(short_time_energy(signal=audio, window_length=l, step=step, magnitude=True))

    plt.subplot(221)
    plt.plot(energy_list[0])
    plt.title('Window length 51')

    plt.subplot(222)
    plt.plot(energy_list[1])
    plt.title('Window length 101')

    plt.subplot(223)
    plt.plot(energy_list[2])
    plt.title('Window length 201')

    plt.subplot(224)
    plt.plot(energy_list[3])
    plt.title('Window length 401')
    plt.show()


main()
