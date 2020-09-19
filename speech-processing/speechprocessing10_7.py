import numpy as np
from math import log10
from scipy.io import wavfile, loadmat
from scipy.signal import resample, lfilter, medfilt
import soundfile as sf
import matplotlib.pyplot as plt
import sys

# The new sampling rate
fsout = 10000

# Load filter coefficients that were designed with FDATool into a list
myfilter = loadmat('firfilter.mat')
firfilter = myfilter['Num'].tolist()
firfilter = [item for sublist in firfilter for item in sublist]


# Calculate window and stepLength and append them to a list (will be unpacked)
# Window length is 40ms and step length is 10ms
def calculate_window_steplength(fs):
    frame_list = []
    window_length = int((40 * fs) / 1000)
    step_length = int((10 * fs) / 1000)
    frame_list.append(window_length)
    frame_list.append(step_length)
    return frame_list


# Function to calculate pitch and trust values for given signal
def pitch_trust(signal, fs, gender, window_length, step_length):
    pd_high = 0
    pd_low = 0
    if gender == 1:  # If speaker is male
        pd_high = int(fs / 75)
        pd_low = int(fs / 200)
    elif gender == 2:  # If speaker is female
        pd_high = int(fs / 150)
        pd_low = int(fs / 300)

    trust = []  # Initialize trust and pitch lists
    pitch_detected = []
    cur_pos = 0  # Start of signal
    length = len(signal)
    # Loop to the end of the signal
    while cur_pos + window_length - 1 <= length:
        # Calculate first frame
        s1n = signal[cur_pos:cur_pos + window_length - 1]
        # Calculate second frame(modified with pd_high)
        s2n = signal[cur_pos:cur_pos + pd_high + window_length - 1]
        # Calculate their correlation.np.correlate equals to the MATLAB function 'xcorr'
        correl = np.correlate(s1n, s2n, 'full').tolist()
        # Calculate whichever is bigger(to search for the pitch)
        max_length = max(len(s1n), len(s2n))
        # Find max of entire correlation matrix
        global_max = max(correl)
        # Find max in correl between pd low and pd high
        local_max = max(correl[max_length + pd_low:max_length + pd_high])
        # Pitch detected is the index of the max value between pd_low and pd_high
        # Loop over both the list and its index
        for index, item in enumerate(correl[max_length + pd_low:max_length + pd_high]):
            if item == local_max:
                pitch_detected.append(index + pd_low)
                # Global max indicates that pitch was detected inside the signal so we need to calculate for each
                # local max how close it is to the global max to obtain a trust measure.
                trust.append(local_max / global_max)

        cur_pos += step_length

    return trust, pitch_detected


# Usage: python/python3 speechProcessing10_7 fileName 1/2
def main():
    # Load signal data and sample_rate from file using soundfile module.
    # File name is 1st argument, gender is 2nd.
    # 1 means male gender,2 means female
    file_in = sys.argv[1] if len(sys.argv) > 1 else 'Data/chapter_10/test_16k.wav'
    gender = int(sys.argv[2]) if len(sys.argv) > 1 else 1
    input_signal, sample_rate = sf.read(file_in)
    input_signal = input_signal.tolist()
    # Calculate length of signal in seconds
    secs = len(input_signal) / sample_rate
    # Calculate number of samples in new sampling rate
    samples = int(secs * fsout)
    # Resample signal and write to disk
    input_signal = resample(input_signal, samples)
    wavfile.write('downsampled.wav', 10000, input_signal)
    # Apply FIR Filter to signal and write to disk.
    y = lfilter(firfilter, [1], input_signal)

    wavfile.write('filtered.wav', 10000, y)
    # Calculate step and window length for given sample rate
    frame_list = calculate_window_steplength(fs=fsout)
    window_length = frame_list[0]
    step_length = frame_list[1]
    # Calculate pitch and trust for both filtered and unfiltered singal
    results = pitch_trust(input_signal, fs=fsout, gender=gender, window_length=window_length,
                          step_length=step_length)
    filtered_results = pitch_trust(y, fs=fsout, gender=gender, window_length=window_length,
                                   step_length=step_length)
    trust = results[0]
    pitch = results[1]
    filtered_trust = filtered_results[0]
    filtered_pitch = filtered_results[1]
    print('Filtered trust is: ', filtered_trust)
    print('Filtered pitch is: ', filtered_pitch)
    # Convert negative log to log for scale purposes.In case the input is negative or zero,calculate the log of a small
    # quantity instead to avoid math domain errors.
    log_trust = [-log10(x + 10 ** -5) if x > 0 else log10(10 ** -5) for x in trust]
    log_filtered_trust = [-log10(x + (10 ** -5)) if x > 0 else log10(10 ** -5) for x in filtered_trust]

    print('Max log filtered_trust: ', max(log_filtered_trust))
    # Set threshold to 0.35 of the max trust.

    threshold = 0.35 * max(log_trust)
    filtered_threshold = 0.35 * max(log_filtered_trust)
    print('Filtered threshold is: ', filtered_threshold)
    # Loop over the trust and each time a trust lower than the threshold is encountered,set the corresponding pitch to
    # zero.
    for index, (t, p) in enumerate(zip(log_filtered_trust, filtered_pitch)):
        if t < filtered_threshold:
            filtered_pitch[index] = 0

    for index, (t, p) in enumerate(zip(log_trust, pitch)):
        if t < threshold:
            pitch[index] = 0
    # Plot all 8 figures
    # print(filtered_pitch)
    plt.figure(1)
    plt.subplot(411)
    plt.plot(np.array(filtered_pitch))
    plt.ylabel('Pitch')
    # plt.legend(loc='upper left')
    plt.legend(bbox_to_anchor=(0., 1.6, 1., .102), loc=2,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.title('Filtered')

    plt.subplot(412)
    plt.plot(np.array(log_filtered_trust))
    plt.ylabel('Log trust')
    plt.subplots_adjust(hspace=1.5)

    plt.subplot(413)
    plt.plot(np.array(pitch))
    plt.title('Unfiltered')
    plt.ylabel('Pitch')

    plt.subplot(414)
    plt.plot(np.array(log_trust))
    plt.xlabel('Frame number')
    plt.ylabel('Log trust')
    plt.subplots_adjust(hspace=1.5)

    # Apply 5 order median filter to the trust and pitch curves.
    filtered_pitch_median = medfilt(filtered_pitch, kernel_size=[5])
    filtered_log_trust_median = medfilt(log_filtered_trust, kernel_size=[5])
    unfiltered_pitch_median = medfilt(pitch, kernel_size=[5])
    unfiltered_log_trust_median = medfilt(log_trust, kernel_size=[5])

    plt.figure(2)
    plt.subplot(411)
    plt.title('With median')
    plt.ylabel('Pitch')
    plt.plot(np.array(filtered_pitch_median))

    plt.subplot(412)
    plt.ylim([0, 1.6])
    plt.ylabel('Log trust')
    plt.plot(np.array(filtered_log_trust_median))

    plt.subplot(413)
    plt.title('Unfiltered(with median)')
    plt.ylabel('Pitch')
    plt.plot(np.array(unfiltered_pitch_median))

    plt.subplot(414)
    plt.ylim([0, 1.6])
    plt.ylabel('Log trust')
    plt.xlabel('Frame unmber')
    plt.plot(np.array(unfiltered_log_trust_median))
    plt.tight_layout()
    plt.show()
    # Save results to file
    np.savetxt('Results/filtered pitch.txt', filtered_pitch, fmt='%i', delimiter=',')
    np.savetxt('Results/filtered logtrust.txt', log_filtered_trust, fmt='%1.4f', delimiter=',')
    np.savetxt('Results/unfiltered pitch.txt', pitch, fmt='%i', delimiter=',')
    np.savetxt('Results/unfiltered logtrust.txt', log_trust, fmt='%1.4f', delimiter=',')


main()
