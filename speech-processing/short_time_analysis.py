# Function to calculate the shortTimeEnergy for given windowLength and step.
# This is a direct python translation from the MATLAB file found in the e-class.
def short_time_energy(signal, window_length, step):
    cur_pos = 0
    length = len(signal)
    energy = []
    # frameCounter = 1
    while cur_pos + window_length - 1 <= length:
        window = signal[cur_pos:cur_pos + window_length - 1]
        window_energy = (1 / window_length) * sum([x ** 2 for x in window])
        energy.append(window_energy)
        cur_pos += step
        # frameCounter +=1

    return energy


# Function to calculate the shortTimeZeroCrossingRate for given windowLength and step.
# This is a direct python translation from the MATLAB file found in the e-class.

def zero_crossing_rate(signal, window_length, step):
    cur_pos = 0
    length = len(signal)
    zcr = []
    # frameCounter = 1
    while cur_pos + window_length - 1 <= length:
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
