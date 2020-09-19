from math import log10
from audiolazy import lazy_lpc
import soundfile as sf


def calculate_lpcloss(signal, window_length, step, auto_corr):
    cur_pos = 0
    signal_length = len(signal)
    vp = []  # the prediction loss

    lpc_sum = 0

    while cur_pos + window_length - 1 <= signal_length:
        window = signal[cur_pos:cur_pos + window_length - 1]

        vp_temp = lazy_lpc.lpc.kautocor(window, order=12)

        for i in range(1, 13):
            lpc_sum += vp_temp.numpoly[i] * auto_corr[i - 1]

        vp_error = 10 * log10((10 ** -6) + auto_corr[0] - lpc_sum) - 10 * log10(auto_corr[0] + 10 ** -6)
        print(vp_error)
        vp.append(vp_error)
        cur_pos += step

    print('Estimated norm vp: ', vp)
    print(len(vp))
    return vp


def calculate_autocorr(signal, window_length, step):
    order = 12
    auto_corr = []
    auto_corr_sum = 0
    signal_length = len(signal)
    cur_pos = 0
    while cur_pos + window_length - 1 <= signal_length:
        window = signal[cur_pos:cur_pos + window_length - 1]
        for i in range(0, window_length - order - 1):
            auto_corr_sum += window[i] * window[i + order]
        auto_corr.append((1 / window_length) * auto_corr_sum)
        auto_corr_sum = 0
        print('Autocorrelation length is: ', len(auto_corr))

        cur_pos += step

    print(auto_corr)
    return auto_corr


def main():
    # Load signal data and sample_rate from file using soundfile module.
    # File name is 1st argument, threshold is 2nd.
    # Default threshold is 0.75
    file_in = 'Data/chapter_10/test_6_fs_10000_sec_1.wav'

    input_signal, sample_rate = sf.read(file_in)
    # input_signal = medfilt(input_signal)
    input_signal = input_signal.tolist()
    print("Signal length is: ", len(input_signal), "Sample rate is: ", sample_rate)

    print("Step length is: ", 100)
    auto_corr = calculate_autocorr(input_signal, window_length=400, step=100)
    vp = calculate_lpcloss(input_signal, window_length=400, step=100, auto_corr=auto_corr)


main()
