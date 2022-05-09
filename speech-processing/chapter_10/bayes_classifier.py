"""10.5 - Bayes Classifier of speech/nonspeech sections in sound file"""
import scipy.io
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import sys
from scipy.signal import medfilt
from math import log10

# Load training data from mat files using scipy.io module and store them
# Logen,zcn refers to the non-voiced data and loges,zcs to the voiced respectively.
matEn = scipy.io.loadmat('../data/chapter_10/nonspeech.mat')
matEs = scipy.io.loadmat('../data/chapter_10/speech.mat')
logen = matEn['logen']
zcn = matEn['zcn']
loges = matEs['loges']
zcs = matEs['zcs']

# Calculate mean and standard deviation for each class and store them into lists.
mean1 = [np.mean(logen), np.mean(zcn)]
std1 = [np.std(logen), np.mean(zcn)]
mean2 = [np.mean(loges), np.mean(zcs)]
std2 = [np.std(loges), np.mean(zcs)]


# Function to calculate the shortTimeEnergy for given windowLength and step.
# This is a direct python implementation from the file found in the gunet e-class.
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
# This is a direct python implementation from the file found in the Gunet e-class.

def zerocrossing_rate(signal, window_length, step):
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


# Function to calculate the distance between given sample-frame and the 2 classes.
# Returns two lists with each containing distances from class 1 and 2 respectively.
def calculate_distance(energy, zcr):
    distance1 = []  # List that stores distance from non-speech class
    distance2 = []  # List that stores distance from speech class

    # Iterate over the data in the wav file   and calculate distances.
    for energy, zcr in zip(energy, zcr):
        d1 = ((energy - mean1[0]) ** 2 / (std1[0] ** 2)) + (zcr - mean1[1]) ** 2 / (std1[1] ** 2)
        d2 = ((energy - mean2[0]) ** 2 / (std2[0] ** 2)) + (zcr - mean2[1]) ** 2 / (std2[1] ** 2)

        distance1.append(d1)
        distance2.append(d2)
    return distance1, distance2


# Function to assign class to test wav file according to minimum distance.
# Returns a list containing '1' and '2' according to whichever class was assigned.

def assign_classes(distance1, distance2):
    target_vector = []
    for d1, d2 in zip(distance1, distance2):
        if d1 < d2:
            target_vector.append(1)
        else:
            target_vector.append(2)
    return target_vector


# Function to calculate the trust for each class assigned by the classifier
# Returns a list containing the trust level for each class.
def evaluate_classifier(distance1, distance2):
    trust_vector = []
    for d1, d2 in zip(distance1, distance2):
        trust = max(d1, d2) / (d1 + d2)
        trust_vector.append(trust)
    return trust_vector


# Function to calculate the classifier's accuracy.
# The threshold is defined at runtime with the 2nd command line argument.

def classifier_accuracy(trust_vector, threshold):
    accuracy_list = []
    for trust in trust_vector:
        if trust < threshold:
            accuracy_list.append(0)
        else:
            accuracy_list.append(1)

    accuracy = sum(accuracy_list) / float(len(accuracy_list))
    return accuracy


# Function to calculate windowLength given the signal and its sample rate.

def calculate_steplength(sample_rate):
    # windowLength = int((len(inputSignal))/((len(inputSignal)/sampleRate)*100))
    step_length = int(10 * sample_rate / 1000)
    return step_length


# Usage: python3 bayes_classifier.py 'file name' threshold.


def main():
    # Load signal data and sample_rate from file using soundfile module.
    # File name is 1st argument, threshold is 2nd.
    # Default threshold is 0.75
    file_in = sys.argv[1] if len(sys.argv) > 1 else '../data/chapter_10/6A.waV'
    threshold = float(sys.argv[2]) if len(sys.argv) > 1 else 0.75
    input_signal, sample_rate = sf.read(file_in)
    input_signal = medfilt(input_signal)
    input_signal = input_signal.tolist()
    print("Signal length is: ", len(input_signal), "Sample rate is: ", sample_rate)

    step_length = calculate_steplength(sample_rate)
    print("Step length is: ", step_length)
    # Calculate energy for given window and step length
    energy = short_time_energy(input_signal, window_length=800, step=step_length)
    # Log normalize it and convert to scale proportionate to the training data
    norm_energy = [10 * log10(x) + 25 for x in energy]
    # Calculate zcr and normalize
    zcr = zerocrossing_rate(input_signal, window_length=800, step=step_length)
    norm_zcr = [100 * x - 5 for x in zcr]
    # Calculate distances and unpack to 2 lists
    distance1, distance2 = calculate_distance(norm_energy, norm_zcr)
    # Call minimum distance classifier to assign classes according to the 2 distance lists
    classification = assign_classes(distance1=distance1, distance2=distance2)
    # Calculate the classifier trust according to the 2 distance lists
    classifier_trust = evaluate_classifier(distance1=distance1, distance2=distance2)

    print('Classification: ', classification)
    # Calculate the classifier accuracy according to the threshold defined at runtime
    accuracy = classifier_accuracy(classifier_trust, threshold=threshold)
    print('Average classifier trust: ', sum(classifier_trust) / float(len(classifier_trust)))
    print('Classifier accuracy:', accuracy)
    # Plot classification and logen,zcr figures.
    plt.subplot(311)
    plt.plot(np.array(classification), label='Class assigned(1 is NS,2 is S)')
    plt.plot(np.array(classifier_trust), label='Classification trust')
    plt.legend(bbox_to_anchor=(0., 1.6, 1., .102), loc=2,
               ncol=2, mode="expand", borderaxespad=0.)

    # To format the name right,find last slash in filename and print its name from the next index to the end
    an_index = file_in.rfind('/')
    plt.title('File name: ' + file_in[an_index + 1:])
    plt.xlabel('Frame number')
    plt.ylabel('Classification trust/Class')

    plt.subplot(312)
    plt.plot(np.array(norm_energy))
    plt.xlabel('Frame number')
    plt.ylabel('Log energy(dB)')

    plt.subplot(313)
    plt.plot(np.array(norm_zcr))
    plt.xlabel('Frame number')
    plt.ylabel('zcr per 10msec')
    plt.subplots_adjust(hspace=1.)
    plt.show()


main()
