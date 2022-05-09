"""
2.32 - Loads a .wav file and plays it
"""
import winsound
from scipy.io import wavfile

fs, audio = wavfile.read('../data/chapter_10/ah_lrr.wav')
winsound.PlaySound('data/chapter_10/ah_lrr.wav', winsound.SND_FILENAME)
