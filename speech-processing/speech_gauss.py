import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.stats

# Load training data from mat files using scipy.io module and store them
# Logen,zcn refers to the non-voiced data and loges,zcs to the voiced respectively.
matEn = scipy.io.loadmat('Data/chapter_10/nonspeech.mat')
matEs = scipy.io.loadmat('Data/chapter_10/speech.mat')
logen = matEn['logen']
zcn = matEn['zcn']
loges = matEs['loges']
zcs = matEs['zcs']

# Calculate mean and standard deviation for each class and store them into lists.
mean1 = [np.mean(logen), np.mean(zcn)]
std1 = [np.std(logen), np.mean(zcn)]
mean2 = [np.mean(loges), np.mean(zcs)]
std2 = [np.std(loges), np.mean(zcs)]

logen = [x for sublist in logen for x in sublist]
logen = [(x-mean1[0])/std1[0] for x in logen]
print(logen)
print(np.mean(logen))
plt.hist(logen)
plt.show()

