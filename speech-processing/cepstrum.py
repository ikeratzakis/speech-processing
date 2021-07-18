"""Calculate real and complex spectrum, from Rabiner & Schafer book"""
import numpy as np
import matplotlib.pyplot as plt
from acoustics import cepstrum

# 8.11
n = np.linspace(-50, 49, 100)
Nfft = 1024
alpha = 0.6
x = []
for number in n:
    x.append((alpha ** number) * np.heaviside(number, 1))

complex_cepstrum = cepstrum.complex_cepstrum(x, Nfft)
real_cepstrum = cepstrum.real_cepstrum(x, Nfft)
quefrency = np.arange(0, Nfft)

fig, axes = plt.subplots(3, 3)
axes[0, 0].plot(n, x)
axes[0, 0].set_title('Signal')
axes[1, 0].plot(quefrency[quefrency.size // 2:], real_cepstrum[quefrency.size // 2:])
axes[1, 0].set_title('Real cepstrum')
axes[2, 0].plot(quefrency[quefrency.size // 2:], complex_cepstrum[0][quefrency.size // 2:])
axes[2, 0].set_title('Complex cepstrum')

# 8.12
n = np.linspace(0, 99, 100)
x = [1] * 100
complex_cepstrum = cepstrum.complex_cepstrum(x, Nfft)
real_cepstrum = cepstrum.real_cepstrum(x, Nfft)
quefrency = np.arange(0, Nfft)
axes[0, 1].plot(n, x)
axes[0, 1].set_title('Signal')
axes[1, 1].plot(quefrency[quefrency.size // 2:], real_cepstrum[quefrency.size // 2:])
axes[1, 1].set_title('Real cepstrum')
axes[2, 1].plot(quefrency[quefrency.size // 2:], complex_cepstrum[0][quefrency.size // 2:])
axes[2, 1].set_title('Complex cepstrum')

# 8.13
x = []
for number in n:
    x.append(np.sin(2 * np.pi * number / 100))

complex_cepstrum = cepstrum.complex_cepstrum(x, Nfft)
real_cepstrum = cepstrum.real_cepstrum(x, Nfft)
quefrency = np.arange(0, Nfft)
axes[0, 2].plot(n, x)
axes[0, 2].set_title('Signal')
axes[1, 2].plot(quefrency[quefrency.size // 2:], real_cepstrum[quefrency.size // 2:])
axes[1, 2].set_title('Real cepstrum')
axes[2, 2].plot(quefrency[quefrency.size // 2:], complex_cepstrum[0][quefrency.size // 2:])
axes[2, 2].set_title('Complex cepstrum')
plt.show()