"""10.2-BERNOULLI TRIALS"""
import numpy as np
import matplotlib.pyplot as plt


def error_prob(p):
    return 1- ((1 - p) ** 7 + 7 * p * ((1 - p) ** 6) + 21 * (p ** 2) * ((1 - p) ** 5) + 35 * (p ** 3) * ((1 - p) ** 4))


y = []
p = np.linspace(0, 1, num=100)
for prob in p:
    y.append(error_prob(p))

plt.plot(p, y)
plt.show()
