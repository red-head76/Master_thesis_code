import numpy as np
import matplotlib.pyplot as plt
import pdb

# Number of Fourier modes
Nf = 100

uniform = np.ones(Nf)


def gauss(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-1/2 * ((x-mu)/sigma)**2)


normal = gauss(np.arange(Nf), 10, 3) * 10

x = np.arange(Nf)


def f_mode(x, a):
    return np.sin(2 * x * np.pi * a / Nf)


unistack = np.zeros(Nf)
normalstack = np.zeros(Nf)
for a in range(Nf):
    unistack += f_mode(x, a) * uniform[a]
    normalstack += f_mode(x, a) * normal[a]


plt.plot(x, unistack, label="uniform distributed f_modes")
plt.plot(x, normalstack, label="normal distributed f_modes")
plt.legend()
plt.show()
