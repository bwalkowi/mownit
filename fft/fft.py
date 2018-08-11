import numpy as np


def dft_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def fft(x):
    """A recursive implementation of the 1D Cooley-Tukey fft"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 == 1:
        raise ValueError("size of x must be a power of 2")
    elif N <= 8:  # this cutoff should be optimized
        return dft_slow(x)
    else:
        x_even = fft(x[::2])
        x_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([x_even + factor[:N / 2] * x_odd,
                               x_even + factor[N / 2:] * x_odd])


print(fft(np.random.random(1024)))
