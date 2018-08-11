import matplotlib.pyplot as plt
import numpy as np
import time


def dft(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def idft(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N) / N
    return np.dot(M, x)


def fft(x):
    N = x.shape[0]
    if N <= 16:
        return dft(x)
    else:
        x_even = fft(x[::2])
        x_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([x_even + factor[:N / 2] * x_odd,
                               x_even + factor[N / 2:] * x_odd])


def main():
    x = np.random.random(1024)
    print(np.allclose(dft(x), np.fft.fft(x)))
    print(np.allclose(x, idft(dft(x))))
    print(np.allclose(fft(x), np.fft.fft(x)))
    a = []
    b = []
    c = []
    d = []
    for y in (32, 128, 1024, 8192):
        print("Times for array of length: %d" % y)
        tmp = np.random.random(y)
        start = time.time()
        d.append(y)

        dft(tmp)
        end = time.time() - start
        a.append(end)
        print("\tdft: %.20f" % end)

        start = time.time()
        fft(tmp)
        end = time.time() - start
        b.append(end)
        print("\tfft: %.20f" % end)

        start = time.time()
        np.fft.fft(tmp)
        end = time.time() - start
        c.append(end)
        print("\tlib: %.20f" % end)

    plt.plot(d, a, color='b')
    plt.plot(d, b, color='r')
    plt.plot(d, c, color='m')
    plt.show()


if __name__ == "__main__":
    main()
