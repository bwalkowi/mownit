import matplotlib.pyplot as plt
import numpy as np
import zad1


def main():
    t = 1024
    freq = (1, 52, 33, 172, 203)
    a = [0] * t
    for f in freq:
        a += np.sin(np.arange(t) * np.pi * 2 * f)
    b = []
    tmp = (0, t//5, 2*t//5, 3*t//5, 4*t//5, t)
    for i in range(len(tmp) - 1):
        b.extend(np.sin(np.arange(tmp[i], tmp[i+1]) * np.pi * 2 * freq[i]))

    a = zad1.fft(np.array(a))
    plt.plot(np.arange(t), a, color='b')
    plt.show()

    b = zad1.fft(np.array(b))
    plt.plot(np.arange(t), b, color='r')
    plt.show()


if __name__ == "__main__":
    main()
