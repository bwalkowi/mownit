import numpy as np
import matplotlib.pylab as plt

from scipy.integrate import quad, simps


def fun1(x):
    return np.log(x)**2 * np.exp(-x**2)


def trap(v, t):
    val = 0
    for i in range(0, len(v), 2):
        val += (t[i+1] - t[i]) * (v[i] + v[i+1]) / 2
    return val


def sim(a, c, u, v, w):
    return (c - a)*(u + 4 * v + w) / 3


def simpson(v, t, e):
    val = sim(t[0], t[-1], v[0], v[len(v)//2], v[-1])
    left = sim(t[0], t[len(t)//2], v[0], v[len(t)//4], v[len(t)//2])
    right = sim(t[len(t)//2], t[-1], v[len(t)//2], v[3*len(t)//4], v[-1])
    if abs(left + right - val) <= 15 * e:
        return left + right + (left + right - val) / 15
    else:
        return simpson(v[:len(v)//2], t[:len(v)//2], e/2) + simpson(v[len(v)//2:], t[len(v)//2:], e/2)


def main():
    a = np.arange(1, 41)
    b = [fun1(x) for x in a]
    plt.plot(a, b)
    plt.show()
    val1 = simpson(a, b, 10e-6)
    print(val1)
    val2 = simps(b, a)
    print(val2)


if __name__ == "__main__":
    main()
