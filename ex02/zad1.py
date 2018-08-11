import numpy as np


def swap(a, i, j):
    tmp = a[i]
    a[i] = a[j]
    a[j] = tmp


def full_pivoting(a, rp, cp, i):
    r = i
    c = i

    for j in range(i, a.shape[0]):
        for k in range(i, a.shape[1]):
            if np.absolute(a[rp[j], cp[k]]) > np.absolute(a[rp[r], cp[c]]):
                r, c = j, k

    swap(rp, i, r)
    swap(cp, i, c)


def gauss_jordan(a, x, b):
    n = a.shape[0]
    rp = np.arange(n)
    cp = np.arange(n)

    for i in range(n):
        full_pivoting(a, rp, cp, i)

        for j in range(n):
            if j == i:
                continue

            z = a[rp[j], cp[i]] / a[rp[i], cp[i]]
            for k in range(i, n):
                a[rp[j], cp[k]] -= z * a[rp[i], cp[k]]
            b[rp[j]] -= z * b[rp[i]]

    for i in range(n):
        x[rp[i]] = b[rp[i]] / a[rp[i], cp[i]]

    return cp


def main():
    a = np.array([[-3, 2, -6], [5, 7, -5], [1, 4, -2]], dtype=np.float64)
    print(a)
    b = np.array([6, 6, 8], dtype=np.float64)
    x = np.array([0, 0, 0], dtype=np.float64)
    print(gauss_jordan(a, x, b))
    print(a)
    print(b)
    print(x)


if __name__ == "__main__":
    main()
