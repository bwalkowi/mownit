from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import random as rand


def svd(a):
    u, s, v = linalg.svd(a)
    return u, linalg.diagsvd(s, s.shape[0]), v


def create_matrix(k, n, m):
    return k * np.random.rand(n, m)


def gen_matrix():
    k = rand.randint(50, 250)
    found = False
    m = None
    while not found:
        m = create_matrix(k, 3, 3)
        s = linalg.svd(m)[1]
        found = True if s[0] / s[-1] > 100 else False
    return m


def create_sphere():
    s = np.linspace(0, 2 * np.pi, 100)
    t = np.linspace(0, np.pi, 100)

    x = np.outer(np.cos(s), np.sin(t)).flatten()
    y = np.outer(np.sin(s), np.sin(t)).flatten()
    z = np.outer(np.ones_like(t), np.cos(t)).flatten()

    return np.array([x, y, z])


def draw(s, a):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    ax.plot_wireframe(s[0], s[1], s[2], color='r')
    ax.plot_wireframe(a[0], a[1], a[2], color='b')
    plt.show()
    plt.clf()


def main():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    S = create_sphere()

    m = create_matrix(2, 3, 3)
    A = np.dot(m, S)
    # B = np.dot(create_matrix(5, 3, 3), S)
    # C = np.dot(create_matrix(12, 3, 3), S)
    u, s, v = linalg.svd(m)
    x = u[0] * s[0]
    y = v[1] * s[1]

    ax.plot_wireframe([0, x[0]], [0, x[1]], [0, x[2]], color='g', ls='-')
    ax.plot_wireframe(S[0], S[1], S[2], color='r')
    ax.plot_wireframe(A[0], A[1], A[2], color='b')
    plt.show()

    # draw(S, B)
    # draw(S, C)


if __name__ == "__main__":
    main()
