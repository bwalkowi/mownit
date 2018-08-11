from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import random as rand


def gen_matrix():
    found = False
    m = None
    while not found:
        m = create_matrix(rand.randint(1, 100), 3, 3)
        s = linalg.svd(m, compute_uv=False)
        first = s[0] if s[0] > 0 else 1
        last = first
        for elem in s:
            last = elem if elem != 0 else last
        found = True if first / last > 100 else False
    return m


def plot_sphere(ax, sphere, col='r'):
    n = int(sphere[0].shape[0]**0.5)
    x = sphere[0].reshape((n, n))
    y = sphere[1].reshape((n, n))
    z = sphere[2].reshape((n, n))
    ax.plot_wireframe(x, y, z, color=col)


def create_matrix(k, n, m):
    return k * np.random.rand(n, m)


def create_sphere():
    s = np.linspace(0, 2 * np.pi, 50)
    t = np.linspace(0, np.pi, 50)

    x = np.outer(np.cos(s), np.sin(t)).flatten()
    y = np.outer(np.sin(s), np.sin(t)).flatten()
    z = np.outer(np.ones_like(t), np.cos(t)).flatten()

    return np.array([x, y, z])


def main():
    sphere = create_sphere()
    matrices = (create_matrix(1, 3, 3),
                create_matrix(2, 3, 3),
                create_matrix(5, 3, 3))

    for matrix in matrices:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')

        ellipsoid = np.dot(matrix, sphere)
        plot_sphere(ax, sphere, 'r')
        plot_sphere(ax, ellipsoid, 'b')
        u, s, v = linalg.svd(matrix)
        ax.plot([0, u[0][0] * s[0]], [0, u[1][0] * s[0]], [0, u[2][0] * s[0]], color='g', lw=2)
        ax.plot([0, u[0][1] * s[1]], [0, u[1][1] * s[1]], [0, u[2][1] * s[1]], color='g', lw=2)
        ax.plot([0, u[0][2] * s[2]], [0, u[1][2] * s[2]], [0, u[2][2] * s[2]], color='g', lw=2)
        plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    plot_sphere(ax, np.dot(gen_matrix(), sphere), 'm')
    plt.show()

    matrix = matrices[rand.randint(0, 2)]
    u, s, v = linalg.svd(matrix)
    svd = (u, linalg.diagsvd(s, s.shape[0], s.shape[0]), v)
    for i in range(3):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')
        obj = np.identity(3)
        for j in range(2 - i, 3):
            obj = np.dot(svd[j], obj)
        plot_sphere(ax, np.dot(obj, sphere), 'g')
        plt.show()


if __name__ == "__main__":
    main()
