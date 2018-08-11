import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sci_lin


def gen_matrix(n, floor=-200, ceiling=200):
    matrix = np.random.random_integers(floor, ceiling, size=(n, n))
    return (matrix + matrix.T)/2


def power_iteration(matrix, x, epsilon):
    old_max_val = max(np.amax(x), np.amin(x), key=abs)
    max_val = old_max_val
    err = max_val / old_max_val
    x = x / max_val
    y = 0
    while err > epsilon:
        y += 1
        old_max_val = max_val
        x = np.dot(matrix, x)
        max_val = max(np.amax(x), np.amin(x), key=abs)
        x = x / max_val
        err = abs(1 - max_val / old_max_val)
    return max_val, x / np.linalg.norm(x), y


def inverse_iteration(matrix, x, epsilon):
    lu_and_perm = sci_lin.lu_factor(matrix)
    old_max_val = max(np.amax(x), np.amin(x), key=abs)
    max_val = old_max_val
    err = max_val / old_max_val
    x = x / max_val
    y = 0
    while err > epsilon:
        y += 1
        old_max_val = max_val
        x = sci_lin.lu_solve(lu_and_perm, x)
        max_val = max(np.amax(x), np.amin(x), key=abs)
        x = x / max_val
        err = abs(1 - max_val / old_max_val)
    return 1 / max_val, x / np.linalg.norm(x), y


def rqi(matrix, x, epsilon):
    old_max_val = max(np.amax(x), np.amin(x), key=abs)
    max_val = old_max_val
    err = max_val / old_max_val
    x = x / max_val
    y = 0
    while err > epsilon:
        y += 1
        r = np.dot(np.dot(x.T, matrix), x) / np.dot(x.T, x)
        old_max_val = max_val
        x = np.linalg.solve((matrix-r*np.eye(matrix.shape[0])), x)
        max_val = max(np.amax(x), np.amin(x), key=abs)
        x = x / max_val
        err = abs(1 - max_val / old_max_val)
    return max_val, x / np.linalg.norm(x), y


def rayleigh(matrix, epsilon, r, x):
    x /= np.linalg.norm(x)
    y = np.linalg.solve((matrix-r*np.eye(matrix.shape[0])), x)
    lam = np.dot(y, x)
    r = r + 1 / lam
    err = np.linalg.norm(y-lam*x) / np.linalg.norm(y)
    while err > epsilon:
        x = y / np.linalg.norm(y)
        y = np.linalg.solve((matrix-r*np.eye(matrix.shape[0])), x)
        lam = np.dot(y, x)
        r = r + 1 / lam
        err = np.linalg.norm(y-lam*x) / np.linalg.norm(y)
    return x


# dekompozycja spektralna
def main():
    x_s = []
    y_s = []
    epsilon = 10e-15
    err = 10e-4
    for i in range(2, 100, 2):
        x_s.append(i)
        matrix = gen_matrix(i)
        x = np.ones(i)
        eig_val1, eig_vec1, y = power_iteration(matrix, x, epsilon)
        y_s.append(y)

        eig_val2, eig_vec2 = np.linalg.eig(matrix)
        tmp = max(np.amax(eig_val2), np.amin(eig_val2), key=abs)
        # itemindex = np.where(eig_val2 == tmp)
        # eig_vec2 = eig_vec2[:, itemindex[0][0]]
        if abs(eig_val1 - tmp) > err:
            print("not working")

    plt.plot(x_s, y_s)
    plt.xlabel('matrix dimension')
    plt.ylabel('iterations')
    plt.show()

    for i in range(2, 8, 2):
        matrix = gen_matrix(i)
        x = np.ones(i)
        eig_val1, eig_vec1, y = inverse_iteration(matrix, x, epsilon)
        eig_val2, eig_vec2 = np.linalg.eig(matrix)
        print(eig_val2)
        print(eig_val1)

if __name__ == "__main__":
    main()
