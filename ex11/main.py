import numpy as np


def multiply(cofficient, v):
    return map((lambda x : x * cofficient), v)


def proj(v1, v2):
    return multiply(coefficient(v1, v2), v1)


def gs(X):
    Y = []
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y :
            proj_vec = proj(inY, X[i])
            temp_vec = map(lambda x, y : x - y, temp_vec, proj_vec)
        Y.append(temp_vec)
    return Y


def coefficient(v1, v2):
    print(v1)
    print(v2)
    return np.dot(v2, v1) / np.dot(v1, v1)


def qr(matrix):
    tmp = np.matrix(matrix)
    for i in range(tmp.shape[1]):
        v = matrix[:, i]
        for j in range(i):
            u = tmp[:, j]
            tmp[:, i] -= coefficient(u.squeeze(), v.squeeze()) * tmp[:, j]
        tmp[:, i] = tmp[:, i] / np.linalg.norm(tmp[:, i])
    return tmp


def main():
    print(qr(np.matrix([[1, 2, 3], [4, 5, 6], [6, 7, 8]], dtype=float)))
    print(np.linalg.qr(np.matrix([[1, 2, 3], [4, 5, 6], [6, 7, 8]]))[0])
    print(np.linalg.qr(np.matrix([[1, 2, 3], [4, 5, 6], [6, 7, 8]]))[1])


if __name__ == '__main__':
    main()
