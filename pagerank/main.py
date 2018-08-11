import numpy as np
from scipy import sparse
import networkx as nx
import matplotlib.pyplot as plt


def draw_hist(x, bins=100):
    plt.hist(x, bins=bins, range=[1e-8, 1e-5])
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def lib_page_rank(file_path, alpha, e):
    with open(file_path, 'r') as f:
        edgelist = [tuple(int(x)-1 for x in line.split())
                    for line in f.readlines()[4:]]

    g = nx.from_edgelist(edgelist, create_using=nx.DiGraph())
    pr = nx.pagerank(g, alpha=alpha, tol=e)
    return max(pr.items(), key=lambda x: x[1])[0] + 1


def data2csr(file_path, nodes, edges):
    row = []
    col = []
    with open(file_path, 'r') as f:
        for line in f.readlines()[4:]:
            origin, destiny = (int(x)-1 for x in line.split())
            row.append(destiny)
            col.append(origin)
    return sparse.csr_matrix(([True] * edges, (row, col)), shape=(nodes, nodes))


def csr_save(filename, csr):
    np.savez(filename,
             nodes=csr.shape[0],
             edges=csr.data.size,
             indices=csr.indices,
             indptr=csr.indptr
             )


def csr_load(filename):
    loader = np.load(filename)
    edges = int(loader['edges'])
    nodes = int(loader['nodes'])
    return sparse.csr_matrix(
        (np.bool_(np.ones(edges)), loader['indices'], loader['indptr']),
        shape=(nodes, nodes)
    )


def page_rank(matrix, alpha=0.85, e=10e-4):
    n = matrix.shape[0]
    iterations = 0
    ranks = np.ones((n, 1)) / n
    tmp = matrix.sum(axis=0).T / alpha

    flag = True
    while flag:
        iterations += 1

        with np.errstate(divide='ignore'):
            new_ranks = matrix.dot((ranks / tmp))
        new_ranks += (1-new_ranks.sum()) / n

        flag = np.linalg.norm(new_ranks - ranks, ord=1) > e
        ranks = new_ranks
    return ranks, iterations


def page_rank2(matrix, vec, e=10e-4):
    n = matrix.shape[0]
    iterations = 0
    ranks = np.ones((n, 1))
    tmp = matrix.sum(axis=0).T

    flag = True
    while flag:
        iterations += 1

        with np.errstate(divide='ignore'):
            new_ranks = matrix.dot((ranks / tmp))
        d = np.linalg.norm(ranks, ord=1) - np.linalg.norm(new_ranks, ord=1)
        new_ranks += d * vec

        flag = np.linalg.norm(new_ranks - ranks, ord=1) > e
        ranks = new_ranks
    return ranks, iterations


def main(load=False):
    file_path = "web-Stanford.txt"
    nodes = 281903
    edges = 2312497
    alpha = 0.85
    e = 1e-6
    if load:
        csr = csr_load('heh1.txt.npz')
    else:
        csr = data2csr(file_path, nodes, edges)
        csr_save('heh1.txt', csr)
    # print("Highest pagerank according to lib has: {0} node".format(lib_page_rank(file_path, alpha, e)))
    pr, iterations = page_rank(csr, np.matrix([[alpha]] * nodes), e)
    print("Iterations: {0}".format(iterations))
    print("Highest pagerank according to us has: {0} node".format(np.argmax(pr)+1))
    draw_hist(np.array(pr.T)[0])
    # pr2, iterations2 = page_rank2(csr, np.matrix([[1-alpha]] * nodes), e)
    # print("Iterations: {0}".format(iterations2))
    # print("Highest pagerank according to us has: {0} node".format(np.argmax(pr2)+1))


if __name__ == "__main__":
    main()
