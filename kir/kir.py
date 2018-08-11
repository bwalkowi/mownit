import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import unittest
import random as rand


def draw_network(graph, curr_max):

    pos = nx.random_layout(graph)

    nx.draw_networkx_nodes(graph, pos,
                           nodelist=graph.nodes(),
                           node_color='w',
                           node_size=200,
                           alpha=0.8)

    for e in graph.edges():
        nx.draw_networkx_edges(graph,
                               pos,
                               edgelist=[e],
                               width=1 + 10 * abs(graph.edge[e[0]][e[1]]['c']) / curr_max,
                               alpha=0.5,
                               edge_color='r' if graph.edge[e[0]][e[1]]['c'] > 0 else 'b')

    labels = {}
    for node in graph.nodes():
        labels[node] = str(node)
    nx.draw_networkx_labels(graph, pos, labels, font_size=8)

    plt.axis('off')
    plt.show()


def rm_not_cyclic_nodes(graph):
    rm = [node for node, degree in graph.degree().items() if degree < 2]
    while rm:
        graph.remove_nodes_from(rm)
        rm = [node for node, degree in graph.degree().items() if degree < 2]


class MainTest(unittest.TestCase):
    def test_main(self, graph):
        for u in graph.nodes():
            curr_sum = 0
            for v in graph.neighbors(u):
                x, y, n = (u, v, -1) if u < v else (v, u, 1)
                curr_sum += n * graph.edge[x][y]['c']
            self.assertAlmostEqual(curr_sum, 0)


def gen_graph(graph, nodes_num):
    for n in range(nodes_num):
        for i in range(1, rand.randint(1, 10)):
            x = rand.randint(1, 20)
            if n + x < nodes_num:
                graph.add_edge(n, n + x, r=rand.randint(1, 5))
    return graph.edges()[0], rand.randint(50000, 500000)


def read_graph(graph, fpath):
    """read graph from g.csv in the form:
    first line: u,v,sem
    rest lines: u,v,impedance
    """
    with open(fpath, 'r') as f:
        e = f.readline().split(',')
        e = (int(e[0]), int(e[1])), int(e[2])

        for line in f.readlines():
            line = line.split(',')
            graph.add_edge(int(line[0]), int(line[1]), r=int(line[2]))
    return e


def main():
    graph = nx.Graph()

    sem = gen_graph(graph, 500)
    # sem = read_graph(graph, 'g.csv')

    # draw graph before modification
    nx.draw(graph)
    plt.show()

    rm_not_cyclic_nodes(graph)

    if graph:
        num_of_edges = graph.number_of_edges()

        # according to KCL and KVL vector b will contain only zeros
        b = np.zeros(num_of_edges)

        # there will be as many equations as edges so A is matrix edges x edges
        A = np.zeros((num_of_edges, num_of_edges))

        edges = graph.edges()
        i = 0
        # firstly, by applying KVL to every basis cycle we will get as
        # many equations as we can (it's still less then number of edges)
        for i, cycle in enumerate(nx.cycle_basis(graph)):
            u = cycle[-1]
            for v in cycle:
                x, y, n = (u, v, -1) if u < v else (v, u, 1)
                A[i, edges.index((x, y))] = n * graph.edge[x][y]['r']
                # if there is SEM on edge we must place it's value in b
                if (x, y) == sem[0]:
                    b[i] = sem[1]
                u = v

        # next, to fill 'empty' equations we use KCL
        for u, i in zip(graph.nodes(), range(i, len(edges), 1)):
            for v in graph.neighbors(u):
                x, y, n = (u, v, -1) if u < v else (v, u, 1)
                A[i, edges.index((x, y))] = n

        x = np.linalg.solve(A, b)

        # finding max c from all nodes, it's needed to draw network - width of edge
        # is proportional to c / max(c)
        curr_max = 0
        for e, c in zip(edges, x):
            graph.edge[e[0]][e[1]]['c'] = c
            if abs(c) > curr_max:
                curr_max = abs(c)

        # testing if everything went as planned
        MainTest().test_main(graph)

        # display graph
        draw_network(graph, curr_max)


if __name__ == "__main__":
    main()
