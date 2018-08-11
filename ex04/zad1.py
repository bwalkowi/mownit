import matplotlib.pyplot as plt
import math
import random as rand
import numpy as np
import anneal as ann


def calc_dist(a, b):
    """Calculate distance between 2 cities.

    :param a: first city
    :param b: second city
    :return: calculated distance
    """
    return math.sqrt((a[0] - b[0])**2 + (a[0] - b[0])**2)


def gen1(n, x_max, y_max):
    """Generate cities using uniform distribution.

    :param y_max: max y coordinate
    :param x_max: max x coordinate
    :param n: number of cities in general
    :return: cities
    """
    x = [rand.randint(0, x_max) for _ in range(n)]
    y = [rand.randint(0, y_max) for _ in range(n)]
    return x, y


def gen2(n, x_max, y_max):
    """Generate cities using normal distribution with 4 distinct parameters.

    :param x_max: max x coordinate
    :param y_max: max y coordinate
    :param n: number of cities in general
    :return: cities
    """
    x = []
    y = []
    sigma = x_max / 4
    mu = y_max / 2
    for row in np.random.randn(n, 2) * sigma + mu:
        x.append(row[0])
        y.append(row[1])
    return x, y


def gen3(n, x_max, y_max):
    """Generate nine separated groups of cities.

    :param x_max: max x coordinate
    :param y_max: max y coordinate
    :param n: number of cities in general
    :return: cities
    """
    xs = [(0, x_max // 9), (x_max // 3, x_max // 2), (7 * x_max // 9, x_max)]
    ys = [(0, y_max // 9), (y_max // 3, y_max // 2), (7 * y_max // 9, y_max)]

    x = [rand.randint(*xs[(i // 3) % 3]) for i in range(n)]
    y = [rand.randint(*ys[i % 3]) for i in range(n)]
    return x, y


def calc_path(cities):
    """Calculate length of path.

    :param cities: cities in visiting order
    :return: length of path
    """
    total_dist = 0
    k_x = cities[0][-1]
    k_y = cities[1][-1]
    for i_x, i_y in zip(*cities):
        total_dist += calc_dist((k_x, k_y), (i_x, i_y))
        k_x, k_y = i_x, i_y
    return total_dist


def swap_cities_cons(cities, path):
    """Swap cities in consecutive manner.

    :param cities: list of cities
    :param path: length of old path
    :return: new list of cities
    """
    xs, ys = cities[0][:], cities[1][:]

    a = rand.randint(0, len(xs) - 1)
    b = (a + 1) % len(xs)
    c = (b + 1) % len(xs)
    d = (c + 1) % len(xs)

    delta = calc_dist((xs[a], ys[a]), (xs[b], ys[b]))
    delta += calc_dist((xs[c], ys[c]), (xs[d], ys[d]))

    xs[b], ys[b], xs[c], ys[c] = xs[c], ys[c], xs[b], ys[b]

    delta -= calc_dist((xs[a], ys[a]), (xs[b], ys[b]))
    delta -= calc_dist((xs[c], ys[c]), (xs[d], ys[d]))

    return (xs, ys), path - delta


def swap_cities_rand(cities, path):
    """Swap cities in random manner.

    :param cities: list of cities
    :param path: length of old path
    :return: new list of cities
    """
    xs, ys = cities[0][:], cities[1][:]

    b = rand.randint(0, len(xs) - 1)
    a = b - 1 if b > 0 else len(xs) - 1
    c = (b + 1) % len(xs)

    e = rand.randint(0, len(cities) - 1)
    while e == b:
        e = rand.randint(0, len(cities) - 1)
    d = e - 1 if e > 0 else len(xs) - 1
    f = (e + 1) % len(xs)

    delta = 0
    if b == f or b == d:
        delta += calc_dist((xs[a], ys[a]), (xs[b], ys[b]))
        delta += calc_dist((xs[b], ys[b]), (xs[e], ys[e]))
        if b == d:
            delta += calc_dist((xs[e], ys[e]), (xs[f], ys[f]))
        else:
            delta += calc_dist((xs[d], ys[d]), (xs[e], ys[e]))

        xs[b], ys[b], xs[e], ys[e] = xs[e], ys[e], xs[b], ys[b]

        delta -= calc_dist((xs[a], ys[a]), (xs[b], ys[b]))
        delta -= calc_dist((xs[b], ys[b]), (xs[e], ys[e]))
        if b == d:
            delta -= calc_dist((xs[e], ys[e]), (xs[f], ys[f]))
        else:
            delta -= calc_dist((xs[d], ys[d]), (xs[e], ys[e]))

    else:
        delta += calc_dist((xs[a], ys[a]), (xs[b], ys[b]))
        delta += calc_dist((xs[b], ys[b]), (xs[c], ys[c]))
        delta += calc_dist((xs[d], ys[d]), (xs[e], ys[e]))
        delta += calc_dist((xs[e], ys[e]), (xs[f], ys[f]))

        xs[b], ys[b], xs[e], ys[e] = xs[e], ys[e], xs[b], ys[b]

        delta -= calc_dist((xs[a], ys[a]), (xs[b], ys[b]))
        delta -= calc_dist((xs[b], ys[b]), (xs[c], ys[c]))
        delta -= calc_dist((xs[d], ys[d]), (xs[e], ys[e]))
        delta -= calc_dist((xs[e], ys[e]), (xs[f], ys[f]))

    return (xs, ys), path - delta


def draw_path(cities, save=False, name='chart', display=True):
    """Draw/save cities and path between them in 2D.

    :param cities: list of next cities in the path
    :param save: whether save plot to file or not
    :param name: if save this is the name of file
    :param display: whether save plot to file or not
    """
    plt.plot(cities[0], cities[1], marker='o', linestyle='--', color='r')
    plt.plot((cities[0][-1], cities[0][0]),
             (cities[1][-1], cities[1][0]),
             marker='o', linestyle='--', color='r')

    if save:
        plt.savefig('./zad1/' + name)
    if display:
        plt.show()

    plt.clf()


def main(n=100, x_max=300, y_max=300, gen=gen3, swap_cities=swap_cities_cons,
         temp=ann.gen_temp(1000, 10e-1, 0.99999), save=False, name='',
         display=True):

    cities = gen(n, x_max, y_max)
    first_cost = calc_path(cities)
    print("First cost: %d" % first_cost)

    gen_name = 'gen1'
    if gen == gen2:
        gen_name = 'gen2'
    elif gen == gen3:
        gen_name = 'gen3'
    name += str(x_max) + 'x' + str(y_max) + '_n' + str(n) + '_' + gen_name + '_'

    draw_path(cities, save, name + 'b', display)

    best_sol, best_cost, energy = ann.anneal(cities,
                                             swap_cities,
                                             first_cost,
                                             ann.acceptance_probability,
                                             temp)

    draw_path(best_sol, save, name + 'a', display)
    print("Best solution cost: %d" % best_cost)

    plt.plot(range(len(energy)), energy)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    if save:
        plt.savefig('./zad1/' + name + 'e')
    if display:
        plt.show()


if __name__ == "__main__":
    main()
