import matplotlib.pyplot as plt
import random as rand
import anneal as ann


def heh(neighbours, dist):
    return neighbours * dist**-2


def four_neighbours(board, x, y):
    """Calculate cost for point at given coordinate.

    :param board: bitmap
    :param x: row
    :param y: column
    """
    blacks = 0
    if (x - 1, y) in board:
        blacks += 1

    if (x + 1, y) in board:
        blacks += 1

    if (x, y + 1) in board:
        blacks += 1

    if (x, y - 1) in board:
        blacks += 1

    return heh(blacks, 1)


def eight_neighbours(board, x, y):
    """Calculate cost for point at given coordinate.

    :param board: bitmap
    :param x: row
    :param y: column
    """
    blacks = 0
    if (x - 1, y - 1) in board:
        blacks += 1

    if (x + 1, y + 1) in board:
        blacks += 1

    if (x - 1, y + 1) in board:
        blacks += 1

    if (x + 1, y - 1) in board:
        blacks += 1

    return four_neighbours(board, x, y) + heh(blacks, 1.4)


def twelve_neighbours(board, x, y):
    """Calculate cost for point at given coordinate.

    :param board: bitmap
    :param x: row
    :param y: column
    """
    blacks = 0
    if (x - 2, y) in board:
        blacks += 1

    if (x + 2, y) in board:
        blacks += 1

    if (x, y + 2) in board:
        blacks += 1

    if (x, y - 2) in board:
        blacks += 1

    return eight_neighbours(board, x, y) + heh(blacks, 2)


def swap_points(bitmap, cost):
    """Generate next solution and it's cost by swapping 2 random points of different color.

    :param bitmap: old solution
    :param cost: cost of old solution
    """
    bm = (set(bitmap[0]), bitmap[1], bitmap[2], bitmap[3])

    board_end = bm[1] - 1

    x = rand.randint(0, board_end)
    y = rand.randint(0, board_end)

    a = rand.randint(0, board_end)
    b = rand.randint(0, board_end)

    if (x, y) in bm[0]:
        while (a, b) in bm[0]:
            a = rand.randint(0, board_end)
            b = rand.randint(0, board_end)
    else:
        while (a, b) not in bm[0]:
            a = rand.randint(0, board_end)
            b = rand.randint(0, board_end)
        x, y, a, b = a, b, x, y

    x_s = ((x - bm[2] if x - bm[2] > 0 else 0,
            x + bm[2] + 1 if x + bm[2] < bm[1] else bm[1]),
           (a - bm[2] if a - bm[2] > 0 else 0,
            a + bm[2] + 1 if a + bm[2] < bm[1] else bm[1]))

    y_s = ((y - bm[2] if y - bm[2] > 0 else 0,
            y + bm[2] + 1 if y + bm[2] < bm[1] else bm[1]),
           (b - bm[2] if b - bm[2] > 0 else 0,
            b + bm[2] + 1 if b + bm[2] < bm[1] else bm[1]))

    points = set()
    for x_lim, y_lim in zip(x_s, y_s):
        for u in range(x_lim[0], x_lim[1]):
            for v in range(y_lim[0], y_lim[1]):
                points.add((u, v))

    delta = 0
    for point in points:
        if point in bm[0]:
            delta += bm[3](bm[0], point[0], point[1])

    bm[0].remove((x, y))
    bm[0].add((a, b))

    for point in points:
        if point in bm[0]:
            delta -= bm[3](bm[0], point[0], point[1])

    return bm, cost - delta


def gen_board(n, blacks):
    """Generate random bitmap.

    :param n: width and height of bitmap
    :param blacks: number of black points in bitmap
    :return: generated bitmap as a set of black points coordinates
    """
    x, y = rand.randint(0, n - 1), rand.randint(0, n - 1)
    bitmap = {(x, y)}
    for i in range(blacks - 1):
        while (x, y) in bitmap:
            x, y = rand.randint(0, n - 1), rand.randint(0, n - 1)
        bitmap.add((x, y))
    return bitmap


def draw_bitmap(bitmap, num=1, save=True, display=False):
    """Draw/save bitmap.

    :param display: whether to display solution
    :param save: whether to save solution
    :param num: solution number
    :param bitmap: set of black points coordinates in bitmap
    """
    if save and num % (bitmap[1] // 5) != 0:
        return

    x = [point[0] for point in bitmap[0]]
    y = [point[1] for point in bitmap[0]]

    plt.ylim(-2, bitmap[1] + 2)
    plt.xlim(-2, bitmap[1] + 2)
    plt.plot(x, y, marker='o', markersize=10 // bitmap[1], color='k', ls='')
    if display:
        plt.show()
    if save:
        plt.savefig('./heh/h' + "{:04}".format(num // (bitmap[1] // 5)))
    plt.clf()


def main(n=200, d=0.5, r=2, cost=twelve_neighbours,
         temp=ann.gen_temp(10e4, 10e-2, 0.99996)):

    bitmap = (gen_board(n, int(d * (n**2))), n, r, cost)
    first_cost = 0
    for point in bitmap[0]:
        first_cost += cost(bitmap[0], point[0], point[1])
    print("First cost: %d" % first_cost)

    draw_bitmap(bitmap, save=False, display=True)

    best_sol, best_cost, energy = ann.anneal(bitmap,
                                             swap_points,
                                             first_cost,
                                             ann.acceptance_probability,
                                             temp)

    draw_bitmap(best_sol, save=False, display=True)
    print("Best solution cost: %d" % best_cost)

    plt.plot(range(len(energy)), energy)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()


if __name__ == "__main__":
    main()
