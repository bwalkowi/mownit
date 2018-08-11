import random as rand
import anneal as ann
import matplotlib.pyplot as plt


def swap_digits(sud, cost):
    """Find next solution by swapping digits in the same row.

    :param sud: 3D board
    :param cost: cost of old solution
    """
    _sud = [[[col[0], col[1]] for col in row] for row in sud]
    row = rand.randint(0, 8)
    x = rand.randint(0, 8)
    while not _sud[row][x][1]:
        x = rand.randint(0, 8)

    y = rand.randint(0, 8)
    while not _sud[row][y][1] or y == x:
        y = rand.randint(0, 8)

    grp_x = (row // 3) * 3 + x // 3
    grp_y = (row // 3) * 3 + y // 3

    delta = calc_col_cost(_sud, x) + calc_col_cost(_sud, y)
    delta += calc_blk_cost(_sud, grp_x) + calc_blk_cost(_sud, grp_y)

    _sud[row][x][0], _sud[row][y][0] = _sud[row][y][0], _sud[row][x][0]

    delta -= calc_col_cost(_sud, x) + calc_col_cost(_sud, y)
    delta -= calc_blk_cost(_sud, grp_x) + calc_blk_cost(_sud, grp_y)

    return _sud, cost - delta


def calc_row_cost(sud, i):
    """Count number of repeated digits in row.

    :param sud: 3D board
    :param i: row number
    """
    digits = [-1 for _ in range(9)]
    for j in range(9):
        digits[sud[i][j][0] - 1] += 1
    return sum(x for x in digits if x > 0)


def calc_col_cost(sud, i):
    """Count number of repeated digits in column.

    :param sud: 3D board
    :param i: column number
    """
    digits = [-1 for _ in range(9)]
    for j in range(9):
        digits[sud[j][i][0] - 1] += 1
    return sum(x for x in digits if x > 0)


def calc_blk_cost(sud, i):
    """Count number of repeated digits in block.

    :param sud: 3D board
    :param i: block number
    """
    start_row = (i // 3) * 3
    start_col = (i % 3) * 3
    digits = [-1 for _ in range(9)]
    for j in range(start_row, start_row + 3):
        for k in range(start_col, start_col + 3):
            digits[sud[j][k][0] - 1] += 1
    return sum(x for x in digits if x > 0)


def calc_cost(sud):
    """Calculate cost of current solution.

    Cost is number of repeated digits in row, column or block
    :param sud: 3D board
    """
    cost = 0
    for i in range(9):
        # because of a way sudoku is filled at the beginning adding
        # to cost also calc_row_cost is unnecessary
        # cost += calc_row_cost(sud, i)
        cost += calc_col_cost(sud, i)
        cost += calc_blk_cost(sud, i)
    return cost


def read_sud(file_path, sud):
    """Read sudoku from file_path into sud.

    :param file_path: path to file with sudoku
    :param sud: 3D list to which read sudoku
    """
    with open(file_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.split()
            for j, char in enumerate(line):
                if char != 'x':
                    sud[i][j][0] = int(char)
                    sud[i][j][1] = False


def gen_digit(digits):
    """Generate next free digits from digits.

    :param digits: available digits
    """
    for i in range(9):
        while digits[i] > 0:
            yield i + 1
            digits[i] -= 1


def fill_sud(sud):
    """Fill missing digits in sudoku so that in rows are only unique digits.

    :param sud: 3D sudoku board
    """
    for i in range(9):
        digits = [1 for _ in range(9)]
        for j in range(9):
            if not sud[i][j][1]:
                digits[sud[i][j][0] - 1] = 0

        free_digit = gen_digit(digits)
        for j in range(9):
            if sud[i][j][1]:
                sud[i][j][0] = next(free_digit)


def print_board(sud):
    """Print sudoku board.

    :param sud: 3D board
    """
    for i in range(9):
        if i % 3 == 0:
            print("=" * 25)
        for j in range(9):
            print("| " + str(sud[i][j][0]) if j % 3 == 0 else sud[i][j][0],
                  end=' ')
        print('|')
    print("=" * 25)


def main(path_name='./zad3/input2.txt',
         temp=ann.gen_temp(10, 10e-2, 0.99999),
         save=True, name='', display=True):

    sud = [[['x', True] for _ in range(9)] for _ in range(9)]
    read_sud(path_name, sud)
    if display:
        print_board(sud)
        print("\n")

    fill_sud(sud)
    if display:
        print_board(sud)
        print("\nFirst solution cost: %d\n" % calc_cost(sud))

    best_sol, best_cost, energy = ann.anneal(sud,
                                             swap_digits,
                                             calc_cost,
                                             ann.acceptance_probability,
                                             temp)
    if display:
        print_board(best_sol)
        print("\nBest solution cost: %d\nIs sudoku solved: %s\n" %
              (best_cost, best_cost == 0))

    plt.plot(range(len(energy)), energy)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    if save:
        plt.savefig('./zad3/' + name + '_s_' if best_cost == 0 else '_ns' + 'e')
    if display:
        plt.show()


if __name__ == "__main__":
    main()
