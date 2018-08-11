import random as rand
import math


def acceptance_probability(old_cost, new_cost, temperature):
    """Standard way of calculating acceptance probability.

    :param old_cost: cost of previous solution
    :param new_cost: cost of current solution
    :param temperature: current temperature
    :return: calculated probability
    """
    return math.exp((old_cost - new_cost) / temperature)


def gen_temp(temp=1.0, end_temp=10e-5, alpha=0.995):
    """Standard way of generating temperatures during simulated annealing.

    :param alpha: cooling rate
    :param end_temp: temperature at end
    :param temp: temperature at start
    """
    while temp > end_temp:
        yield temp
        temp *= alpha


def anneal(old_sol, neighbour, old_cost, acc_prob, temperatures, save_sol=None):
    """Conduct simulated annealing.

    :param temperatures: list/generator of temperatures during process
    :param old_sol: first generated solution
    :param neighbour: function to generate next solution
    :param old_cost: cost of the first solution
    :param acc_prob: function to calculate acceptance probability
    :param save_sol: function to save solution
    :returns: best solution and it's cost, list of every encountered cost
    """
    energy = [old_cost]

    best_cost = old_cost
    best_sol = old_sol

    for i, temperature in enumerate(temperatures):
        if save_sol:
            save_sol(old_sol, i)

        new_sol, new_cost = neighbour(old_sol, old_cost)
        if new_cost < best_cost:
            best_cost = new_cost
            best_sol = new_sol

        if rand.random() < acc_prob(old_cost, new_cost, temperature):
            old_sol = new_sol
            old_cost = new_cost
        energy.append(old_cost)
    return best_sol, best_cost, energy
