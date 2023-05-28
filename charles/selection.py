from random import uniform, choice, sample, random
from operator import attrgetter
import numpy as np


def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """

    if population.optim == "max":

        # Sum total fitness
        total_fitness = sum([i.fitness for i in population])
        # Get a 'position' on the wheel
        spin = uniform(0, total_fitness)
        position = 0
        # Find individual in the position of the spin
        for individual in population:
            position += individual.fitness
            if position > spin:
                return individual

    elif population.optim == "min":
        raise NotImplementedError

    else:
        raise Exception("No optimization specified (min or max).")


def tournament_sel(population, size=5):
    """Tournament selection implementation.

    Args:
        population (Population): The population we want to select from.
        size (int): Size of the tournament.

    Returns:
        Individual: The best individual in the tournament.
    """

    # Select individuals based on tournament size
    # with choice, there is a possibility of repetition in the choices,
    # so every individual has a chance of getting selected
    tournament = [choice(population.individuals) for _ in range(size)]

    # with sample, there is no repetition of choices
    # tournament = sample(population.individuals, size)
    if population.optim == "max":
        return max(tournament, key=attrgetter("fitness"))
    if population.optim == "min":
        return min(tournament, key=attrgetter("fitness"))

def rank_selection(population):
    """Ranking Selection

    Args:
        population (Population): The population we want to select from.

    Returns:
        list: List of selected individuals.
    """
    pop_size = len(population)
    ranked_pop = sorted(population, key=lambda x: attrgetter("fitness")(x))
    fitness_sum = sum(i for i in range(1, pop_size + 1))
    probabilities = [i / fitness_sum for i in range(1, pop_size + 1)]
    selected = []

    while len(selected) < pop_size:
        r = np.random.random()
        for i, prob in enumerate(probabilities):
            if r <= prob:
                selected.append(ranked_pop[i])
                break

    return selected