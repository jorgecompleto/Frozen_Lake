from random import randint, sample, uniform
import numpy as np


def single_point_co(p1, p2):
    """Implementation of single point crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    co_point = randint(1, len(p1)-2)

    offspring1 = p1[:co_point] + p2[co_point:]
    offspring2 = p2[:co_point] + p1[co_point:]

    return offspring1, offspring2


def cycle_xo(p1, p2):
    """Implementation of cycle crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    # offspring placeholders
    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)

    while None in offspring1:
        index = offspring1.index(None)
        val1 = p1[index]
        val2 = p2[index]

        # copy the cycle elements
        while val1 != val2:
            offspring1[index] = p1[index]
            offspring2[index] = p2[index]
            val2 = p2[index]
            index = p1.np.where(val2)

        # copy the rest
        for element in offspring1:
            if element is None:
                index = offspring1.index(None)
                if offspring1[index] is None:
                    offspring1[index] = p2[index]
                    offspring2[index] = p1[index]

    return offspring1, offspring2

def pmx(p1,p2):
    xo_points = sample(range(0, len(p1)),2)
    xo_points.sort()
    def pmx_off(x,y):
        o = [None] * len(p1)
        #offspring 2
        o[xo_points[0]:xo_points[1]] = x[xo_points[0]:xo_points[1]]
        z = set(y[xo_points[0]:xo_points[1]]) - set(x[xo_points[0]:xo_points[1]])
        for number in z:
            temp = number
            index = y.index(x[y.index(temp)])
            while o[index] is not None:
                temp = index
                index = y.index(x[temp])
            o[index] = number
        while None in o:
            index = o.index(None)
            o[index] = y[index]
        return o
    offspring1, offspring2 = pmx_off(p1,p2), pmx_off(p2,p1)
    return offspring1, offspring2


def prob_xo(p1, p2):
    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)
    for i in range(len(offspring1)):
        prob = np.random.uniform()
        if prob < 0.5:
            offspring1[i] = p1[i]
            offspring2[i] = p2[i]
        else:
            offspring1[i] = p2[i]
            offspring2[i] = p1[i]
    return offspring1, offspring2


if __name__ == '__main__':
    p1, p2 = [9, 8, 4, 5, 6, 7, 1, 3, 2, 10], [8, 7, 1, 2, 3, 10, 9, 5, 4, 6]
    o1, o2 = arithmetic_xo(p1, p2)
    print(o1, o2)