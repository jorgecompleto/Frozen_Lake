#Frozen lake problem

import gym
import numpy as np
from charles.charles import Population, Individual
from charles.crossover import single_point_co, cycle_xo, prob_xo, pmx
from charles.mutation import binary_mutation, swap_mutation, inversion_mutation, scramble_mutation
from charles.selection import tournament_sel, fps, rank_selection
#import seaborn as sns


def get_fitness(self, n_tries=100):
    """A simple objective function how efficient is each policy

    Returns:
        int: the percentage of times that the goal was reached.
    """
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")
    total_rewards = 0.0
    for i in range(n_tries):
        total_rewards += run_episode(env, self.representation)
    return total_rewards / n_tries

#Maximum number of moves - 16 (number of possible places)

#path is the path chosen by the agent (individual)
def run_episode(env, policy):
    total_reward = 0
    # env.render()
    obs = env.reset()
    terminated = False
    truncated = False
    action = np.array(policy)
    steps = 0
    for i in action:
        obs, reward, terminated, truncated, info = env.step(i)
        #print(i, obs, reward, info)
        # env.render() - para usar o render acho que é preciso os wrappers
        total_reward += reward
        steps += 1
        if terminated or truncated:
            #print(f'Episode finished after {steps} timesteps.')
            break
    return total_reward



# Monkey patching
Individual.get_fitness = get_fitness

###################



def compare_fitness(times, gens, xo_prob, mut_prob, select, mutate, crossover, elitism= True):
    average_fitness = 0
    for i in range(times):
        pop= Population(sol_size=16, size=50, replacement=True, valid_set=[0, 1, 2, 3],
                          optim="max")  # pop size igual a 50
        best_fitness = pop.evolve(gens=gens, xo_prob=xo_prob, mut_prob=mut_prob, select=select, mutate=mutate, crossover=crossover, elitism=elitism)
        average_fitness += best_fitness
    return average_fitness / times


combination1 = compare_fitness(times=10, gens=50, xo_prob=0.8, mut_prob=0.2, select=tournament_sel, mutate=scramble_mutation, crossover=prob_xo, elitism=True)
combination2 = compare_fitness(times=10, gens=75, xo_prob=0.8, mut_prob=0.2, select=tournament_sel, mutate=scramble_mutation, crossover=prob_xo, elitism=True)
combination3 = compare_fitness(times=10, gens=100, xo_prob=0.8, mut_prob=0.2, select=tournament_sel, mutate=scramble_mutation, crossover=prob_xo, elitism=True)
print(combination1, combination2, combination3)