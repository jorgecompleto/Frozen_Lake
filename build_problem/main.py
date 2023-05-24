#Frozen lake problem

import gym
import numpy as np
from charles.charles import Population, Individual
from charles.crossover import single_point_co, cycle_xo, pmx
from charles.mutation import binary_mutation, inversion_mutation
from charles.selection import tournament_sel, fps


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
    action = policy
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

#ind1 = Individual(size=16, replacement=True, valid_set=[0,1,2,3])
#print(ind1.representation)
pop1 = Population(sol_size=16, size=50, replacement=True, valid_set=[0, 1, 2, 3], optim="max") #pop size igual a 50
print(pop1[0].representation)

pop1.evolve(gens=100, xo_prob=0.9, mut_prob=0, select=tournament_sel, mutate=binary_mutation, crossover=single_point_co, elitism=True)
#print(pop1[0].representation)
#policies_pop_1 = [gen_random_policy() for _ in range(20)]
#print(policies_pop_1)
#policy_scores_gen1 = [get_fitness(env, p) for p in policies_pop_1]
#print(policy_scores_gen1)
#print(max(policy_scores_gen1))

#policies_pop_2 = [single_point_co(i, i+1) for i in policies_pop_1]
#policy_scores_gen2 = [compare_policy(env, p) for p in policies_pop_2]
#print(policy_scores_gen2)
#print(max(policy_scores_gen2))
#print(policies_pop_1[-1], policies_pop_1[0])
#print(policies_pop_2[0], policies_pop_2[1])

#policies_pop_1.evolve()