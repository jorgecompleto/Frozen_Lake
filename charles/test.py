import gym
import numpy as np
import random

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="rgb_array")
obs = env.reset()
print(obs)
action1 = env.action_space.sample()
action2 = env.action_space.sample()



total_reward = 0
for i in range(100):
    obs = env.reset()
    actions = np.random.choice(4, size=(16))
    for i in actions:
        obs, reward, terminated, truncated, info = env.step(i)
        print(obs, reward, terminated, truncated, info)
        total_reward += reward
    print(total_reward)