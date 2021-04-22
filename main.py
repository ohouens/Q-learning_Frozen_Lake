# -*- coding: utf-8 -*-
import gym

env = gym.make("FrozenLake-v0")
max_episodes = 3
max_steps = 30

for i in range(max_episodes):
    observation = env.reset()
    env.render()
    for t in range(max_steps):
        print("")
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        print("new state:", observation)
        print("reward:", reward)
        if(done):
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()