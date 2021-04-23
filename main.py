# -*- coding: utf-8 -*-
import gym
from QLearningAlgorithm import QLearning

env = gym.make("FrozenLake-v0")
Q = QLearning(env, alpha=0.8, gamma=0.95)
Q.train(totalEpisodes=15000, maxSteps=99)
print(Q.Qtable)
print("Average score:", Q.score)
print("")

#Tests
numberOfTests = 5
for i in range(numberOfTests):
    print("Test number", i+1)
    state = env.reset()
    for t in range(100):
        state, reward, done, info = env.step(Q.chooseAction(state))
        if done:
            env.render()
            print("Number of steps: {} \n".format(t))
            break
env.close()