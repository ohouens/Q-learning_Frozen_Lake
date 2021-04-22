#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.9):
        self.env = env
        self.Qtable = np.zeros((env.observation_space.n, env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        
    def train(self, epsilon=0.9, max_episodes=10,  max_steps=100):
        for i in range(max_episodes):
            state = self.env.reset()
            for t in range(max_steps):
                oldState = state
                if random.random() <= epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Qtable[state,:])
                state, reward, done, info = self.env.step(action)
                self.Qtable[oldState, action] += self.alpha*(reward+self.gamma*np.max(self.Qtable[state,:])-self.Qtable[oldState, action])
                if(done):
                    break
            #decay epsilon greedy
            r = max((max_episodes-i)/max_episodes, 0)
            epsilon = (epsilon - 0.1)*r + 0.1
    
    def chooseAction(self,state):
        return np.argmax(self.Qtable[state,:])
        
