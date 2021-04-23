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
        self.score = 0
        
    def train(self, totalEpisodes=10, maxSteps=100, maxEpsilon=1, minEpsilon=0.01, decayRate=0.005):
        rewards = []
        epsilon = maxEpsilon
        for episode in range(totalEpisodes):
            state = self.env.reset()
            totalRewards = 0
            for t in range(maxSteps):
                oldState = state
                if random.uniform(0,1) <= epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Qtable[state,:])
                state, reward, done, info = self.env.step(action)
                self.Qtable[oldState, action] += self.alpha*(reward+self.gamma*np.max(self.Qtable[state,:])-self.Qtable[oldState, action])
                totalRewards += reward
                if(done):
                    break
            epsilon = minEpsilon + (maxEpsilon - minEpsilon)*np.exp(-decayRate*episode)
            rewards.append(totalRewards)
        self.score = sum(rewards)/totalEpisodes
    
    def chooseAction(self,state):
        return np.argmax(self.Qtable[state,:])
        
