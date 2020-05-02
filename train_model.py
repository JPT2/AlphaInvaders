import matplotlib.pyplot as plt

import cv2
from .models.brain_dqn import BrainDQN
import numpy as np

import gym

'''
Models are allowed to train for a fixed amount of timesteps for a fixed number of episodes
'''
def train(agent, episodes_to_play, timesteps_alloted):
    env = gym.make('SpaceInvaders-v0')
    env.reset()

    actions = env.action_space.n
    agent.initialize(actions)

    # Define preprocess step to be applied for all agents (reduce scope of problem for runtime)
    def preprocess(observation):
        observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        observation = observation[26:110, :]
        ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
        return np.reshape(observation, (84, 84, 1))

    rewards = []

    while episodes_to_play >= 0:
        timesteps = timesteps_alloted

        action0 = 0 # do nothing for first action
        observation0, reward0, is_done, info = env.step(action0)
        observation0 = preprocess(observation0)
        agent.setInitState(observation0)
        total_reward = 0
        while timesteps >= 0:
            action = agent.getAction() # Feel like state/observation should be passed here
            
            next_observation, reward, is_done, info = env.step(action)

            if is_done:
                next_observation = env.reset()
                reward = -10 # Penalize retries
            agent.setPerception(preprocess(next_observation), action, reward, is_done)
            timesteps = timesteps - 1
            total_reward = total_reward + reward
        rewards.append(total_reward)
        episodes_to_play = episodes_to_play - 1
        print("T-" + str(episodes_to_play) + ": " + str(total_reward))
    print("Training Session Completed")
    print("Rewards: " + str(rewards))