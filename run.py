import numpy as np
import gym
import tensorflow as tf
import pickle
import AiGym
import Agent

# Action Space Definition (Pulled from environment)
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
SHOOT = 1
ACTION_LEFT = 3
ACTION_LEFT_SHOOT = 5
ACTION_RIGHT = 2
ACTION_RIGHT_SHOOT = 4

SCREEN_HEIGHT = 210 // 2
SCREEN_WIDTH = 160

# Setup the environments

env = gym.make('SpaceInvaders-v0')
print("Action space: " +  str(env.action_space.n))
model = Agent.Model(num_actions=env.action_space.n)
agent = Agent.Agent("First Attempt", model) # Should take in a batch size

# I want to create a gym for this agent, I want it to save its model versions
training_gym = AiGym.AiGym("First Test", env, agent, {"render": False, "update_interval": 5});
training_gym.train(10)
