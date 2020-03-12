import numpy as np
import gym
import tensorflow as tf
import pickle
import AiGym
import Brain

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
demo_agent = Brain.Brain("Simpleton", env.action_space)
training_gym = AiGym.AiGym("Demo", demo_agent, 10, -1, False, -1, {"update_interval": 5})
training_gym.start()
