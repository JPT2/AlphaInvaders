import numpy as np
import gym
import pickle

# Action Space Definition (Pulled from environment)
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
SHOOT = 1
ACTION_LEFT = 3
ACTION_LEFT_SHOOT = 5
ACTION_RIGHT = 2
ACTION_RIGHT_SHOOT = 4

SCREEN_HEIGHT = 210 // 2
SCREEN_WIDTH = 160

env = gym.make('SpaceInvaders-v0')
observation = env.reset()
while True:
    env.render()

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    
    if done:
        env.reset();