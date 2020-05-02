from .models.brain_dqn import BrainDQN
import numpy as np

'''
Wrapper class for models, use for providing consistent api to environment for purpose of training
'''
class Agent_DQN:
    def __init__(self):
        self.model = None

    def initialize(self, actions):
        self.model = BrainDQN(actions)

    def setInitState(self, observation):
        self.model.setInitState(observation)
        self.model.currentState = np.squeeze(self.model.currentState)

    def getAction(self):
        return np.argmax(self.model.getAction())

    def setPerception(self, next_observation, action, reward, is_done):
        self.model.setPerception(next_observation, action, reward, is_done);

