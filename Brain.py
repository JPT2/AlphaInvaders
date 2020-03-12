"""
Brain.py
---------
    Helper class for model, handles specific training, memory, and parameter requirements
    Any changes to how a model trains should be done via extending this class

    Needs to be able to
        Create a model
        Manage hyper-params
        Update model
            Requiures
                Data from episodes (reward, observations)
"""
import tensorflow as tf


class Brain:
    def __init__(self, name, action_space):
        self.name = name
        self.actions = action_space

    def start_new_experience(self):
        # Reset anything relevant to a new episode of training
        pass

    def end_experience(self):
        pass

    def get_action(self, observation):
        # Run model to get action
        return 1

    def give_feedback(self, reward):
        # Store reward
        pass

    def learn(self):
        pass
