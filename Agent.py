"""
Agent.py
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
import tensorflow.keras.layers as kl
import numpy as np

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # Note: no tf.get_variable(), just simple Keras API!
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # Logits are unnormalized log probabilities.
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)
        # Separate hidden layers from the same input tensor.
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        # Another way to sample actions:
        #   action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


# Stores examples/training data. Does any data manipulation beyond generation from environment
    # So memory, reward computation, episode bundling, batch sizes
class Agent:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def get_action(self, observation):
        # Pre-process input
        obs = np.concatenate(observation).ravel()

        # Get the action, value pair output from model
        action, value = self.model.action_value(obs[None, :])

        # Query Model
        return action

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def add_feedback(self, observation, reward):
        pass

    def learn(self):
        # TODO Train agent on experiences
        pass