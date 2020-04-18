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
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kls
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
        hidden_logits = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logits), self.value(hidden_vals)

    def action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        # Another way to sample actions:
        #   action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=(0,1))


# Stores examples/training data. Does any data manipulation beyond generation from environment
# So memory, reward computation, episode bundling, batch sizes
class Agent:
    def __init__(self, name, model, settings=None):
        if settings is None:
            self.settings = {
                        "filepath": "./models/",
                        "learning_rate": 7e-3,
                        "value_coef": 0.5,
                        "entropy_coef": 0.5,
                        "gamma": 0.99
                        }
        else:
            self.settings = settings
        self.name = name
        self.model = model

        # Model Management
        self.model.compile(
            optimizer=ko.RMSprop(lr=self.settings["learning_rate"]),
            # Define separate losses for policy logits and value estimate.
            loss=[self._policy_loss, self._value_loss])

        # Memory store
        self.experience_cap = 10  # For now just remember 10 most recent episodes
        self.experiences = []

        # Current episode tracking
        self.ep_observations = []
        self.ep_rewards = []
        self.ep_actions = []
        self.ep_values = []

    def get_action_value(self, observation):
        # Pre-process input
        obs = np.concatenate(observation).ravel()

        # Get the action, value pair output from model
        action, value = self.model.action_value(obs[None, :])

        # Query Model
        return action, value

    def start_episode(self):
        self.ep_actions = []
        self.ep_observations = []
        self.ep_rewards = []
        self.ep_values = []

    def end_episode(self, next_value):
        self.experiences.append({
            "actions": np.array(self.ep_actions),
            "observations": np.array(self.ep_observations),
            "rewards": np.array(self.ep_rewards),
            "values": np.array(self.ep_values),
            "next_value": next_value
        })

        if len(self.experiences) > self.experience_cap:
            self.experiences.pop(0)

    def add_feedback(self, observation, reward, action, value):
        self.ep_actions.append(action)
        obs = np.concatenate(observation).ravel()
        self.ep_observations.append(obs)

        # Call the reward advantages here so don't have to recompute everytime since its what we care about anyways
        self.ep_rewards.append(reward)
        self.ep_values.append(value)

    def learn(self):
        # TODO Train agent on experiences
        actions = self.experiences[-1]["actions"]
        observations = self.experiences[-1]["observations"]
        rewards = self.experiences[-1]["rewards"]
        values = self.experiences[-1]["values"]
        next_value = self.experiences[-1]["next_value"]

        returns, advs = self._reward_advantages(rewards, values, next_value)
        acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)

        losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
        # print("[%d/%d] Losses: %s" % (update + 1, updates, losses))
        print("Losses: %s" % losses)

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.settings["value_coef"] * kls.mean_squared_error(returns, value)

    def _policy_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)

        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.settings["entropy_coef"] * entropy_loss

    def _reward_advantages(self, rewards, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value)

        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.settings["gamma"] * returns[t + 1]
        returns = returns[:-1]
        advantages = returns - values.T
        return returns, advantages.T

    def save(self):
        tf.keras.models.save_model(
            self.model, self.settings["filepath"], overwrite=True, include_optimizer=True, save_format=None,
            signatures=None, options=None
        )

    def load(self):
        self.model = tf.keras.models.load_model(
            self.settings["filepath"], compile=False
        )
        self.model.compile(
            optimizer=ko.RMSprop(lr=self.settings["learning_rate"]),
            # Define separate losses for policy logits and value estimate.
            loss=[self._policy_loss, self._value_loss])
