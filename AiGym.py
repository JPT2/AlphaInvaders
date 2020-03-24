"""
AiGym
----------
Manages the training environment for an agent and tracks agent progress for general comparison
"""
import gym
import tensorflow as tf


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Might want to make the environment an input arg
class AiGym:
    def __init__(self, name, agent, num_episodes, episode_length, render, save_interval, print_config={}):
        self.env = gym.make('SpaceInvaders-v0')

        self.name = name  # Identifier for purpose of comparison
        self.agent = agent  # Model helper

        # -1 means run until done, otherwise runs episode for given number of steps
        self.episode_length = episode_length

        # -1 means run forever, otherwise runs for specified episodes
        self.num_episodes = num_episodes

        # Environment Control
        self.render = render

        self.update_interval = print_config["update_interval"]  # In terms of episodes

        # Monitored Metrics
        self.reward_average_episode = 0
        self.reward_best_episode = float('-inf')  # Set to min value # Might want to forget after some interval
        self.best_episode = -1
        self.reward_worst_episode = float('inf')  # Set to max value
        self.worst_episode = -1

        self.save_interval = save_interval  # Tells how often to save the model (-1 means don't save)

        # TODO - Add some type of print-frequency so can get updates

    def start(self):
        print("===============================")
        print("Booting up " + self.name + " for " + self.agent.name)
        # Get initial observation to kick off training
        episode_count = 0
        is_training = True

        while is_training:
            # Start new episode
            episode_count += 1
            curr_episode_reward = 0
            curr_episode_length = 0
            observation = self.env.reset()
            while True:
                curr_episode_length += 1
                if self.render:
                    self.env.render()  # Seem can run as human if pass string human
                # action = self.agent.action_value(observation)
                action = self.agent.get_action(observation)

                # Execute action
                observation, reward, done, info = self.env.step(action)

                # Process result (store or do online learning)
                self.agent.add_feedback(observation, reward)

                # Update logs
                curr_episode_reward = curr_episode_reward + reward

                # If episode finished need to reset environment
                if done:
                    break

                # Check if finishing with training
                if 0 <= self.episode_length <= curr_episode_length:
                    break

            # Update stats
            self.reward_average_episode = self.reward_average_episode * (episode_count - 1) / episode_count + \
                curr_episode_reward / episode_count

            # Check if need to update metrics
            if curr_episode_reward > self.reward_best_episode:
                self.reward_best_episode = curr_episode_reward

            if curr_episode_reward < self.reward_worst_episode:
                self.reward_worst_episode = curr_episode_reward

            # Print out progress so far if requested
            if self.update_interval and episode_count % self.update_interval == 0 and episode_count != self.num_episodes:
                print("Currently on episode: ", episode_count, "/", self.num_episodes)
                self.get_results()

            if 0 < self.num_episodes <= episode_count:
                break

            # Train (Depending on batch count - TODO)
            self.agent.learn()

        print("******************************")
        print(f"{bcolors.OKBLUE}Training Complete :){bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Final Stats{bcolors.ENDC}")
        self.get_results()
        print("===============================")

    def get_results(self):
        # TODO Do something fancier, for now just print to console
        print(f"{bcolors.HEADER}--------------------------------------{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Agent: ", self.name)
        print(f"{bcolors.OKGREEN}Average Reward: ", self.reward_average_episode)
        print(f"{bcolors.OKBLUE}Best Reward: ", self.reward_best_episode)
        print(f"{bcolors.FAIL}Worst Reward: ", self.reward_worst_episode)
        print(f"{bcolors.HEADER}--------------------------------------{bcolors.ENDC}")
