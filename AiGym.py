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
    def __init__(self, name, env, agent, settings=None):
        if settings is None:
            settings = {"": ""}
        self.env = env
        self.name = name
        self.agent = agent
        self.settings = settings

        # Monitored Metrics
        self.reward_average_episode = 0
        self.reward_best_episode = float('-inf')  # Set to min value # Might want to forget after some interval
        self.best_episode = -1
        self.reward_worst_episode = float('inf')  # Set to max value
        self.worst_episode = -1

    def train(self, num_episodes, episode_length=-1, train_interval=1, save_interval=-1, settings=None):
        if settings is None:
            settings = {"render": False, "stat_print_interval": num_episodes // 2}
        print("===============================")
        print("Booting up " + self.name + " for " + self.agent.name)
        # Get initial observation to kick off training
        episode_count = 0
        is_training = True

        while episode_count < num_episodes:
            # Start new episode
            episode_count += 1
            curr_episode_reward = 0
            curr_episode_length = 0
            observation = self.env.reset()
            self.agent.start_episode()
            while episode_length == -1 or episode_length < curr_episode_length:
                curr_episode_length += 1
                if settings["render"]:
                    self.env.render()  # Seem can run as human if pass string human
                # action = self.agent.action_value(observation)
                action, value = self.agent.get_action_value(observation)

                # Execute action
                observation, reward, done, info = self.env.step(action)

                # Process result (store or do online learning)
                self.agent.add_feedback(observation, reward, action, value)

                # Update logs
                curr_episode_reward = curr_episode_reward + reward

                # If episode finished need to reset environment
                if done:
                    break

            # Used to determine "baseline" for reward signal
            _, next_value = self.agent.get_action_value(observation)
            self.agent.end_episode(next_value)

            # Update stats
            self.reward_average_episode = self.reward_average_episode * (episode_count - 1) / episode_count + \
                curr_episode_reward / episode_count

            # Check if need to update metrics
            if curr_episode_reward > self.reward_best_episode:
                self.reward_best_episode = curr_episode_reward

            if curr_episode_reward < self.reward_worst_episode:
                self.reward_worst_episode = curr_episode_reward

            # Print out progress so far if requested
            if settings["stat_print_interval"] and episode_count % settings["stat_print_interval"] == 0 and episode_count != num_episodes:
                print("Currently on episode: ", episode_count, "/", num_episodes)
                self.get_results()

            if 0 < num_episodes <= episode_count:
                break

            # Train (Depending on batch count - TODO)
            if episode_count % train_interval == 0:
                self.agent.learn()

        self.save()
        print("******************************")
        print(f"{bcolors.OKBLUE}Training Complete :){bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Final Stats{bcolors.ENDC}")
        self.get_results()
        print("===============================")

    def save(self):
        self.agent.save()

    def get_results(self):
        # TODO Do something fancier, for now just print to console
        print(f"{bcolors.HEADER}--------------------------------------{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Agent: ", self.name)
        print(f"{bcolors.OKGREEN}Average Reward: ", self.reward_average_episode)
        print(f"{bcolors.OKBLUE}Best Reward: ", self.reward_best_episode)
        print(f"{bcolors.FAIL}Worst Reward: ", self.reward_worst_episode)
        print(f"{bcolors.HEADER}--------------------------------------{bcolors.ENDC}")
