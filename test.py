import ast

import numpy as np
import tensorflow as tf
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import (X_TIMESTEPS, load_results, ts2xy)
from stable_baselines.deepq.policies import FeedForwardPolicy

import wandb
from metareasoning_env import EXPANSION_STRATEGY_MAP, MetareasoningEnv

PROJECT = 'metareasoning-for-state-abstractions'
CONFIG = {
    'policy_type': 'MlpPolicy',
    'total_timesteps': 5000
}

LOGGING_DIRECTORY = 'logs'
INFO_KEYWORDS = ('ground_state', 'abstract_state', 'decisions')

REWARD_EPISODE_WINDOW = 1
ACTION_EPISODE_WINDOW = 10

MODEL_DIRECTORY = 'models'
MODEL_FILE = 'dqn-reference-3'
MODEL_PATH = '{}/{}'.format(MODEL_DIRECTORY, MODEL_FILE)


class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
            layers=[64, 32], # The layers of the neural network
            act_fun=tf.nn.relu, # The activation function of the neural network
            layer_norm=False, # The layer normalization flat
            dueling=True, # The dueling parameter that doubles the neural network for action score comparisons
            feature_extraction="mlp" # The feature extraction type
        )


def get_mean_reward(y):
    return np.mean(y[-REWARD_EPISODE_WINDOW:])


def get_action_probabilities(results):
    action_trajectories = (results.loc[:, 'decisions'].values)[-ACTION_EPISODE_WINDOW:]

    action_frequencies = {expansion_strategy: 0 for expansion_strategy in EXPANSION_STRATEGY_MAP.values()}
    for action_trajectory in action_trajectories:
        interpretted_action_trajectory = ast.literal_eval(action_trajectory)
        for action in interpretted_action_trajectory:
            action_frequencies[action] += 1

    total_count = sum(action_frequencies.values())

    return {expansion_strategy: action_frequencies[expansion_strategy] / total_count for expansion_strategy in action_frequencies.keys()}


class WandbCallback(BaseCallback):
    def __init__(self, project, config, logging_directory):
        super(WandbCallback, self).__init__()

        self.project = project
        self.config = config
        self.logging_directory = logging_directory

        self.run = wandb.init(
            project=self.project, 
            config=self.config
        )

    def _on_step(self) -> bool:
        results = load_results(self.logging_directory)
        x, y = ts2xy(results, X_TIMESTEPS)

        if len(x) > 0:
            mean_reward = get_mean_reward(y)
            action_probabilities = get_action_probabilities(results)

            wandb.log({
                'Training/Reward': mean_reward,
                'Training/Naive': action_probabilities['NAIVE'],
                'Training/Proactive': action_probabilities['PROACTIVE']
            })

        return True

    def _on_training_end(self) -> None:
        self.run.finish()


def main():
    env = Monitor(MetareasoningEnv(), LOGGING_DIRECTORY, info_keywords=INFO_KEYWORDS)
    
    model = DQN(CustomDQNPolicy, env,
        learning_rate=0.0001, # The learning rate
        buffer_size=1000000, # The size of the experience buffer
        learning_starts=50000, # The number of steps before gradient updates start
        batch_size=32, # The minibatch size of each gradient update
        tau=1.0, # The update coefficient between 1 and 0 (1 for hard updating; 0 for soft updating)
        gamma=0.99, # The discount factor
        train_freq=4, # The number of steps before each gradient update
        gradient_steps=1, # The number of gradient steps within each gradient update
        target_update_interval=10000, # The number of steps before the target network is updated
        exploration_fraction=0.1, # The fraction of episodes over which the exploration probability is reduced
        exploration_initial_eps=1.0, # The initial exploration probability
        exploration_final_eps=0.05, # The final exploration probability
        max_grad_norm=10, # The maximum value for clipping the gradient in each gradient update
        verbose=1, # The verbosity level
        seed=None # The seed for pseudorandom number generation
    )

    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=WandbCallback(PROJECT, CONFIG, LOGGING_DIRECTORY)
    )

    model.save(MODEL_PATH)


if __name__ == '__main__':
    main()
