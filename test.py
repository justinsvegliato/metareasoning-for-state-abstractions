import ast

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import (X_TIMESTEPS, load_results, ts2xy)

import wandb
from metareasoning_env import MetareasoningEnv, EXPANSION_STRATEGY_MAP

PROJECT = 'metareasoning-for-state-abstractions'
CONFIG = {
    'policy_type': 'MlpPolicy',
    'total_timesteps': 1000
}

LOGGING_DIRECTORY = 'logs'
INFO_KEYWORDS = ('ground_state', 'abstract_state', 'decisions')

REWARD_EPISODE_WINDOW = 1
ACTION_EPISODE_WINDOW = 10

MODEL_DIRECTORY = 'models'
MODEL_FILE = 'dqn-reference-3'
MODEL_PATH = '{}/{}'.format(MODEL_DIRECTORY, MODEL_FILE)


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
    
    model = DQN(CONFIG['policy_type'], env)
    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=WandbCallback(PROJECT, CONFIG, LOGGING_DIRECTORY)
    )
    model.save(MODEL_PATH)


if __name__ == '__main__':
    main()
