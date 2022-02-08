import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import (X_TIMESTEPS, load_results, ts2xy)

import wandb
from metareasoning_env import MetareasoningEnv

PROJECT = 'metareasoning-for-state-abstractions'
CONFIG = {
    'policy_type': 'MlpPolicy',
    'total_timesteps': 10000
}
LOGGING_DIRECTORY = 'logs'
INFO_KEYWORDS = ('action',)


def get_mean_reward(y):
    return np.mean(y[-100:])


def get_isolated_mean_rewards(results):
    isolated_mean_rewards = [[], []]

    for _, row in results.iterrows():
        isolated_mean_rewards[int(row['action'])].append(row['r'])

    return [np.mean(isolated_mean_rewards[0]), np.mean(isolated_mean_rewards[1])]


def get_action_probabilities(results):
    actions = (results.loc[:, 'action'].values)[-100:]

    action_frequencies = [0, 0]
    for action in actions:
        action_frequencies[action] += 1

    return [action_frequency / len(actions) for action_frequency in action_frequencies]


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
            isolated_mean_rewards = get_isolated_mean_rewards(results)
            action_probabilities = get_action_probabilities(results)

            wandb.log({
                'Training/Reward': mean_reward,
                'Training/Reward_Naive_Only': isolated_mean_rewards[0],
                'Training/Reward_Proactive_Only': isolated_mean_rewards[1],
                'Training/Naive': action_probabilities[0],
                'Training/Proactive': action_probabilities[1]
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


if __name__ == '__main__':
    main()
