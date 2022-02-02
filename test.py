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
            actions = (results.loc[:,'action'].values)[-100:]

            action_frequencies = [0, 0]
            for action in actions:
                action_frequencies[action] += 1

            action_probabilities = [action_frequency / len(actions) for action_frequency in action_frequencies]

            mean_reward = np.mean(y[-100:])

            wandb.log({
                'Training/Reward': mean_reward,
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
