import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import X_TIMESTEPS, load_results, ts2xy

import wandb
from metareasoning_env import MetareasoningEnv

PROJECT = 'metareasoning-for-state-abstractions'
CONFIG = {
    'policy_type': 'MlpPolicy',
    'total_timesteps': 10000
}
LOGGING_DIRECTORY = 'logs'


class WandbCallback(BaseCallback):
    def __init__(self, project, config, logging_directory):
        super(WandbCallback, self).__init__()

        self.project = project
        self.config = config
        self.logging_directory = logging_directory

    def _on_training_start(self) -> None:
        self.run = wandb.init(
            project=self.project, 
            config=self.config
        )

    def _on_step(self) -> bool:
        x, y = ts2xy(load_results(self.logging_directory), X_TIMESTEPS)

        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            wandb.log({'train/reward': mean_reward})

        return True

    def _on_training_end(self) -> None:
        self.run.finish()


def main():
    env = Monitor(MetareasoningEnv(), LOGGING_DIRECTORY)
    
    model = PPO(CONFIG['policy_type'], env)
    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=WandbCallback(PROJECT, CONFIG, LOGGING_DIRECTORY)
    )


if __name__ == '__main__':
    main()
