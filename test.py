from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb
from metareasoning_env import MetareasoningEnv

PROJECT = 'metareasoning-for-state-abstractions'
CONFIG = {
    'policy_type': 'MlpPolicy',
    'total_timesteps': 25000
}


def get_env():
    return Monitor(MetareasoningEnv())


def main():
    run = wandb.init(project=PROJECT, config=CONFIG)

    env = DummyVecEnv([get_env])
    
    model = PPO(CONFIG['policy_type'], env)

    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f'models/{run.id}',
        )
    )


if __name__ == '__main__':
    main()
