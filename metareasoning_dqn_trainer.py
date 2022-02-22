import ast

import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import (X_TIMESTEPS, load_results, ts2xy)
from stable_baselines3.dqn.policies import DQNPolicy
from torch import nn

import wandb
from metareasoning_env import EXPANSION_STRATEGY_MAP, MetareasoningEnv

PROJECT = 'metareasoning-for-state-abstractions'
LOG_DIRECTORY = 'logs'
INFO_KEYWORDS = ('ground_state', 'abstract_state', 'decisions')

CONFIG = {
    # The total number of time steps [Default = None]
    'total_timesteps': 5000,
    # The learning rate [Default = 0.0001]
    'learning_rate': 0.0001,
    # The size of the experience buffer [Default = 1000000]
    'buffer_size': 1000000,
    # The number of steps before gradient updates start [Default = 50000]
    'learning_starts': 300,
    # The minibatch size of each gradient update [Default = 32]
    'batch_size': 64,
    # The hard/soft update coefficient (1 for hard updating; 0 for soft updating) [Default = 1]
    'tau': 1.0,
    # The discount factor [Default = 0.99]
    'gamma': 0.99,
    # The number of steps before each gradient update [Default = 4]
    'train_freq': 4,
    # The number of gradient steps within each gradient update [Default = 1]
    'gradient_steps': 1,
    # The number of steps before the target network is updated [Default = 10000]
    'target_update_interval': 500,
    # The fraction of steps over which the exploration probability is reduced [Default = 0.1]
    'exploration_fraction': 0.1,
    # The initial exploration probability [Default = 1.0]
    'exploration_initial_eps': 1.0,
    # The final exploration probability [Default = 0.05]
    'exploration_final_eps': 0.05,
    # The maximum value for clipping the gradient in each gradient update [Default = 10]
    'max_grad_norm': 10,
    # The verbosity level [Default = 0]
    'verbose': 0,
    # The seed for pseudorandom number generation [Default = None]
    'seed': None,
    # The execution architecture [Default = cpu]
    'device': 'cpu'
}

REWARD_EPISODE_WINDOW = 1
ACTION_EPISODE_WINDOW = 10

MODEL_DIRECTORY = 'models'
MODEL_TAG = 'dqn'
MODEL_TEMPLATE = '{}/{}-{}'

ENV = None


def get_mean_episode_reward(y):
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


class TrackerCallback(BaseCallback):
    def __init__(self, project, config, log_directory):
        super(TrackerCallback, self).__init__()

        self.run = wandb.init(
            project=project, 
            config=config
        )

        self.log_directory = log_directory

    def _on_step(self) -> bool:
        global ENV

        done = self.locals['dones'][0]

        if done:
            results = load_results(self.log_directory)
            x, y = ts2xy(results, X_TIMESTEPS)

            if len(x) > 0:
                mean_episode_reward = get_mean_episode_reward(y)
                action_probabilities = get_action_probabilities(results)

                log_entry = {
                    'Training/Naive': action_probabilities['NAIVE'],
                    'Training/Proactive': action_probabilities['PROACTIVE'],
                    'Training/Episodes': self.model._episode_num,
                    'Training/Episode Reward': mean_episode_reward,
                    'Training/Episode Length': ENV.episode_lengths[-1]
                }

                with th.no_grad():
                    samples = self.model.replay_buffer.sample(256 , env=self.model._vec_normalize_env)
                    sampled_values, _ = self.model.q_net(samples.observations).max(dim=1)
                    average_value = sampled_values.mean()
                    log_entry['Training/Average Value'] = average_value

                log_entry['Training/Exploration Rate'] = self.model.logger.name_to_value['rollout/exploration_rate']

                wandb.log(log_entry, step=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        self.run.finish()


class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
            # The layers of the neural network [Default = [64, 64]]
            net_arch=[64, 32],
            # The activation function of the neural network [Default = nn.ReLU]
            activation_fn=nn.ReLU,
            # The normalization layer that divides by 255 for images [Default = False]
            normalize_images=False,
            # The optimizer [Default = th.optim.Adam]
            optimizer_class=th.optim.Adam
        )

def main():
    global ENV

    ENV = Monitor(MetareasoningEnv(), LOG_DIRECTORY, info_keywords=INFO_KEYWORDS)

    model = DQN(CustomDQNPolicy, ENV,
        learning_rate=CONFIG['learning_rate'],
        buffer_size=CONFIG['buffer_size'],
        learning_starts=CONFIG['learning_starts'],
        batch_size=CONFIG['batch_size'],
        tau=CONFIG['tau'],
        gamma=CONFIG['gamma'],
        train_freq=CONFIG['train_freq'],
        gradient_steps=CONFIG['gradient_steps'],
        target_update_interval=CONFIG['target_update_interval'],
        exploration_fraction=CONFIG['exploration_fraction'],
        exploration_initial_eps=CONFIG['exploration_initial_eps'],
        exploration_final_eps=CONFIG['exploration_final_eps'],
        max_grad_norm=CONFIG['max_grad_norm'],
        verbose=CONFIG['verbose'],
        seed=CONFIG['seed'],
        device=CONFIG['device']
    )

    logger = configure(LOG_DIRECTORY, ['csv', ])
    model.set_logger(logger)

    tracker_callback = TrackerCallback(PROJECT, CONFIG, LOG_DIRECTORY)

    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=tracker_callback,
        log_interval=1
    )

    model_path = MODEL_TEMPLATE.format(MODEL_DIRECTORY, MODEL_TAG, tracker_callback.run.name)
    model.save(model_path)


if __name__ == '__main__':
    main()
