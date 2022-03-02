import logging

import matplotlib.pylab as plt
import numpy as np
from pyparsing import col
import seaborn as sns
from matplotlib.patches import Rectangle
from stable_baselines3 import DQN

from metareasoning_env import EXPANSION_STRATEGY_MAP, MetareasoningEnv

RUN_NAME = 'fanciful-plant-61'
RUN_CHECKPOINT = 'final' 

MODEL_DIRECTORY = 'models'
MODEL_TAG = 'dqn'
MODEL_PATH = '{}/{}-{}-[{}]'.format(MODEL_DIRECTORY, MODEL_TAG, RUN_NAME, RUN_CHECKPOINT)

DETERMINISTIC = True

# TODO Make this consistent across files
ROWS = 2
COLUMNS = 4
WEATHER_STATUSES = 4
REWARD_ACTION = 'IMAGE'

HEATMAP_DIRECTORY = 'heatmaps'
HEATMAP_NAME_TEMPLATE = '{}-[{}]-{}-{}'
HEATMAP_PATH_TEMPLATE = '{}/{}'
HEATMAPS = 10

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


def save_heatmap(env, decisions, heatmap_name):
    abstract_state_tracker = {}
    for abstract_state in env.abstract_mdp.states():
        abstract_state_tracker[abstract_state] = {}
        for action in EXPANSION_STRATEGY_MAP.values():
            abstract_state_tracker[abstract_state][action] = 0

    for abstract_state, action in decisions:
        abstract_state_tracker[abstract_state][action] += 1

    heatmap_matrix = np.zeros((len(EXPANSION_STRATEGY_MAP) * ROWS, COLUMNS))
    abstract_reward_locations = set()

    for row in range(ROWS):
        for column in range(COLUMNS):
            location_index = row * COLUMNS + column

            total_count = 0
            for weather_index in range(WEATHER_STATUSES):
                abstract_state = 'abstract_{}_{}'.format(location_index, weather_index)
                total_count += sum([abstract_state_tracker[abstract_state][action] for action in EXPANSION_STRATEGY_MAP.values()])

                reward = env.abstract_mdp.reward_function(abstract_state, REWARD_ACTION)
                if reward > 0:
                    abstract_reward_locations.add((row, column))

            for action_focus_offset, action_focus in enumerate(EXPANSION_STRATEGY_MAP.values()):
                count = 0
                for weather_index in range(WEATHER_STATUSES):
                    abstract_state = 'abstract_{}_{}'.format(location_index, weather_index)
                    count += abstract_state_tracker[abstract_state][action_focus]

                heatmap_matrix[len(EXPANSION_STRATEGY_MAP) * row + action_focus_offset, column] = count / total_count if total_count > 0 else 0

    plt.figure(figsize=(8, 3))
    plt.rcParams["hatch.linewidth"] = 4

    sns.set(font_scale=1.2)
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})

    cmaps = ['Blues', 'Oranges', 'Greens'] * len(EXPANSION_STRATEGY_MAP)
    _, axes = plt.subplots(len(EXPANSION_STRATEGY_MAP) * ROWS, 1, figsize=(12, 4), gridspec_kw={'hspace': 0, 'wspace': 0})

    for row_index, (row, axis, cmap) in enumerate(zip(heatmap_matrix, axes, cmaps)):
        sub_heatmap_matrix = np.reshape(row, (len(row), 1)).T
        sns.heatmap(sub_heatmap_matrix, vmin=0.0, vmax=1.0, ax=axis, cmap=cmap, linewidths=0.5, linecolor='0.5', cbar=False, yticklabels=False, xticklabels=False, annot=True, fmt='.0%')
        
        if row_index != 0 and row_index % len(EXPANSION_STRATEGY_MAP) == 0:
            axis.hlines([0, ], *axis.get_xlim(), linewidths=8.0, colors='k')

        axis.vlines(range(1, COLUMNS), *axis.get_ylim(), linewidths=4.0, colors='k')

        axis.xaxis.set_ticks([])
        axis.yaxis.set_ticks([])

        for abstract_column_index in range(COLUMNS):
            abstract_location = (int(row_index / len(EXPANSION_STRATEGY_MAP)), abstract_column_index)
            if abstract_location in abstract_reward_locations:
                adjusted_location = (abstract_column_index, 0)
                axis.add_patch(Rectangle(adjusted_location, 1, 1, facecolor='gray', edgecolor='white', alpha=0.2, hatch=r"//"))

    plt.savefig(HEATMAP_PATH_TEMPLATE.format(HEATMAP_DIRECTORY, heatmap_name), bbox_inches='tight', pad_inches=0.025)
    plt.clf()


def main():
    print(MODEL_PATH)
    model = DQN.load(MODEL_PATH)

    policy_tag = 'deterministic' if DETERMINISTIC else 'stochastic'

    for index in range(1, HEATMAPS + 1):
        logging.info('Generating [%s] heatmap [%s]', policy_tag, index)

        heatmap_name = HEATMAP_NAME_TEMPLATE.format(RUN_NAME, RUN_CHECKPOINT, policy_tag, index)

        step = 0
        decisions = []

        env = MetareasoningEnv()

        observation = env.reset()
        print("Observation:", observation)

        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=DETERMINISTIC)

            observation, reward, done, info = env.step(int(action))

            print("Observation:", observation)
            print("Reward:", reward)
            print("Done:", done)

            step += 1

            decisions.append((info['abstract_state'], info['decisions'][-1]))

        save_heatmap(env, decisions, heatmap_name)
        logging.info('Generated [%s] heatmap [%s]', policy_tag, index)

        
if __name__ == '__main__':
    main()
