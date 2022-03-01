import logging

import matplotlib.pylab as plt
import numpy as np
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
HEATMAPS = 1

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

# https://stackoverflow.com/questions/54397334/annotated-heatmap-with-multiple-color-schemes
def save_heatmap(env, decisions, heatmap_name):
    abstract_state_tracker = {}
    for abstract_state in env.abstract_mdp.states():
        abstract_state_tracker[abstract_state] = {}
        for action in EXPANSION_STRATEGY_MAP.values():
            abstract_state_tracker[abstract_state][action] = 0

    for abstract_state, action in decisions:
        abstract_state_tracker[abstract_state][action] += 1

    heatmap_matrix = np.zeros((len(EXPANSION_STRATEGY_MAP) * ROWS, COLUMNS))
    reward_locations = set()

    for row in range(ROWS):
        for column in range(COLUMNS):
            location_index = row * COLUMNS + column

            total_count = 0
            for weather_index in range(WEATHER_STATUSES):
                abstract_state = 'abstract_{}_{}'.format(location_index, weather_index)
                total_count += sum([abstract_state_tracker[abstract_state][action] for action in EXPANSION_STRATEGY_MAP.values()])

                reward = env.abstract_mdp.reward_function(abstract_state, REWARD_ACTION)
                if reward > 0:
                    reward_locations.add((column, len(EXPANSION_STRATEGY_MAP) * row))

            for action_focus_offset, action_focus in enumerate(EXPANSION_STRATEGY_MAP.values()):
                count = 0
                for weather_index in range(WEATHER_STATUSES):
                    abstract_state = 'abstract_{}_{}'.format(location_index, weather_index)
                    count += abstract_state_tracker[abstract_state][action_focus]

                heatmap_matrix[len(EXPANSION_STRATEGY_MAP) * row + action_focus_offset, column] = count / total_count if total_count > 0 else 0

    sns.set(font_scale=1.1)
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    plt.rcParams["hatch.linewidth"] = 4

    axis = sns.heatmap(heatmap_matrix, vmin=0.0, vmax=1.0, linewidths=1.0, cbar_kws={'orientation': 'horizontal', 'pad': 0.035}, annot=True, fmt='.0%', yticklabels=['N', 'G', 'P'] * ROWS, xticklabels=False, cmap=sns.diverging_palette(220, 20, as_cmap=True))
    axis.tick_params(labelright=True, rotation=0)
    axis.hlines([row * len(EXPANSION_STRATEGY_MAP) for row in range(1, ROWS)], *axis.get_xlim(), linewidths=4.0, colors='k')
    axis.vlines(range(1, COLUMNS), *axis.get_ylim(), linewidths=4.0, colors='k')

    for reward_location in reward_locations:
        axis.add_patch(Rectangle(reward_location, 1, 3, facecolor="gray", edgecolor="white", alpha=0.2, hatch=r"//"))
    
    cbar = axis.collections[0].colorbar
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0%', '100%'])

    plt.tight_layout()
    plt.savefig(HEATMAP_PATH_TEMPLATE.format(HEATMAP_DIRECTORY, heatmap_name), bbox_inches='tight')
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
