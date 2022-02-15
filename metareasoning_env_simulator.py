import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from stable_baselines3 import DQN

from metareasoning_env import MetareasoningEnv, EXPANSION_STRATEGY_MAP

HEATMAP_INTERVAL = 10

# TODO Make this consistent across files
ROWS = 2
COLUMNS = 4

ENV = MetareasoningEnv()


def show_heatmap(decisions):
    abstract_state_tracker = {abstract_state: {'VISITATION': 0, 'NAIVE': 0, 'PROACTIVE': 0} for abstract_state in ENV.abstract_mdp.states()}

    for abstract_state, action in decisions:
        abstract_state_tracker[abstract_state]['VISITATION'] += 1
        abstract_state_tracker[abstract_state][EXPANSION_STRATEGY_MAP[action]] += 1

    labels = [['' for _ in range(COLUMNS)] for _ in range(ROWS)]
    heatmap_matrix = np.zeros((ROWS, COLUMNS))

    for row in range(ROWS):
        for column in range(COLUMNS):
            proactive_count = 0
            total_count = 0

            location_index = row * COLUMNS + column
            for weather_index in range(4):
                abstract_state = 'abstract_{}_{}'.format(location_index, weather_index)

                proactive_count += abstract_state_tracker[abstract_state]['PROACTIVE'] 
                total_count += abstract_state_tracker[abstract_state]['VISITATION']

                reward = ENV.abstract_mdp.reward_function(abstract_state, 'IMAGE')
                if reward > 0:
                    labels[row][column] = 'P'

            heatmap_matrix[row, column] = proactive_count / total_count if total_count > 0 else 0

    sns.heatmap(heatmap_matrix, vmin=0.0, vmax=1.0, annot=labels, square=True, linewidth=0.5, fmt='')
    plt.show()


def main():
    model = DQN.load("weights/dqn")
    decisions = []
    step = 0

    observation = ENV.reset()

    done = False
    while not done:
        action, _ = model.predict(observation)
        observation, _, done, info = ENV.step(int(action))
        decisions.append((info['abstract_state'], info['action']))
        step += 1

    show_heatmap(decisions)

        
if __name__ == '__main__':
    main()
