import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from stable_baselines3 import DQN

from metareasoning_env import EXPANSION_STRATEGY_MAP, MetareasoningEnv

MODEL_DIRECTORY = 'models'
MODEL_FILE = 'dqn-reference-3'
MODEL_PATH = '{}/{}'.format(MODEL_DIRECTORY, MODEL_FILE)

# TODO Make this consistent across files
ROWS = 2
COLUMNS = 4


def show_heatmap(env, decisions, action_focus='PROACTIVE'):
    abstract_state_tracker = {}
    for abstract_state in env.abstract_mdp.states():
        abstract_state_tracker[abstract_state] = {}
        for action in EXPANSION_STRATEGY_MAP.values():
            abstract_state_tracker[abstract_state][action] = 0

    for abstract_state, action in decisions:
        abstract_state_tracker[abstract_state][action] += 1

    heatmap_matrix = np.zeros((ROWS, COLUMNS))
    labels = [['' for _ in range(COLUMNS)] for _ in range(ROWS)]

    for row in range(ROWS):
        for column in range(COLUMNS):
            count = 0
            total_count = 0

            location_index = row * COLUMNS + column

            for weather_index in range(4):
                abstract_state = 'abstract_{}_{}'.format(location_index, weather_index)

                count += abstract_state_tracker[abstract_state][action_focus] 
                total_count += sum([abstract_state_tracker[abstract_state][action] for action in EXPANSION_STRATEGY_MAP.values()])

                reward = env.abstract_mdp.reward_function(abstract_state, 'IMAGE')
                if reward > 0:
                    labels[row][column] = 'X'

            heatmap_matrix[row, column] = count / total_count if total_count > 0 else 0

    sns.set(font_scale=1.4)

    sns.heatmap(heatmap_matrix, vmin=0.0, vmax=1.0, annot=labels, square=True, linewidths=1.0, cbar_kws={"orientation": "horizontal"}, fmt='')
    
    plt.show()


def main():
    model = DQN.load(MODEL_PATH)

    step = 0
    decisions = []

    env = MetareasoningEnv()

    observation = env.reset()
    print("Observation:", observation)

    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=False)

        observation, reward, done, info = env.step(int(action))

        print("Observation:", observation)
        print("Reward:", reward)
        print("Done:", done)

        step += 1

        decisions.append((info['abstract_state'], info['decisions'][-1]))

    show_heatmap(env, decisions)

        
if __name__ == '__main__':
    main()
