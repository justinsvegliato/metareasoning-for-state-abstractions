import time

import numpy as np

import earth_observation_mdp
from earth_observation_mdp import EarthObservationMDP

STATE_WIDTH = 12
STATE_HEIGHT = 6
SIZE = (STATE_HEIGHT, STATE_WIDTH)
POINTS_OF_INTEREST = 2

ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST)

n_states = len(ground_mdp.states())
n_actions = len(ground_mdp.actions())

def reward_mapper(row):
    return ground_mdp.transition_function(row[0], earth_observation_mdp.ACTIONS[row[1]], row[2])

start = time.time()
index_matrix = np.stack(np.meshgrid(np.arange(n_states), np.arange(n_actions), np.arange(n_states)), -1).reshape(-1, 3)
reward_matrix = np.apply_along_axis(reward_mapper, 1, index_matrix).reshape(n_states, n_actions, n_states)
print(time.time() - start)

