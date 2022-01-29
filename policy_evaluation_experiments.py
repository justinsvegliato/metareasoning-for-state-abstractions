import logging
import random
import statistics
import time

import numpy as np

import cplex_mdp_solver
import printer
import utils
from earth_observation_mdp import EarthObservationMDP, ACTION_MAP

STATE_WIDTH = 12
STATE_HEIGHT = 6
SIZE = (STATE_WIDTH, STATE_HEIGHT)
POINTS_OF_INTEREST = 4
VISIBILITY = None

ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 3
ABSTRACT_STATE_HEIGHT = 3

INITIAL_GROUND_STATE = 0

EXPAND_POINTS_OF_INTEREST = True
GAMMA = 0.99

HORIZON = 12
SIMULATIONS = 500

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


def policy_evaluation(mdp, policy):
  values = {state: 0.0 for state in mdp.states()}
  
  step = 0

  while step < HORIZON:
    new_values = {}

    for state in mdp.states():
      action = policy[state]

      immediate_reward = mdp.reward_function(state, action)

      expected_future_reward = 0
      for successor_state in mdp.states():
        expected_future_reward += mdp.transition_function(state, action, successor_state) * values[successor_state]

      new_values[state] = immediate_reward + GAMMA * expected_future_reward

    values = new_values

    step += 1

  return values
  

def simulated_policy_evaluation(mdp, policy):
    simulated_cumulative_ground_rewards = []

    for _ in range(SIMULATIONS):
      simulated_cumulative_ground_reward = 0

      simulated_ground_state = INITIAL_GROUND_STATE
      for _ in range(HORIZON):
          simulated_action = policy[simulated_ground_state]
          simulated_cumulative_ground_reward += mdp.reward_function(simulated_ground_state, simulated_action)
          simulated_ground_state = utils.get_successor_state(simulated_ground_state, simulated_action, mdp)
        
      simulated_cumulative_ground_rewards.append(simulated_cumulative_ground_reward)

    return statistics.mean(simulated_cumulative_ground_rewards)


def vectorized_policy_evaluation(memory_mdp, policy):
    states = memory_mdp.states
    actions = memory_mdp.actions
    rewards = memory_mdp.rewards
    transition_probabilities = memory_mdp.transition_probabilities

    values = np.zeros((len(states))).astype('float32').reshape(-1, 1)
    action_values = np.zeros((len(states), len(actions))).astype('float32')

    dimension_array = np.ones((1, transition_probabilities.ndim), int).ravel()
    dimension_array[2] = -1

    step = 0

    policy_indexes = [ACTION_MAP[policy[state]] for state in states]

    while step < HORIZON:
      action_values = rewards + GAMMA * np.sum(transition_probabilities * values.reshape(dimension_array), axis=2)
      values = np.choose(policy_indexes, action_values.T)
      step += 1

    return {state: values[state] for state in states}


def main():
  for seed in [5, ]:
    logging.info("Starting trial: [seed=%d]", seed)
    random.seed(seed)
    
    mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST, VISIBILITY)
    logging.info("Built the earth observation MDP: [states=%s, actions=%s]", len(mdp.states()), len(mdp.actions()))

    logging.info("Storing the earth observation memory MDP...")
    memory_mdp = cplex_mdp_solver.MemoryMDP(mdp)

    logging.info("Solving the earth observation MDP...")
    ground_solution = cplex_mdp_solver.solve(mdp, GAMMA, constant_state_values={}, relax_infeasible=False)
    printer.print_earth_observation_policy(mdp, [INITIAL_GROUND_STATE], [], ground_solution['policy'])

    # print("--------------------------------------------------")

    # start = time.time()
    # logging.info("Policy Evaluation")
    # values = policy_evaluation(mdp, ground_solution['policy'])
    # print("Values:", values[INITIAL_GROUND_STATE])
    # print("Stopwatch:", time.time() - start)

    print("--------------------------------------------------")

    start = time.time()
    logging.info("Simulation-Based Policy Evaluation")
    simulated_value = simulated_policy_evaluation(mdp, ground_solution['policy'])
    print("Simulated Values:", simulated_value)
    print("Stopwatch:", time.time() - start)

    print("--------------------------------------------------")

    start = time.time()
    logging.info("Vectorized Policy Evaluation")
    values = vectorized_policy_evaluation(memory_mdp, ground_solution['policy'])
    print("Vectorized Values:", values[INITIAL_GROUND_STATE])
    print("Stopwatch:", time.time() - start)


if __name__ == '__main__':
  main()
