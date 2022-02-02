
import logging
import random

import gym
import numpy as np
from gym import spaces

import cplex_mdp_solver
import earth_observation_mdp
import policy_sketch_refine
import printer
import utils
from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_mdp import EarthObservationMDP

STATE_WIDTH = 12
STATE_HEIGHT = 6
SIZE = (STATE_HEIGHT, STATE_WIDTH)
POINTS_OF_INTEREST = 2

ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 3
ABSTRACT_STATE_HEIGHT = 3

INITIAL_GROUND_STATE = 0

EXPAND_POINTS_OF_INTEREST = True
GAMMA = 0.99

TRAVERSES = 2
HORIZON = TRAVERSES * STATE_WIDTH

ACTION_MAP = {
    'STAY': 0,
    'NORTH': 1,
    'SOUTH': 2,
    'IMAGE': 3
}

EXPANSION_STRATEGY_MAP = {
  0: 'NAIVE',
  1: 'PROACTIVE'
}

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


# TODO: Build the ground MDP and memory MDP once
# TODO: Vectorize as many operations as possible
class MetareasoningEnv(gym.Env):

  def __init__(self, ):
    super(MetareasoningEnv, self).__init__()

    # TODO: Implement features
    # (1) All the nearest abstract states that have reward
    # (2) Distance from the current abstract state to other abstract states that have reward
    # (3) Measure for the current abstract state's local connectivity
    # (4) Fixed cost for abstract states expanded in your PAMDP or the number of variables (ground/abstract variables)
    self.observation_space = spaces.Box(
      low=np.array([
        np.float32(0.0),
        np.int32(0)
      ]), 
      high=np.array([
        np.float32(1.0),
        np.int32((STATE_WIDTH / ABSTRACT_STATE_WIDTH) * (STATE_HEIGHT / ABSTRACT_STATE_HEIGHT) * (POINTS_OF_INTEREST ** earth_observation_mdp.VISIBILITY_FIDELITY))
      ]), 
      shape=(2, )
    )
    self.action_space = spaces.Discrete(2)

    self.ground_mdp = None
    self.ground_memory_mdp = None
    self.abstract_mdp = None

    self.ground_policy_cache = {}
    self.solved_ground_states = []
    self.visited_ground_states = []

    self.current_ground_state = None
    self.current_abstract_state = None
    self.current_action = None

    self.previous_quality = None
    self.current_quality = None
    self.current_expansions = None
    self.current_step = None

  def step(self, action):    
    logging.info(printer.padder("Environment Step [%d, %s, %s]", 75), self.current_step, EXPANSION_STRATEGY_MAP[action], self.current_abstract_state)

    logging.info("-- Executing the policy sketch refine algorithm...")
    solution = policy_sketch_refine.solve(self.ground_mdp, self.current_ground_state, self.abstract_mdp, self.current_abstract_state, EXPAND_POINTS_OF_INTEREST, EXPANSION_STRATEGY_MAP[action], GAMMA)
    logging.info("-- Executed the policy sketch refine algorithm")

    ground_values = utils.get_ground_values(solution['values'], self.ground_mdp, self.abstract_mdp)
    ground_states = self.abstract_mdp.get_ground_states([self.current_abstract_state])
    ground_policy = utils.get_ground_policy(ground_values, self.ground_mdp, self.abstract_mdp, ground_states, self.current_abstract_state, GAMMA)

    self.solved_ground_states += ground_states
    for ground_state in ground_states:
      self.ground_policy_cache[ground_state] = ground_policy[ground_state]
    logging.info("-- Updated the ground policy cache for the new abstract state: [%s]", self.current_abstract_state)

    self.previous_quality = self.current_quality
    self.current_quality = self._get_current_quality()
    self.current_expansions += 1

    logging.info(printer.padder("Simulator", 91))
    while self.current_ground_state in self.solved_ground_states and self.current_step < HORIZON:
      self.visited_ground_states.append(self.current_ground_state)

      self.current_action = self.ground_policy_cache[self.current_ground_state]

      logging.info(">>>> Ground State: [%s] | Abstract State: [%s] | Action: [%s]", self.current_ground_state, self.current_abstract_state, self.current_action)
      printer.print_earth_observation_policy(self.ground_mdp, visited_ground_states=self.visited_ground_states, expanded_ground_states=ground_states, ground_policy_cache=self.ground_policy_cache)

      self.current_ground_state = utils.get_successor_state(self.current_ground_state, self.current_action, self.ground_mdp)
      self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)

      self.current_step += 1

    return self.__get_observation(), self.__get_reward(), self.__get_done(), self.__get_info(action)

  def reset(self):
    logging.info(printer.padder("Environment Reset", 91))

    self.ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST)
    logging.info("-- Built the earth observation MDP: [states=%d, actions=%d]", len(self.ground_mdp.states()), len(self.ground_mdp.actions()))

    self.ground_memory_mdp = cplex_mdp_solver.MemoryMDP(self.ground_mdp)
    logging.info("-- Built the earth observation memory MDP: [states=%d, actions=%d]", self.ground_memory_mdp.n_states, self.ground_memory_mdp.n_actions)

    self.abstract_mdp = EarthObservationAbstractMDP(self.ground_mdp, ABSTRACTION, ABSTRACT_STATE_WIDTH, ABSTRACT_STATE_HEIGHT)
    logging.info("-- Built the abstract earth observation MDP: [states=%d, actions=%d]", len(self.abstract_mdp.states()), len(self.abstract_mdp.actions()))

    abstract_solution = cplex_mdp_solver.solve(self.abstract_mdp, GAMMA)
    abstract_policy = utils.get_full_ground_policy(abstract_solution['values'], self.abstract_mdp, self.abstract_mdp.states(), GAMMA)
    logging.info("-- Solved the abstract earth observation MDP: [states=%d, actions=%d]", len(self.abstract_mdp.states()), len(self.abstract_mdp.actions()))

    self.ground_policy_cache = {}
    for ground_state in self.ground_mdp.states():
      self.ground_policy_cache[ground_state] = abstract_policy[self.abstract_mdp.get_abstract_state(ground_state)]
    logging.info("-- Built the ground policy cache from the abstract policy")

    self.solved_ground_states = []
    self.visited_ground_states = []

    self.current_ground_state = INITIAL_GROUND_STATE
    self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)
    self.current_action = self.ground_policy_cache[self.current_ground_state]

    self.previous_quality = None
    self.current_quality = self._get_current_quality()
    self.current_expansions = 0
    self.current_step = 0

    return self.__get_observation()

  def _get_current_quality(self):
    states = self.ground_memory_mdp.states
    actions = self.ground_memory_mdp.actions
    rewards = self.ground_memory_mdp.rewards
    transition_probabilities = self.ground_memory_mdp.transition_probabilities

    values = np.zeros((len(states))).astype('float32').reshape(-1, 1)
    action_values = np.zeros((len(states), len(actions))).astype('float32')

    dimension_array = np.ones((1, transition_probabilities.ndim), int).ravel()
    dimension_array[2] = -1

    step = 0

    action_sequence = [ACTION_MAP[self.ground_policy_cache[state]] for state in states]

    while step < HORIZON:
      action_values = rewards + GAMMA * np.sum(transition_probabilities * values.reshape(dimension_array), axis=2)
      values = np.choose(action_sequence, action_values.T)
      step += 1

    values = {state: values[state] for state in states}

    return values[INITIAL_GROUND_STATE]

  def __get_observation(self):
    return np.array([
      np.float32(self.current_quality), 
      np.int32(self.current_expansions)
    ])

  def __get_reward(self):
    return self.current_quality - self.previous_quality

  def __get_done(self):
    return self.current_step > HORIZON

  def __get_info(self, action):
    return {'action': action}


def main():
  random.seed(5)

  env = MetareasoningEnv()
  env.reset()
  env.step(1)


if __name__ == '__main__':
  main()
