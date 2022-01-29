
import logging
import statistics
import time

import gym
import numpy as np
from gym import spaces

import cplex_mdp_solver
import earth_observation_mdp
import policy_sketch_refine
import printer
import utils
from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_mdp import ACTION_MAP, EarthObservationMDP

STATE_WIDTH = 12
STATE_HEIGHT = 6
SIZE = (STATE_HEIGHT, STATE_WIDTH)
POINTS_OF_INTEREST = 2
VISIBILITY = None

ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 3
ABSTRACT_STATE_HEIGHT = 3

INITIAL_GROUND_STATE = 0

EXPAND_POINTS_OF_INTEREST = True
GAMMA = 0.99

TRAVERSES = 2
HORIZON = TRAVERSES * STATE_WIDTH

EXPANSION_STRATEGY_MAP = {
  0: 'NAIVE',
  1: 'GREEDY',
  2: 'PROACTIVE'
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
        np.float64(0.0), 
        np.int64(0)
      ]), 
      high=np.array([
        np.float64(1.0), 
        np.int64((STATE_WIDTH / ABSTRACT_STATE_WIDTH) * (STATE_HEIGHT / ABSTRACT_STATE_HEIGHT) * (POINTS_OF_INTEREST ** earth_observation_mdp.VISIBILITY_FIDELITY))
      ]), 
      shape=(2, )
    )
    self.action_space = spaces.Discrete(3)

    self.ground_mdp = None
    self.ground_memory_mdp = None
    self.abstract_mdp = None

    self.current_ground_state = None
    self.current_abstract_state = None
    self.current_action = None

    self.ground_policy_cache = {}
    self.visited_ground_states = []
    self.solved_ground_states = []

    self.previous_quality = 0
    self.current_quality = 0
    self.current_expansions = 0
    self.current_step = 0

  def step(self, action):    
    logging.info("Environment Step [%d, %s]", self.current_step, EXPANSION_STRATEGY_MAP[action])

    logging.info("-- Visited a new abstract state: [%s]", self.current_abstract_state)

    logging.info("-- Executing the policy sketch refine algorithm...")
    start = time.time()
    solution = policy_sketch_refine.solve(self.ground_mdp, self.current_ground_state, self.abstract_mdp, self.current_abstract_state, EXPAND_POINTS_OF_INTEREST, EXPANSION_STRATEGY_MAP[action], GAMMA)
    logging.info("-- Executed the policy sketch refine algorithm: [time=%f]", time.time() - start)

    start = time.time()
    ground_values = utils.get_ground_values(solution['values'], self.ground_mdp, self.abstract_mdp)
    ground_states = self.abstract_mdp.get_ground_states([self.current_abstract_state])
    ground_policy = utils.get_ground_policy(ground_values, self.ground_mdp, self.abstract_mdp, ground_states, self.current_abstract_state, GAMMA)
    logging.info("-- Calculated the ground policy from the solution of the policy sketch refine algorithm: [time=%f]", time.time() - start)

    self.solved_ground_states += ground_states
    for ground_state in ground_states:
      self.ground_policy_cache[ground_state] = ground_policy[ground_state]
    logging.info("-- Updated the ground policy cache for the new abstract state: [%s]", self.current_abstract_state)

    self.previous_quality = self.current_quality
    self.current_quality = self._get_current_quality()
    self.current_expansions += 1

    logging.info("Environment Simulator")
    while self.current_ground_state in self.solved_ground_states and self.current_step <= HORIZON:
      self.visited_ground_states.append(self.current_ground_state)

      self.current_action = self.ground_policy_cache[self.current_ground_state] 

      logging.info(">>>> Ground State: [%s] | Abstract State: [%s] | Action: [%s]", self.current_ground_state, self.current_abstract_state, self.current_action)
      printer.print_earth_observation_policy(self.ground_mdp, visited_ground_states=self.visited_ground_states, expanded_ground_states=ground_states, ground_policy_cache=self.ground_policy_cache)

      self.current_ground_state = utils.get_successor_state(self.current_ground_state, self.current_action, self.ground_mdp)
      self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)

      self.current_step += 1

    return self.__get_observation(), self.__get_reward(), self.__get_done(), self.__get_info(action)

  def reset(self):
    logging.info("Environment Reset")

    start = time.time()
    self.ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST, VISIBILITY)
    logging.info("-- Built the earth observation MDP: [states=%d, actions=%d, time=%f]", len(self.ground_mdp.states()), len(self.ground_mdp.actions()), time.time() - start)

    start = time.time()
    self.ground_memory_mdp = cplex_mdp_solver.MemoryMDP(self.ground_mdp)
    logging.info("-- Built the earth observation memory MDP: [states=%d, actions=%d, time=%f]", self.ground_memory_mdp.n_states, self.ground_memory_mdp.n_actions, time.time() - start)

    start = time.time()
    self.abstract_mdp = EarthObservationAbstractMDP(self.ground_mdp, ABSTRACTION, ABSTRACT_STATE_WIDTH, ABSTRACT_STATE_HEIGHT)
    logging.info("-- Built the abstract earth observation MDP: [states=%d, actions=%d, time=%f]", len(self.abstract_mdp.states()), len(self.abstract_mdp.actions()), time.time() - start)  

    start = time.time()
    abstract_solution = cplex_mdp_solver.solve(self.abstract_mdp, GAMMA, constant_state_values={}, relax_infeasible=False)
    abstract_policy = utils.get_full_ground_policy(abstract_solution['values'], self.abstract_mdp, self.abstract_mdp.states(), GAMMA)
    logging.info("-- Solved the abstract earth observation MDP: [states=%d, actions=%d, time=%f]", len(self.abstract_mdp.states()), len(self.abstract_mdp.actions()), time.time() - start)

    self.current_ground_state = INITIAL_GROUND_STATE
    self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)
    logging.info("-- Set the current ground state and abstract state: [%s, %s]", self.current_ground_state, self.current_abstract_state)

    self.ground_policy_cache = {}
    for ground_state in self.ground_mdp.states():
      self.ground_policy_cache[ground_state] = abstract_policy[self.abstract_mdp.get_abstract_state(ground_state)]

    self.visited_ground_states = []
    self.solved_ground_states = []

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

    policy_indexes = [ACTION_MAP[self.ground_policy_cache[state]] for state in states]

    while step < HORIZON:
      action_values = rewards + GAMMA * np.sum(transition_probabilities * values.reshape(dimension_array), axis=2)
      values = np.choose(policy_indexes, action_values.T)
      step += 1

    values = {state: values[state] for state in states}

    return values[INITIAL_GROUND_STATE]

  def __get_observation(self):
    return np.array([
      np.float64(self.current_quality), 
      np.int64(self.current_expansions)
    ])

  def __get_reward(self):
    return self.current_quality - self.previous_quality

  def __get_done(self):
    return self.current_step > HORIZON

  def __get_info(self, action):
    return {'action': action}


def main():
  env = MetareasoningEnv()
  print(env.reset())
  print(env.step(2))
  print(env.step(2))


if __name__ == '__main__':
  main()
