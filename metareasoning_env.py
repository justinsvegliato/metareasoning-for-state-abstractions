
import logging
import statistics
import time

import gym
import numpy as np
from gym import spaces

import cplex_mdp_solver
import policy_sketch_refine
import printer
import utils
from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_mdp import EarthObservationMDP

SIZE = (6, 12)
POINTS_OF_INTEREST = 2
VISIBILITY = None

ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 3
ABSTRACT_STATE_HEIGHT = 3

INITIAL_GROUND_STATE = 0

EXPAND_POINTS_OF_INTEREST = True
GAMMA = 0.99

HORIZON = 100
SIMULATIONS = 100

EXPANSION_STRATEGY_MAP = {
  0: 'NAIVE',
  1: 'GREEDY',
  2: 'PROACTIVE'
}

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


class MetareasoningEnv(gym.Env):

  def __init__(self, ):
    super(MetareasoningEnv, self).__init__()
    
    self.observation_space = spaces.Box(low=np.array([np.float32(0.0), ]), high=np.array([np.float32(1.0), ]))
    self.action_space = spaces.Discrete(3)

    self.ground_mdp = None
    self.abstract_mdp = None

    self.current_ground_state = None
    self.current_abstract_state = None
    self.current_action = None

    self.ground_policy_cache = {}
    self.visited_ground_states = []
    self.solved_ground_states = []

    self.steps = 0

  def step(self, action):    
    logging.info("Environment Step [%d]", self.steps)

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

    logging.info("Environment Simulator")
    while self.current_ground_state in self.solved_ground_states and self.steps <= HORIZON:
      self.visited_ground_states.append(self.current_ground_state)

      self.current_action = self.ground_policy_cache[self.current_ground_state] 

      logging.info(">>>> Ground State: [%s] | Abstract State: [%s] | Action: [%s]", self.current_ground_state, self.current_abstract_state, self.current_action)
      printer.print_earth_observation_policy(self.ground_mdp, visited_ground_states=self.visited_ground_states, expanded_ground_states=ground_states, ground_policy_cache=self.ground_policy_cache)

      self.current_ground_state = utils.get_successor_state(self.current_ground_state, self.current_action, self.ground_mdp)
      self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)

      self.steps += 1

    return self.__get_observation(), self.__get_reward(), self.__get_done(), None

  def reset(self):
    logging.info("Environment Reset")

    start = time.time()
    self.ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST, VISIBILITY)
    logging.info("-- Built the earth observation MDP: [states=%d, actions=%d, time=%f]", len(self.ground_mdp.states()), len(self.ground_mdp.actions()), time.time() - start)

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

    return self.__get_observation()

  # TODO Implement policy evaluation because it will still be efficient
  def __get_simulated_cumulative_ground_reward(self):
    simulated_cumulative_ground_rewards = []

    for _ in range(SIMULATIONS):
      simulated_cumulative_ground_reward = 0

      simulated_ground_state = INITIAL_GROUND_STATE
      for _ in range(HORIZON):
          simulated_action = self.ground_policy_cache[simulated_ground_state]
          simulated_cumulative_ground_reward += self.ground_mdp.reward_function(simulated_ground_state, simulated_action)
          simulated_ground_state = utils.get_successor_state(simulated_ground_state, simulated_action, self.ground_mdp)
        
      simulated_cumulative_ground_rewards.append(simulated_cumulative_ground_reward)

    return statistics.mean(simulated_cumulative_ground_rewards)

  def __get_maximum_ground_reward(self):
    states = self.ground_mdp.states()
    actions = self.ground_mdp.actions()

    maximum_immediate_reward = float('-inf')
    for state in states:
        for action in actions:
            immediate_reward = self.ground_mdp.reward_function(state, action)
            maximum_immediate_reward = max(maximum_immediate_reward, immediate_reward)

    return maximum_immediate_reward

  def __get_maximum_cumulative_ground_reward(self):
    maximum_ground_reward = self.__get_maximum_ground_reward()
    maximum_photo_count = (HORIZON / SIZE[0]) * POINTS_OF_INTEREST
    return maximum_ground_reward * maximum_photo_count

  def __get_observation(self):
    quality = self.__get_simulated_cumulative_ground_reward() / self.__get_maximum_cumulative_ground_reward() 
    return np.array([np.float32(quality),])

  def __get_reward(self):
    return self.__get_maximum_cumulative_ground_reward() - self.__get_simulated_cumulative_ground_reward()

  def __get_done(self):
    return self.steps >= HORIZON
  

def main():
  env = MetareasoningEnv()
  
  print(env.reset())

  for _ in range(5):
    print(env.step(2))


if __name__ == '__main__':
  main()
