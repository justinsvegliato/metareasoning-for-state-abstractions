
import logging
import random

import gym
import numpy as np
from gym import spaces

import cplex_mdp_solver
import policy_sketch_refine
import printer
import utils
import earth_observation_mdp
from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_mdp import EarthObservationMDP

STATE_WIDTH = 12
STATE_HEIGHT = 6
SIZE = (STATE_HEIGHT, STATE_WIDTH)
POINTS_OF_INTEREST = 2

ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 3
ABSTRACT_STATE_HEIGHT = 3

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


# TODO Build the ground MDP and memory MDP once
# TODO Vectorize as many operations as possible
# TODO Check to see if the actions are working properly
class MetareasoningEnv(gym.Env):

    def __init__(self, ):
        super(MetareasoningEnv, self).__init__()

        # TODO Implement a feature that represents all of the nearest abstract states with reward
        # TODO Implement a feature that measures the current abstract state's local connectivity
        # TODO Implement a feature that indicates whether we have we seen this weather pattern already
        self.observation_space = spaces.Box(
            # (1) Feature 1: The difference between the value of the current policy and the value of the previous policy
            # (2) Feature 2: The percentage of abstract states that have been expanded
            # (3) Feature 3: The distance from the current ground state to the nearest ground state with a point of interest
            low=np.array([
                np.float32(-np.Infinity),
                np.float32(0.0),
                np.float32(0.0)
            ]),
            high=np.array([
                np.float32(np.Infinity),
                np.float32(1.0),
                np.float32(STATE_WIDTH)
            ]),
            shape=(3, )
        )
        self.action_space = spaces.Discrete(2)

        self.ground_mdp = None
        self.ground_memory_mdp = None
        self.abstract_mdp = None

        self.solved_ground_states = []
        self.ground_policy_cache = {}

        self.current_ground_state = None
        self.current_abstract_state = None
        self.current_action = None
        self.current_step = None

        self.previous_quality = None
        self.current_quality = None
        self.current_expansion_ratio = None
        self.current_reward_distance = None

    def step(self, action):
        logging.info("ENVIRONMENT STEP [%d, %s, %s]", self.current_step, EXPANSION_STRATEGY_MAP[action], self.current_abstract_state)

        logging.info("-- Executed the policy sketch refine algorithm")
        solution = policy_sketch_refine.solve(self.ground_mdp, self.current_ground_state, self.abstract_mdp, self.current_abstract_state, EXPANSION_STRATEGY_MAP[action], GAMMA)

        # TODO Verify and clean up this confusing code
        ground_values = utils.get_values(solution['values'], self.ground_mdp, self.abstract_mdp)
        ground_states = self.abstract_mdp.get_ground_states([self.current_abstract_state])
        ground_policy = utils.get_ground_policy(ground_values, self.ground_mdp, self.abstract_mdp, ground_states, self.current_abstract_state, GAMMA)

        self.solved_ground_states += ground_states
        for ground_state in ground_states:
            self.ground_policy_cache[ground_state] = ground_policy[ground_state]
        logging.info("-- Updated the ground policy cache for the new abstract state: [%s]", self.current_abstract_state)

        logging.info("SIMULATION")

        if self.current_ground_state not in self.solved_ground_states:
            logging.info(">>>> Encountered a ground state not in the solved ground states")

        if self.__get_done():
            logging.info(">>>> Encountered a step greater than the horizon")

        while self.current_ground_state in self.solved_ground_states and not self.__get_done():
            self.current_action = self.ground_policy_cache[self.current_ground_state]

            logging.info(">>>> Ground State: [%s] | Abstract State: [%s] | Action: [%s]", self.current_ground_state, self.current_abstract_state, self.current_action)
            printer.print_earth_observation_policy(self.ground_mdp, current_ground_state=self.current_ground_state, expanded_ground_states=self.abstract_mdp.get_ground_states([self.current_abstract_state]), ground_policy_cache=self.ground_policy_cache)

            self.current_ground_state = utils.get_successor_state(self.current_ground_state, self.current_action, self.ground_mdp)
            self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)
            self.current_step += 1

        self.previous_quality = self.current_quality
        self.current_quality = self.__get_current_quality()
        self.current_expansions += 1
        self.current_expansion_ratio = self.__get_current_expansion_ratio()
        self.current_reward_distance = self.__get_current_reward_distance()

        return self.__get_observation(), self.__get_reward(), self.__get_done(), self.__get_info(action)

    def reset(self):
        logging.info("ENVIRONMENT RESET")

        self.ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST)
        logging.info("-- Built the earth observation MDP: [states=%d, actions=%d]", len(self.ground_mdp.states()), len(self.ground_mdp.actions()))

        self.ground_memory_mdp = cplex_mdp_solver.MemoryMDP(self.ground_mdp)
        logging.info("-- Built the earth observation memory MDP: [states=%d, actions=%d]", len(self.ground_memory_mdp.states), len(self.ground_memory_mdp.actions))

        self.abstract_mdp = EarthObservationAbstractMDP(self.ground_mdp, ABSTRACTION, ABSTRACT_STATE_WIDTH, ABSTRACT_STATE_HEIGHT)
        logging.info("-- Built the abstract earth observation MDP: [states=%d, actions=%d]", len(self.abstract_mdp.states()), len(self.abstract_mdp.actions()))

        abstract_solution = cplex_mdp_solver.solve(self.abstract_mdp, GAMMA)
        abstract_policy = utils.get_policy(abstract_solution['values'], self.abstract_mdp, GAMMA)
        logging.info("-- Solved the abstract earth observation MDP: [states=%d, actions=%d]", len(self.abstract_mdp.states()), len(self.abstract_mdp.actions()))

        self.solved_ground_states = []    
        self.ground_policy_cache = {}
        for ground_state in self.ground_mdp.states():
            self.ground_policy_cache[ground_state] = abstract_policy[self.abstract_mdp.get_abstract_state(ground_state)]
        logging.info("-- Built the ground policy cache from the abstract policy")    

        # TODO Verify this code for generating an initial ground state
        initial_location = (0, 0)
        initial_point_of_interest_description =  {key: earth_observation_mdp.MAX_VISIBILITY for key in self.ground_mdp.point_of_interest_description}
        self.initial_ground_state = self.ground_mdp.get_state_from_state_factors(initial_location, initial_point_of_interest_description)

        self.current_ground_state = self.initial_ground_state
        self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)
        self.current_action = self.ground_policy_cache[self.current_ground_state]
        self.current_step = 0

        self.previous_quality = 0
        self.current_quality = self.__get_current_quality()
        self.current_expansions = 0
        self.current_expansion_ratio = self.__get_current_expansion_ratio()
        self.current_reward_distance = self.__get_current_reward_distance()

        logging.info("SIMULATION")
        logging.info(">>>> Ground State: [%s] | Abstract State: [%s] | Action: [%s]", self.current_ground_state, self.current_abstract_state, self.current_action)
        printer.print_earth_observation_policy(self.ground_mdp, current_ground_state=self.current_ground_state, expanded_ground_states=[], ground_policy_cache=self.ground_policy_cache)

        return self.__get_observation()

    # TODO Verify the correctness of policy evaluation
    def __get_current_quality(self):
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

        return values[self.initial_ground_state]

    def __get_current_expansion_ratio(self):
        return self.current_expansions / len(self.abstract_mdp.states())

    def __get_current_reward_distance(self):
        current_location, current_weather_status = self.ground_mdp.get_state_factors_from_state(self.current_ground_state)

        current_reward_distance = float('inf')
        
        for point_of_interest_location in current_weather_status:
            vertical_distance = abs(current_location[0] - point_of_interest_location[0])
            horizontal_displacement = point_of_interest_location[1] - current_location[1]
            horizontal_distance = abs(horizontal_displacement) if horizontal_displacement >= 0 else self.ground_mdp.width() - abs(horizontal_displacement)
            manhattan_distance = vertical_distance + horizontal_distance

            current_reward_distance = min(current_reward_distance, manhattan_distance)

        return current_reward_distance

    def __get_observation(self):
        return np.array([
            np.float32(self.current_quality),
            np.float32(self.current_expansion_ratio),
            np.float32(self.current_reward_distance)
        ])
    
    # TODO Add a fixed cost for the abstract states expanded in the PAMDP or the number of ground/abstract variables
    def __get_reward(self):
        return self.current_quality - self.previous_quality

    def __get_done(self):
        return self.current_step > HORIZON

    def __get_info(self, action):
        return {'action': action}

 
def main():
    random.seed(5)

    env = MetareasoningEnv()

    observation = env.reset()
    print("Observation:", observation)

    done = False
    while not done:
        observation, reward, done, _ = env.step(1)
        print("Observation:", observation)
        print("Reward:", reward)
        print("Done:", done)


if __name__ == '__main__':
    main()
