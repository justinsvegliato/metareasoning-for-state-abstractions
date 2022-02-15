import math
import logging
import statistics

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

EXPAND_POINTS_OF_INTEREST = True
GAMMA = 0.99

TRAVERSES = 10
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

ALPHA = 100
BETA = 0
SCALE = 0.000001

REWARD_TYPE = 'SINGLE_DECISION_POINT_GROUND_STATE'

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


class MetareasoningEnv(gym.Env):

    def __init__(self, ):
        super(MetareasoningEnv, self).__init__()

        # TODO Implement a feature that represents all of the nearest abstract states with reward
        # TODO Implement a feature that measures the current abstract state's local connectivity
        # TODO Implement a feature that indicates whether we have we seen this weather pattern already
        # TODO Update Feature 3
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

        self.decision_point_ground_state = None
        self.decision_point_ground_states = []
        self.decision_point_abstract_state = None

        self.computations = []

        self.current_ground_state = None
        self.current_abstract_state = None
        self.current_action = None
        self.current_step = None

        self.previous_computation_time = None
        self.current_computation_time = None
        self.previous_quality = None
        self.current_quality = None
        self.current_expansion_ratio = None
        self.current_reward_distance = None

    def step(self, action):
        logging.info("ENVIRONMENT STEP [%d, %s, %s]", self.current_step, EXPANSION_STRATEGY_MAP[action], self.current_abstract_state)

        logging.info("-- Executed the policy sketch refine algorithm")
        solution = policy_sketch_refine.solve(self.ground_mdp, self.current_ground_state, self.abstract_mdp, self.current_abstract_state, EXPANSION_STRATEGY_MAP[action], GAMMA)

        # TODO Verify and clean up this confusing code
        new_solved_ground_values = utils.get_values(solution['values'], self.ground_mdp, self.abstract_mdp)
        new_solved_ground_states = self.abstract_mdp.get_ground_states([self.current_abstract_state])
        new_solved_ground_policy = utils.get_ground_policy(new_solved_ground_values, self.ground_mdp, self.abstract_mdp, new_solved_ground_states, self.current_abstract_state, GAMMA)

        self.solved_ground_states += new_solved_ground_states
        for new_solved_ground_state in new_solved_ground_states:
            self.ground_policy_cache[new_solved_ground_state] = new_solved_ground_policy[new_solved_ground_state]
        logging.info("-- Updated the ground policy cache for the new abstract state: [%s]", self.current_abstract_state)

        logging.info("SIMULATION")

        self.decision_point_ground_state = self.current_ground_state
        self.decision_point_ground_states = new_solved_ground_states
        self.decision_point_abstract_state = self.current_abstract_state

        self.computations.append({
            'state_space_size': solution['state_space_size'],
            'action_space_size': solution['action_space_size']
        })

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

        self.previous_computation_time = self.current_computation_time
        self.current_computation_time += utils.get_computation_time(solution['state_space_size'], solution['action_space_size'], SCALE)
        self.previous_quality = self.current_quality
        self.current_quality = self.__get_current_quality()
        self.current_expansions += 1
        self.current_expansion_ratio = self.__get_current_expansion_ratio()
        self.current_reward_distance = self.__get_current_reward_distance()

        return self.__get_observation(), self.__get_reward(), self.__get_done(), self.__get_info(self.decision_point_ground_state, self.decision_point_abstract_state, action)

    def reset(self):
        logging.info("ENVIRONMENT RESET")

        self.ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST)
        logging.info("-- Built the earth observation MDP: [states=%d, actions=%d]", len(self.ground_mdp.states()), len(self.ground_mdp.actions()))

        self.ground_memory_mdp = cplex_mdp_solver.MemoryMDP(self.ground_mdp, parallelize=True)
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

        initial_location = (0, 0)
        initial_point_of_interest_description = {key: earth_observation_mdp.MAX_VISIBILITY for key in self.ground_mdp.point_of_interest_description}
        self.initial_ground_state = self.ground_mdp.get_state_from_state_factors(initial_location, initial_point_of_interest_description)

        self.decision_point_ground_state = self.initial_ground_state
        self.decision_point_ground_states = [self.initial_ground_state]
        self.decision_point_abstract_state = None

        self.computations = []

        self.current_ground_state = self.initial_ground_state
        self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)
        self.current_action = self.ground_policy_cache[self.current_ground_state]
        self.current_step = 0

        self.previous_computation_time = 0
        self.current_computation_time = 0
        self.previous_quality = 0
        self.current_quality = self.__get_current_quality()
        self.current_expansions = 0
        self.current_expansion_ratio = self.__get_current_expansion_ratio()
        self.current_reward_distance = self.__get_current_reward_distance()

        logging.info("SIMULATION")
        logging.info(">>>> Ground State: [%s] | Abstract State: [%s] | Action: [%s]", self.current_ground_state, self.current_abstract_state, self.current_action)
        printer.print_earth_observation_policy(self.ground_mdp, current_ground_state=self.current_ground_state, expanded_ground_states=[], ground_policy_cache=self.ground_policy_cache)

        return self.__get_observation()

    def __get_values(self):
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

        return {state: values[state] for state in states}

    def __get_current_quality(self):
        if REWARD_TYPE == 'INITIAL_GROUND_STATE':
            return self.__get_current_quality_from_initial_ground_state()

        if REWARD_TYPE == 'SINGLE_DECISION_POINT_GROUND_STATE':
            return self.__get_current_quality_from_decision_point_ground_state()

        if REWARD_TYPE == 'ALL_DECISION_POINT_GROUND_STATES':
            return self.__get_current_quality_from_decision_point_ground_states()

    def __get_current_quality_from_initial_ground_state(self):
        values = self.__get_values()
        return values[self.initial_ground_state]

    def __get_current_quality_from_decision_point_ground_state(self):
        values = self.__get_values()
        return values[self.decision_point_ground_state]

    def __get_current_quality_from_decision_point_ground_states(self):
        values = self.__get_values()        
        return statistics.mean([values[decision_point_ground_state] for decision_point_ground_state in self.decision_point_ground_states])

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
    
    def __get_reward(self):
        return utils.get_time_dependent_utility(self.current_quality, self.current_computation_time, ALPHA, BETA) - utils.get_time_dependent_utility(self.previous_quality, self.previous_computation_time, ALPHA, BETA)

    def __get_done(self):
        return self.current_step > HORIZON

    def __get_info(self, ground_state, abstract_state, action):
        return {
            'ground_state': ground_state,
            'abstract_state': abstract_state,
            'action': action
        }

    def closest_goal(self):
        current_location, current_weather_status = self.ground_mdp.get_state_factors_from_state(self.current_ground_state)
        min_dist_loc = None
        min_dist = math.inf
        for key in current_weather_status.keys():
            dist = 0
            if key[1] > current_location[1]:
                dist = key[1] - current_location[1]
            else:
                dist = (key[1] + STATE_WIDTH) - current_location[1]
            if dist < min_dist:
                min_dist = dist
                min_dist_loc = key

        return min_dist, min_dist_loc

    def is_kSR(self):
        current_location, current_weather_status = self.ground_mdp.get_state_factors_from_state(self.current_ground_state)
        min_dist, min_dist_loc = self.closest_goal()
        extreme_north = (max(0, (current_location[0] - ABSTRACT_STATE_WIDTH)), (current_location[1] + ABSTRACT_STATE_WIDTH) % STATE_WIDTH)
        extreme_south = (min(STATE_HEIGHT, (current_location[0] + ABSTRACT_STATE_WIDTH)), (current_location[1] + ABSTRACT_STATE_WIDTH) % STATE_WIDTH)
        if (abs(extreme_north[0] - min_dist_loc[0]) > min_dist) or (abs(extreme_south[0] - min_dist_loc[0]) > min_dist):
            return False
        return True

    #NOTE: This relaxes some assumptions from EOD, but also still relies on a bunch of EOD-specific stuff about how state is represented. Also, this relies on the constant, one-step motion of the agent east at every step. In the general case, the for loop over (min_dist - k) will have to be some general estimate of the number of actions needed to encounter the goal. Also, the goal being reached will need to be checked within that loop rather than once at the end, since the agent may reach it part way through the trajectory.
    def is_probably_kSR(self, k, n, m):
        #NOTE: could generate list of "goal" states a different way, or perhaps as an argument to the function instead
        min_dist, min_dist_loc = self.closest_goal()
        #TODO: what if goal is within k steps?

        possible_states_after_k_arbitrary_actions = []
        for i in range(n):
            current_state = self.current_ground_state
            for j in range(k):
                action = random.sample(ACTION_MAP.keys(), 1)[0]
                current_state = utils.get_successor_state(current_state, action, self.ground_mdp)
            possible_states_after_k_arbitrary_actions.append(current_state)
            curr_loc, curr_weather = self.ground_mdp.get_state_factors_from_state(current_state)

        #NOTE: could check this for dupes to make slightly faster
        reachable = [0] * len(possible_states_after_k_arbitrary_actions)
        for s in range(len(possible_states_after_k_arbitrary_actions)):
            state = possible_states_after_k_arbitrary_actions[s]
            goal_found = False
            for i in range(m):
                current_state = state
                for j in range(min_dist - k):
                    action = random.sample(ACTION_MAP.keys(), 1)[0]
                    current_state = utils.get_successor_state(current_state, action, self.ground_mdp)
        
                current_location, _ = self.ground_mdp.get_state_factors_from_state(current_state)
                if current_location == min_dist_loc:
                    goal_found = True
                    break
            if goal_found:
                reachable[s] = 1
          
        #print(reachable) 
        #kSR_prob = sum(reachable)            
        kSR_prob = float(sum(reachable) / len(reachable))            
        if min_dist < k:
            kSR_prob = "too close"

        return kSR_prob 
 
def main():
    random.seed(50)

    env = MetareasoningEnv()

    observation = env.reset()
    print("Observation:", observation)

    done = False
    while not done:
        action = 1
        prob_kSR = env.is_probably_kSR(ABSTRACT_STATE_WIDTH, 100, 100)
        print("SOFT")
        print(prob_kSR)
        kSR = env.is_kSR()
        print("HARD")
        print(kSR)
        if kSR:
            action = 0
        observation, reward, done, _ = env.step(action)
        #observation, reward, done, _ = env.step(1)
        print("Observation:", observation)
        print("Reward:", reward)
        print("Done:", done)


if __name__ == '__main__':
    main()
