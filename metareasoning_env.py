import copy
import logging
import math
import random
import statistics

import gym
import numpy as np
import scipy.stats
from gym import spaces

import cplex_mdp_solver
import earth_observation_mdp
import policy_sketch_refine
import printer
import utils
from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_mdp import EarthObservationMDP

# Ground MDP Settings
STATE_WIDTH = 12
STATE_HEIGHT = 6
SIZE = (STATE_HEIGHT, STATE_WIDTH)
POINTS_OF_INTEREST = 2
START_LOCATION = (0, 0)
START_VISIBILITY = earth_observation_mdp.MAX_VISIBILITY

# Abstract MDP Settings
ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 3
ABSTRACT_STATE_HEIGHT = 3

# Solution Method Settings
GAMMA = 0.99

# Simulator Settings
TRAVERSES = 20
HORIZON = TRAVERSES * STATE_WIDTH

# Time-Dependent Utility Settings
ALPHA = 1
BETA = 0.000002

# Policy Quality Calculation Settings
VALUE_FOCUS = 'INITIAL_GROUND_STATE'
VALUE_DETERMINATION = 'EXACT'
VALUE_NORMALIZATION = False
SIMULATIONS = 1000

# Helpful Mappings
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

# Logger Initialization
logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

class MetareasoningEnv(gym.Env):

    def __init__(self, ):
        super(MetareasoningEnv, self).__init__()

        self.observation_space = spaces.Box(
            # (1) Feature 1: The distance from the current ground state to the nearest ground state with a point of interest
            # (2) Feature 2: The number of PoI within a certain distance
            # (3) Feature 3: The distance d to the nearest PoI expressed as the function 1 / (1+|d-ABSTRACT_STATE_WIDTH|)
            # (4) Feature 4: The entropy of the abstract successor distribution for the current abstract state
            # (5) Feature 5: The normalized, discounted, state occupancy frequency of the current abstract state
            # (6) Feature 6: Whether or not the current ground state is (1.0) kSR wrt the closest PoI, or not (0.0) # NOTE The sampling version is also between 0.0 and 1.0 - not in use
            low=np.array([
                np.float32(0.0),
                np.float32(0.0),
                np.float32(0.0),
                np.float32(0.0),
                np.float32(0.0),
                np.float32(0.0)
            ]),
            high=np.array([
                np.float32(1.0),
                np.float32(1.0),
                np.float32(1.0),
                np.float32(math.log((STATE_WIDTH / ABSTRACT_STATE_WIDTH) * (STATE_HEIGHT / ABSTRACT_STATE_HEIGHT))),
                np.float32(1.0),
                np.float32(1.0)
            ]),
            shape=(6, )
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
        self.decisions = []

        self.current_ground_state = None
        self.current_abstract_state = None
        self.current_action = None
        self.current_step = None

        self.previous_computation_time = None
        self.current_computation_time = None
        self.previous_quality = None
        self.current_quality = None

    def step(self, action):
        logging.info("ENVIRONMENT STEP [%d, %s, %s]", self.current_step, EXPANSION_STRATEGY_MAP[action], self.current_abstract_state)

        logging.info("-- Executed the policy sketch refine algorithm")
        solution = policy_sketch_refine.solve(self.ground_mdp, self.current_ground_state, self.abstract_mdp, self.current_abstract_state, EXPANSION_STRATEGY_MAP[action], GAMMA)

        new_solved_ground_values = utils.get_values(solution['values'], self.ground_mdp, self.abstract_mdp)
        new_solved_ground_states = self.abstract_mdp.get_ground_states([self.current_abstract_state])
        new_solved_ground_policy = utils.get_ground_policy(new_solved_ground_values, self.ground_mdp, self.abstract_mdp, new_solved_ground_states, self.current_abstract_state, GAMMA)

        self.solved_ground_states += new_solved_ground_states
        for new_solved_ground_state in new_solved_ground_states:
            self.ground_policy_cache[new_solved_ground_state] = new_solved_ground_policy[new_solved_ground_state]
        logging.info("-- Updated the ground policy cache for the new abstract state: [%s]", self.current_abstract_state)

        self.decision_point_ground_state = self.current_ground_state
        self.decision_point_ground_states = new_solved_ground_states
        self.decision_point_abstract_state = self.current_abstract_state
        self.decisions.append({
            'expansion_strategy': EXPANSION_STRATEGY_MAP[action], 
            'state_space_size': solution['state_space_size'],
            'action_space_size': solution['action_space_size']
        })

        logging.info("SIMULATION")

        if self.current_ground_state not in self.solved_ground_states:
            logging.info(">>>> Encountered a ground state not in the solved ground states")

        if self.get_done():
            logging.info(">>>> Encountered a step greater than the horizon")

        while self.current_ground_state in self.solved_ground_states and not self.get_done():
            self.current_action = self.ground_policy_cache[self.current_ground_state]

            logging.info(">>>> Ground State: [%s] | Abstract State: [%s] | Action: [%s]", self.current_ground_state, self.current_abstract_state, self.current_action)
            printer.print_earth_observation_policy(self.ground_mdp, current_ground_state=self.current_ground_state, expanded_ground_states=self.abstract_mdp.get_ground_states([self.current_abstract_state]), ground_policy_cache=self.ground_policy_cache)

            self.current_ground_state = utils.get_successor_state(self.current_ground_state, self.current_action, self.ground_mdp)
            self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)
            self.current_step += 1

        self.previous_computation_time = self.current_computation_time
        self.current_computation_time = utils.get_computation_time(solution['state_space_size'], solution['action_space_size'])
        self.previous_quality = self.current_quality
        self.current_quality = self.__get_current_quality()

        return self.get_observation(), self.get_reward(), self.get_done(), self.get_info(self.decision_point_ground_state, self.decision_point_abstract_state, self.decisions)

    def reset(self):
        logging.info("ENVIRONMENT RESET")

        self.ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST)
        logging.info("-- Built the earth observation MDP: [states=%d, actions=%d]", len(self.ground_mdp.states()), len(self.ground_mdp.actions()))

        self.ground_memory_mdp = cplex_mdp_solver.MemoryMDP(self.ground_mdp, parallelize=True)
        logging.info("-- Built the earth observation memory MDP: [states=%d, actions=%d]", len(self.ground_memory_mdp.states), len(self.ground_memory_mdp.actions))

        self.abstract_mdp = EarthObservationAbstractMDP(self.ground_mdp, ABSTRACTION, ABSTRACT_STATE_WIDTH, ABSTRACT_STATE_HEIGHT)
        logging.info("-- Built the abstract earth observation MDP: [states=%d, actions=%d]", len(self.abstract_mdp.states()), len(self.abstract_mdp.actions()))

        self.abstract_solution = cplex_mdp_solver.solve(self.abstract_mdp, GAMMA)
        self.abstract_policy = utils.get_policy(self.abstract_solution['values'], self.abstract_mdp, GAMMA)
        logging.info("-- Solved the abstract earth observation MDP: [states=%d, actions=%d]", len(self.abstract_mdp.states()), len(self.abstract_mdp.actions()))

        if VALUE_NORMALIZATION:
            self.value_normalizer = self.__get_maximum_value()

        # NOTE This feature can be commented out if we're not using it
        self.abstract_occupancy_frequency = self.__calculate_abstract_occupancy_frequency() 

        self.solved_ground_states = []    
        self.ground_policy_cache = {}
        for ground_state in self.ground_mdp.states():
            self.ground_policy_cache[ground_state] = self.abstract_policy[self.abstract_mdp.get_abstract_state(ground_state)]
        logging.info("-- Built the ground policy cache from the abstract policy")    

        initial_location = START_LOCATION
        initial_point_of_interest_description = {key: START_VISIBILITY for key in self.ground_mdp.point_of_interest_description}
        self.initial_ground_state = self.ground_mdp.get_state_from_state_factors(initial_location, initial_point_of_interest_description)

        self.decision_point_ground_state = self.initial_ground_state
        self.decision_point_ground_states = [self.initial_ground_state]
        self.decision_point_abstract_state = None
        self.decisions = []

        self.current_ground_state = self.initial_ground_state
        self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)
        self.current_action = self.ground_policy_cache[self.current_ground_state]
        self.current_step = 0

        self.previous_computation_time = 0
        self.current_computation_time = 0
        self.previous_quality = 0
        self.current_quality = self.__get_current_quality()
        self.start_quality = self.current_quality

        logging.info("SIMULATION")
        logging.info(">>>> Ground State: [%s] | Abstract State: [%s] | Action: [%s]", self.current_ground_state, self.current_abstract_state, self.current_action)
        printer.print_earth_observation_policy(self.ground_mdp, current_ground_state=self.current_ground_state, expanded_ground_states=[], ground_policy_cache=self.ground_policy_cache)

        return self.get_observation()

    def get_observation(self):
        return np.array([
            np.float32(self.__get_current_reward_distance()), 
            np.float32(self.__get_num_close_rewards()), 
            np.float32(self.__face_check_goals()),
            np.float32(self.__get_entropy_of_abstract_successor_distribution()),
            np.float32(self.__get_abstract_occupancy_frequency(self.current_abstract_state)), 
            np.float32(self.__is_k_step_reachable())
        ])
    
    def get_reward(self):
        # current_time_dependent_utility = utils.get_time_dependent_utility(self.current_quality, self.current_computation_time, ALPHA, BETA, True)
        # previous_time_dependent_utility = utils.get_time_dependent_utility(self.previous_quality, self.previous_computation_time, ALPHA, BETA, True)
        # return current_time_dependent_utility - previous_time_dependent_utility 

        current_intrinsic_value = utils.get_intrinisic_value(self.current_quality, ALPHA)
        previous_intrinsic_value = utils.get_intrinisic_value(self.previous_quality, ALPHA)

        # NOTE The time cost of the current state is the sum of all time costs so far
        time_cost = utils.get_exponential_time_cost(self.current_computation_time, BETA) 

        return current_intrinsic_value - previous_intrinsic_value - time_cost

    def get_done(self):
        return self.current_step > HORIZON

    def get_info(self, ground_state, abstract_state, decisions):
        return {
            'ground_state': ground_state,
            'abstract_state': abstract_state,
            'decisions': [decision['expansion_strategy'] for decision in decisions]
        }

    def __get_exact_values(self):
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

    def __get_approximate_values(self, states):
        monte_carlo_value_container = {state: [] for state in states}

        for state in states:
            for _ in range(SIMULATIONS):
                monte_carlo_value = 0

                simulated_state = state
                for _ in range(HORIZON):
                    simulated_action = self.ground_policy_cache[simulated_state]
                    monte_carlo_value += self.ground_mdp.reward_function(simulated_state, simulated_action)
                    simulated_state = utils.get_successor_state(simulated_state, simulated_action, self.ground_mdp)
                
                monte_carlo_value_container[state].append(monte_carlo_value)

        return {state: statistics.mean(monte_carlo_value_container[state]) for state in states}

    # TODO Improve this with a given state and its reachability X
    # TODO Improve this with the expected point of interest weather X
    # TODO Improve this with discounting X
    def __get_maximum_value(self):
        average_reward = [1, 4, 7, 10] / earth_observation_mdp.VISIBILITY_FIDELITY

        maximum_photo_count = (HORIZON / STATE_WIDTH) * POINTS_OF_INTEREST

        return average_reward * maximum_photo_count

    def __get_current_quality(self):
        current_quality = None

        if VALUE_FOCUS == 'INITIAL_GROUND_STATE':
            values = self.__get_exact_values() if VALUE_DETERMINATION == 'EXACT' else self.__get_approximate_values([self.initial_ground_state])
            current_quality = values[self.initial_ground_state]

        if VALUE_FOCUS == 'SINGLE_DECISION_POINT_GROUND_STATE':
            values = self.__get_exact_values() if VALUE_DETERMINATION == 'EXACT' else self.__get_approximate_values([self.decision_point_ground_state])
            current_quality = values[self.decision_point_ground_state]

        if VALUE_FOCUS == 'ALL_DECISION_POINT_GROUND_STATES':
            values = self.__get_exact_values() if VALUE_DETERMINATION == 'EXACT' else self.__get_approximate_values(self.decision_point_ground_states)
            current_quality = statistics.mean([values[decision_point_ground_state] for decision_point_ground_state in self.decision_point_ground_states])

        return current_quality / self.value_normalizer if VALUE_NORMALIZATION else current_quality

    def __get_current_reward_distance(self):
        current_location, current_weather_status = self.ground_mdp.get_state_factors_from_state(self.current_ground_state)

        current_reward_distance = float('inf')
        
        for point_of_interest_location in current_weather_status:
            vertical_distance = abs(current_location[0] - point_of_interest_location[0])
            horizontal_displacement = point_of_interest_location[1] - current_location[1]
            horizontal_distance = abs(horizontal_displacement) if horizontal_displacement >= 0 else self.ground_mdp.width() - abs(horizontal_displacement)
            manhattan_distance = vertical_distance + horizontal_distance
 
            current_reward_distance = min(current_reward_distance, manhattan_distance)

        return float(current_reward_distance / STATE_WIDTH)

    # TODO Generalize to reachable points of interest
    def __get_closest_goal(self):
        current_location, current_weather_status = self.ground_mdp.get_state_factors_from_state(self.current_ground_state)

        minimum_distance = math.inf
        minimum_distance_location = None

        for key in current_weather_status.keys():
            distance = 0
            if key[1] > current_location[1]:
                distance = key[1] - current_location[1]
            else:
                distance = (key[1] + STATE_WIDTH) - current_location[1]

            if distance < minimum_distance:
                minimum_distance = distance
                minimum_distance_location = key

        return minimum_distance, minimum_distance_location

    def __is_k_step_reachable(self):
        current_location, _ = self.ground_mdp.get_state_factors_from_state(self.current_ground_state)

        minimum_distance, minimum_distance_location = self.__get_closest_goal()

        extreme_north = (max(0, (current_location[0] - ABSTRACT_STATE_WIDTH)), (current_location[1] + ABSTRACT_STATE_WIDTH) % STATE_WIDTH)
        extreme_south = (min(STATE_HEIGHT - 1, (current_location[0] + ABSTRACT_STATE_WIDTH)), (current_location[1] + ABSTRACT_STATE_WIDTH) % STATE_WIDTH)

        if (abs(extreme_north[0] - minimum_distance_location[0]) > minimum_distance) or (abs(extreme_south[0] - minimum_distance_location[0]) > minimum_distance):
            return False

        return True
 
    def __get_num_close_rewards(self):
        current_location, current_weather_status = self.ground_mdp.get_state_factors_from_state(self.current_ground_state)

        num_close_goals = 0
        for key in current_weather_status.keys():
            distance = 0
            if key[1] > current_location[1]:
                distance = key[1] - current_location[1]
            else:
                distance = (key[1] + STATE_WIDTH) - current_location[1]

            if distance < 2 * ABSTRACT_STATE_WIDTH:
                num_close_goals += 1

        return float(num_close_goals / POINTS_OF_INTEREST)

    def __face_check_goals(self):
        minimum_distance, _ = self.__get_closest_goal()
        return 1.0 / (1.0 + abs(minimum_distance - ABSTRACT_STATE_WIDTH))

    def __get_entropy_of_abstract_successor_distribution(self):
        states = self.abstract_mdp.states()
        action = self.abstract_policy[self.current_abstract_state]

        successor_distribution = np.zeros(len(states))
        for index, successor_state in enumerate(states):
            successor_distribution[index] = self.abstract_mdp.transition_function(self.current_abstract_state, action, successor_state)

        return scipy.stats.entropy(successor_distribution)

    # NOTE Can we get this from the policy/value function without re-solving the problem? 
    # The only ways I could figure out doing this involved either matrix inverses or 
    # something that looks like value iteration, which is what I programmed.
    def __calculate_abstract_occupancy_frequency(self):
        start_state_distribution = np.full((len(self.abstract_mdp.states())), 1.0 / len(self.abstract_mdp.states()))

        previous_occupancy_frequency = np.zeros(len(self.abstract_mdp.states()))
        occupancy_frequency = np.zeros(len(self.abstract_mdp.states()))

        epsilon = 0.001
        
        done = False
        while not done:
            # NOTE These are start state probabilities that could change if we wanted to
            occupancy_frequency = copy.deepcopy(start_state_distribution)

            for index, state in enumerate(self.abstract_mdp.states()):
                action = self.abstract_policy[state]
                for successor_state_index, successor_state in enumerate(self.abstract_mdp.states()):
                    occupancy_frequency[successor_state_index] += previous_occupancy_frequency[index] * GAMMA * self.abstract_mdp.transition_function(state, action, successor_state)

            maximum_difference = np.max(np.abs(np.subtract(previous_occupancy_frequency, occupancy_frequency)))
            previous_occupancy_frequency = copy.deepcopy(occupancy_frequency)
            if maximum_difference < epsilon:
                done = True

        occupancy_frequency = occupancy_frequency / np.linalg.norm(occupancy_frequency)

        return {state: occupancy_frequency[index] for index, state in enumerate(self.abstract_mdp.states())}

    def __get_abstract_occupancy_frequency(self, abstract_state):
        return self.abstract_occupancy_frequency[abstract_state]

    # NOTE This relaxes some assumptions from EOD, but also still relies on a bunch of EOD-specific 
    # stuff about how state is represented. Also, this relies on the constant, one-step motion of 
    # the agent east at every step. In the general case, the for loop over (min_dist - k) will have 
    # to be some general estimate of the number of actions needed to encounter the goal. Also, the 
    # goal being reached will need to be checked within that loop rather than once at the end, since 
    # the agent may reach it part way through the trajectory.
    def __is_probably_k_step_reachable(self, k, n, m):
        # NOTE could generate list of "goal" states a different way, or perhaps as an argument to the function instead
        min_dist, min_dist_loc = self.__get_closest_goal()
        possible_states_after_k_arbitrary_actions = []
        for _ in range(n):
            current_state = self.current_ground_state
            for _ in range(k):
                action = random.sample(ACTION_MAP.keys(), 1)[0]
                current_state = utils.get_successor_state(current_state, action, self.ground_mdp)
            possible_states_after_k_arbitrary_actions.append(current_state)

        # NOTE could check this for dupes to make slightly faster
        reachable = [0] * len(possible_states_after_k_arbitrary_actions)
        for s in range(len(possible_states_after_k_arbitrary_actions)):
            state = possible_states_after_k_arbitrary_actions[s]
            goal_found = False
            for _ in range(m):
                current_state = state
                for _ in range(min_dist - k):
                    action = random.sample(ACTION_MAP.keys(), 1)[0]
                    current_state = utils.get_successor_state(current_state, action, self.ground_mdp)
        
                current_location, _ = self.ground_mdp.get_state_factors_from_state(current_state)
                if current_location == min_dist_loc:
                    goal_found = True
                    break
            if goal_found:
                reachable[s] = 1
          
        kSR_prob = float(sum(reachable) / len(reachable))            
        if min_dist < k:
            kSR_prob = "too close"

        return kSR_prob 


def main():
    pure_naive_rewards = []
    pure_proactive_rewards = []
    hard_k_step_reachable_rewards = []
    
    for seed in range(10):
        random.seed(seed)

        # env = MetareasoningEnv()
        # observation = env.reset()
        # print("Observation:", observation)
        # done = False
        # while not done:
        #     action = 0
        #     observation, reward, done, _ = env.step(action)
        #     print("Observation:", observation)
        #     print("Reward:", reward)
        #     print("Done:", done)
        #     pure_naive_rewards.append(reward)

        # env = MetareasoningEnv()
        # observation = env.reset()
        # print("Observation:", observation)
        # done = False
        # while not done:
        #     action = 1
        #     observation, reward, done, _ = env.step(action)
        #     print("Observation:", observation)
        #     print("Reward:", reward)
        #     print("Done:", done)
        #     pure_proactive_rewards.append(reward)

        env = MetareasoningEnv()
        observation = env.reset()
        print("Observation:", observation)
        done = False
        while not done:
            hard_k_step_reachable = env.__is_k_step_reachable()
            print("k-Step Reachable:", hard_k_step_reachable)
        
            face_check_score = env.__face_check_goals()
            print("Face Check Score", face_check_score)
            
            num_close_rewards = env.__get_num_close_rewards()
            print("Num Close Rewards:", num_close_rewards)

            entropy = env.__get_entropy_of_abstract_successor_distribution()
            print("Entropy:", entropy)

            occupancy_frequency = env.__get_abstract_occupancy_frequency(env.current_abstract_state)
            print("Occupancy Frequency:", occupancy_frequency)

            action = 0 if hard_k_step_reachable else 1
            observation, reward, done, _ = env.step(action)
            print("Observation:", observation)
            print("Reward:", reward)
            print("Done:", done)
            hard_k_step_reachable_rewards.append(reward)

    print("Pure Naive Rewards:", sum(pure_naive_rewards))
    print("Pure Proactive Rewards:", sum(pure_proactive_rewards))
    print("Hard k-Step Reachable Rewards:", sum(hard_k_step_reachable_rewards))


if __name__ == '__main__':
    main()
