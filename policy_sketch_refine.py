import logging

import cplex_mdp_solver
from partially_abstract_mdp import PartiallyAbstractMDP

GREEDY_LOOKAHEAD_MULTIPLIER = 1
PROACTIVE_LOOKAHEAD_MULTIPLIER = 2

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


def is_relevant(ground_mdp, abstract_mdp, current_location, point_of_interest_location, lookahead_multiplier):
    vertical_distance = abs(current_location[0] - point_of_interest_location[0])
    horizontal_displacement = point_of_interest_location[1] - current_location[1]
    horizontal_distance = abs(horizontal_displacement) if horizontal_displacement >= 0 else ground_mdp.width() - abs(horizontal_displacement)
    if vertical_distance > abstract_mdp.abstract_state_height * lookahead_multiplier or horizontal_distance > abstract_mdp.abstract_state_width * lookahead_multiplier:
        return False

    return True


def get_greedy_point_of_interest_abstract_states(ground_mdp, abstract_mdp, current_location, current_weather_status):
    point_of_interest_abstract_states = set()

    for point_of_interest_location in current_weather_status:
        if not is_relevant(ground_mdp, abstract_mdp, current_location, point_of_interest_location, GREEDY_LOOKAHEAD_MULTIPLIER):
            continue

        point_of_interest_ground_state = ground_mdp.get_state_from_state_factors(point_of_interest_location, current_weather_status)
        point_of_interest_abstract_state = abstract_mdp.get_abstract_state(point_of_interest_ground_state)
        point_of_interest_abstract_states.add(point_of_interest_abstract_state)

    return point_of_interest_abstract_states


def get_proactive_point_of_interest_abstract_states(ground_mdp, abstract_mdp, current_location, current_weather_status):
    point_of_interest_abstract_states = set()

    for point_of_interest_location in current_weather_status:
        if not is_relevant(ground_mdp, abstract_mdp, current_location, point_of_interest_location, PROACTIVE_LOOKAHEAD_MULTIPLIER):
            continue

        x_range = []
        if current_location[1] < point_of_interest_location[1]:
            x_range += range(current_location[1], point_of_interest_location[1] + 1)
        else:
            x_range += range(current_location[1], ground_mdp.width())
            x_range += range(0, point_of_interest_location[1] + 1)

        y_range = []
        if current_location[0] < point_of_interest_location[0]:
            y_range += range(current_location[0], point_of_interest_location[0] + 1)
        else:
            y_range += range(point_of_interest_location[0], current_location[0] + 1)

        for x in x_range:
            for y in y_range:
                point_of_interest_ground_state = ground_mdp.get_state_from_state_factors((y, x), current_weather_status)
                point_of_interest_abstract_state = abstract_mdp.get_abstract_state(point_of_interest_ground_state)
                point_of_interest_abstract_states.add(point_of_interest_abstract_state)

    return point_of_interest_abstract_states


def sketch(abstract_mdp, gamma):
    return cplex_mdp_solver.solve(abstract_mdp, gamma)


def refine(ground_mdp, ground_state, abstract_mdp, abstract_state, sketched_solution, expansion_strategy, gamma):
    if expansion_strategy == 'NONE':
        sketched_solution['state_space_size'] = abstract_mdp.state_space_size
        sketched_solution['action_space_size'] = abstract_mdp.action_space_size
        return sketched_solution

    current_location, current_weather_status = ground_mdp.get_state_factors_from_state(ground_state)

    point_of_interest_abstract_state_set = set([abstract_state])

    if expansion_strategy == 'GREEDY' or expansion_strategy == 'PROACTIVE':
        point_of_interest_abstract_state_set.update(get_greedy_point_of_interest_abstract_states(ground_mdp, abstract_mdp, current_location, current_weather_status))
    
    if expansion_strategy == 'PROACTIVE':
        point_of_interest_abstract_state_set.update(get_proactive_point_of_interest_abstract_states(ground_mdp, abstract_mdp, current_location, current_weather_status))

    logging.info("---- Expanded the abstract states: %s", point_of_interest_abstract_state_set)

    grounding_abstract_states = list(point_of_interest_abstract_state_set)
    partially_abstract_mdp = PartiallyAbstractMDP(ground_mdp, abstract_mdp, grounding_abstract_states)
    logging.info("---- Built the PAMDP: [states=%d, actions=%d]", len(partially_abstract_mdp.states()), len(partially_abstract_mdp.actions()))

    refined_solution = cplex_mdp_solver.solve(partially_abstract_mdp, gamma)
    
    if refined_solution:
        logging.info("---- Ran the CPLEX solver")
    else:
        logging.info("---- Failed to run the CPLEX solver")

    refined_solution['state_space_size'] = partially_abstract_mdp.state_space_size
    refined_solution['action_space_size'] = partially_abstract_mdp.action_space_size
    return refined_solution


def solve(ground_mdp, ground_state, abstract_mdp, abstract_state, expansion_strategy, gamma):
    sketched_solution = sketch(abstract_mdp, gamma)
    refined_solution = refine(ground_mdp, ground_state, abstract_mdp, abstract_state, sketched_solution, expansion_strategy, gamma)
    return refined_solution
