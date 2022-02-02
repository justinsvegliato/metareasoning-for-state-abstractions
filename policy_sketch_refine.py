import logging
import time

import cplex_mdp_solver
from partially_abstract_mdp import PartiallyAbstractMDP

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


def sketch(abstract_mdp, gamma):
    return cplex_mdp_solver.solve(abstract_mdp, gamma)


def refine(ground_mdp, ground_state, abstract_mdp, abstract_state, sketched_solution, expand_points_of_interest, expansion_strategy, gamma):
    if expansion_strategy == 'NAIVE':
        return sketched_solution

    start = time.time()

    # TODO Definitely move this code to anywhere but here
    point_of_interest_locations = []
    point_of_interest_abstract_state_set = set()
    if expand_points_of_interest:
        current_location, current_weather_status = ground_mdp.get_state_factors_from_state(ground_state)
        for point_of_interest_location in current_weather_status:                
            vertical_distance = abs(current_location[0] - point_of_interest_location[0])
            horizontal_displacement = point_of_interest_location[1] - current_location[1]
            horizontal_distance = abs(horizontal_displacement) if horizontal_displacement >= 0 else ground_mdp.width() - abs(horizontal_displacement)
            if vertical_distance > abstract_mdp.abstract_state_height * 3 or horizontal_distance > abstract_mdp.abstract_state_width * 3:
                continue

            point_of_interest_ground_state = ground_mdp.get_state_from_state_factors(point_of_interest_location, current_weather_status)
            point_of_interest_abstract_state = abstract_mdp.get_abstract_state(point_of_interest_ground_state)
            point_of_interest_abstract_state_set.add(point_of_interest_abstract_state)
            point_of_interest_locations.append(point_of_interest_location)

            if expansion_strategy == 'PROACTIVE':
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
                      point_of_interest_abstract_state_set.add(point_of_interest_abstract_state)

        logging.info("---- Enabled point of interest abstract state expansion: [abstract_states=%s]", point_of_interest_locations)

    # TODO Yikes...
    grounding_abstract_states = list(set([abstract_state] + list(point_of_interest_abstract_state_set)))
    partially_abstract_mdp = PartiallyAbstractMDP(ground_mdp, abstract_mdp, grounding_abstract_states)
    logging.info("---- Built the PAMDP: [states=%d, actions=%d, time=%f]", len(partially_abstract_mdp.states()), len(partially_abstract_mdp.actions()), time.time() - start)

    start = time.time()
    refined_solution = cplex_mdp_solver.solve(partially_abstract_mdp, gamma)

    if refined_solution:
        logging.info("---- Ran the CPLEX solver: [time=%f]", time.time() - start)
    else:
        logging.info("---- Failed to run the CPLEX solver: [time=%f]", time.time() - start)

    return refined_solution

def solve(ground_mdp, ground_state, abstract_mdp, abstract_state, expand_points_of_interest, expansion_level, gamma):
    sketched_solution = sketch(abstract_mdp, gamma)
    refined_solution = refine(ground_mdp, ground_state, abstract_mdp, abstract_state, sketched_solution, expand_points_of_interest, expansion_level, gamma)
    return refined_solution
