import sys

from termcolor import colored

IS_ACTIVE = False

SYMBOLS = {
    0: '\u00b7',
    1: '\u205a',
    2: '\u22ee',
    3: '\u229e',
    'STAY': '\u2205',
    'NORTH': '\u2191',
    'SOUTH': '\u2193',
    'IMAGE': '\u2726',
    'NULL': '\u2A09'
}

BORDER_CHARACTER = "="
BORDER_SIZE = 150


def print_states(mdp):
    print("States:")

    for index, state in enumerate(mdp.states()):
        print(f"  State {index}: {state}")


def print_actions(mdp):
    print("Actions:")

    for index, action in enumerate(mdp.actions()):
        print(f"  Action {index}: {action}")


def print_transition_function(mdp):
    print("Transition Function:")

    is_valid = True

    for state in mdp.states():
        for action in mdp.actions():
            print(f"  Transition: ({state}, {action})")

            total_probability = 0

            for successor_state in mdp.states():
                probability = mdp.transition_function(
                    state, action, successor_state)

                total_probability += probability

                if probability > 0:
                    print(
                        f"    Successor State: {successor_state} -> {probability}")

            is_valid = is_valid and 0.99 <= total_probability <= 1.01
            print(f"    Total Probability: {total_probability}")

            if not is_valid:
                return

    print(f"  Is Valid: {is_valid}")


def print_reward_function(mdp):
    print("Reward Function:")

    for state in mdp.states():
        print(f"  State: {mdp.state_factors_from_int(state)}")

        for action in mdp.actions():
            reward = mdp.reward_function(state, action)
            print(f"    Action: {action} -> {reward}")


def print_start_state_function(mdp):
    print("Start State Function:")

    total_probability = 0

    for state in mdp.states():
        probability = mdp.start_state_function(state)
        total_probability += probability
        print(f"  State {state}: {probability}")

    print(f"  Total Probability: {total_probability}")

    is_valid = total_probability == 1.0
    print(f"  Is Valid: {is_valid}")


def print_mdp(mdp):
    print_states(mdp)
    print_actions(mdp)
    print_transition_function(mdp)
    print_reward_function(mdp)
    print_start_state_function(mdp)


def print_earth_observation_policy(earth_observation_mdp, current_ground_state, expanded_ground_states=[], ground_policy_cache={}):
    if not IS_ACTIVE:
        return False

    print_border()

    height = earth_observation_mdp.height()
    width = earth_observation_mdp.width()

    for row in range(height):
        text = ""

        for column in range(width):
            location = (row, column)

            _, current_poi_weather = earth_observation_mdp.get_state_factors_from_state(current_ground_state)

            state = earth_observation_mdp.get_state_from_state_factors(location, current_poi_weather)

            symbol = None
            if location in current_poi_weather:
                weather_symbol = SYMBOLS[current_poi_weather[location]]
                symbol = weather_symbol
            else:
                if state in ground_policy_cache:
                    action = ground_policy_cache[state]
                    symbol = SYMBOLS[action]
                else:
                    symbol = SYMBOLS['NULL']

            if state == current_ground_state:
                symbol = colored(symbol, 'red')
            elif state in expanded_ground_states:
                symbol = colored(symbol, 'green')
            elif state in ground_policy_cache:
                symbol = colored(symbol, 'blue')

            text += symbol
            text += "  "

        print(f"{text}")

    print_border()


def print_loading_bar(count, total, label):
    maximum_loading_bar_length = 60
    current_loading_bar_length = int(round(maximum_loading_bar_length * count / float(total)))

    percent = round(100.0 * count / float(total), 1)
    loading_bar = '#' * current_loading_bar_length + '-' * (maximum_loading_bar_length - current_loading_bar_length)

    sys.stdout.write('%s: [%s] %s%s %s\r' % (label, loading_bar, percent, '%', ''))
    sys.stdout.flush()


def print_border():
    print(BORDER_CHARACTER * BORDER_SIZE)


def padder(header, border_size):
    return header + " " + BORDER_CHARACTER * (border_size - len(header) - 1)
