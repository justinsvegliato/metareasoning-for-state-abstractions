import random


def get_ground_values(values, ground_mdp, abstract_mdp):
    ground_values = {}

    for ground_state in ground_mdp.states():
        if ground_state in values:
            ground_values[ground_state] = values[ground_state]
        else:
            abstract_state = abstract_mdp.get_abstract_state(ground_state)
            ground_values[ground_state] = values[abstract_state]

    return ground_values


def get_ground_policy(values, ground_mdp, abstract_mdp, ground_states, abstract_state, gamma):
    policy = {}

    for state in ground_states:
        best_action = None
        best_action_value = None

        for action in ground_mdp.actions():
            immediate_reward = ground_mdp.reward_function(state, action)

            expected_future_reward = 0
            for successor_abstract_state in abstract_mdp.states():
                if abstract_mdp.transition_function(abstract_state, action, successor_abstract_state) > 0:
                    for successor_state in abstract_mdp.get_ground_states([successor_abstract_state]):
                        expected_future_reward += ground_mdp.transition_function(state, action, successor_state) * values[successor_state]

            action_value = immediate_reward + gamma * expected_future_reward

            if best_action_value is None or action_value > best_action_value:
                best_action = action
                best_action_value = action_value

        policy[state] = best_action

    return policy

def get_full_ground_policy(values, ground_mdp, ground_states, gamma):
    policy = {}

    for state in ground_states:
        best_action = None
        best_action_value = None

        for action in ground_mdp.actions():
            immediate_reward = ground_mdp.reward_function(state, action)

            expected_future_reward = 0
            for successor_state in ground_mdp.states():
                if ground_mdp.transition_function(state, action, successor_state) > 0:
                    expected_future_reward += ground_mdp.transition_function(state, action, successor_state) * values[successor_state]

            action_value = immediate_reward + gamma * expected_future_reward

            if best_action_value is None or action_value > best_action_value:
                best_action = action
                best_action_value = action_value

        policy[state] = best_action

    return policy


def get_successor_state(current_state, current_action, mdp):
    probability_threshold = random.random()

    total_probability = 0

    successor_states = mdp.states()
    for successor_state in successor_states:
        transition_probability = mdp.transition_function(current_state, current_action, successor_state)

        if transition_probability == 0:
            continue

        total_probability += transition_probability

        if total_probability >= probability_threshold:
            return successor_state

    return False


def get_partitions(l, num_partitions):
    return [l[i:i + num_partitions] for i in range(0, len(l), num_partitions)]
