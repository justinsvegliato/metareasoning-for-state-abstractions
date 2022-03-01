import random

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

import cplex_mdp_solver
import earth_observation_mdp
import utils
from earth_observation_mdp import EarthObservationMDP
from metareasoning_env import MetareasoningEnv

# Ground MDP Settings
STATE_WIDTH = 12
STATE_HEIGHT = 6
SIZE = (STATE_HEIGHT, STATE_WIDTH)
POINTS_OF_INTEREST = 2
START_LOCATION = (0, 0)
START_VISIBILITY = earth_observation_mdp.MAX_VISIBILITY

# Solution Method Settings
GAMMA = 0.99

# Simulator Settings
TRAVERSES = 20
HORIZON = TRAVERSES * STATE_WIDTH

# Helpful Mappings
ACTION_MAP = {
    'STAY': 0,
    'NORTH': 1,
    'SOUTH': 2,
    'IMAGE': 3
}

EXPANSION_STRATEGY_MAP = {
    0: 'NAIVE',
    1: 'GREEDY',
    2: 'PROACTIVE'
}

MODEL_PATH = 'models/dqn-rural-feather-155-[final]'

JUST_PLOTTING = False

def testPureNaive(seed_min, seed_max):
    times = []
    meta_rewards = []
    ground_rewards = []
    individual_times = []
    for seed in range(seed_min, seed_max):
        random.seed(seed)
        print("\n\n\n")
        print("Running PURE NAIVE test, instance "+str(seed))
        print("\n\n\n")

        env = MetareasoningEnv()
        observation = env.reset()
        done = False
        total_time = 0
        total_meta_reward = 0 
        single_run_times = []
        while not done:
            action = 0
            observation, reward, done, _ = env.step(action)
            total_time += env.time
            single_run_times.append(env.time)
            total_meta_reward += reward
        
        individual_times.append(single_run_times)
        times.append(total_time)
        meta_rewards.append(total_meta_reward)
        ground_rewards.append(env.ground_reward)
    
    return ground_rewards, times, individual_times, meta_rewards

def testPureGreedy(seed_min, seed_max):
    times = []
    meta_rewards = []
    ground_rewards = []
    individual_times = []
    for seed in range(seed_min, seed_max):
        random.seed(seed)
        print("\n\n\n")
        print("Running PURE GREEDY test, instance "+str(seed))
        print("\n\n\n")

        env = MetareasoningEnv()
        observation = env.reset()
        done = False
        total_time = 0
        total_meta_reward = 0 
        single_run_individual_times = []
        while not done:
            action = 1
            observation, reward, done, _ = env.step(action)
            total_time += env.time
            single_run_individual_times.append(env.time)
            total_meta_reward += reward
        
        individual_times.append(single_run_individual_times)
        times.append(total_time)
        meta_rewards.append(total_meta_reward)
        ground_rewards.append(env.ground_reward)

    return ground_rewards, times, individual_times, meta_rewards

def testPureProactive(seed_min, seed_max):
    times = []
    meta_rewards = []
    ground_rewards = []
    individual_times = []
    for seed in range(seed_min, seed_max):
        random.seed(seed)
        print("\n\n\n")
        print("Running PURE PROACTIVE test, instance "+str(seed))
        print("\n\n\n")
    
        env = MetareasoningEnv()
        observation = env.reset()
        done = False
        total_time = 0
        total_meta_reward = 0 
        single_run_individual_times = []
        while not done:
            action = 2
            observation, reward, done, _ = env.step(action)
            total_time += env.time
            single_run_individual_times.append(env.time)
            total_meta_reward += reward
        
        individual_times.append(single_run_individual_times)
        times.append(total_time)
        meta_rewards.append(total_meta_reward)
        ground_rewards.append(env.ground_reward)

    return ground_rewards, times, individual_times, meta_rewards

def testHardKER(seed_min, seed_max):
    times = []
    meta_rewards = []
    ground_rewards = []
    individual_times = []
    for seed in range(seed_min, seed_max):
        random.seed(seed)
        print("\n\n\n")
        print("Running HARD KER test, instance "+str(seed))
        print("\n\n\n")
    
        env = MetareasoningEnv()
        observation = env.reset()
        done = False
        total_time = 0
        total_meta_reward = 0 
        single_run_individual_times = []
        while not done:
            action = 1
            kSR = env.is_k_step_reachable()
            occupancy_frequency = env.get_abstract_occupancy_frequency(env.current_abstract_state)
            if kSR:
                action = 0
            elif occupancy_frequency > 0.5:
                action = 2

            observation, reward, done, _ = env.step(action)
            total_time += env.time
            single_run_individual_times.append(env.time)
            total_meta_reward += reward
        
        individual_times.append(single_run_individual_times)
        times.append(total_time)
        meta_rewards.append(total_meta_reward)
        ground_rewards.append(env.ground_reward)

    return ground_rewards, times, individual_times, meta_rewards

# def testSoftKER(seed_min, seed_max):
#     times = []
#     meta_rewards = []
#     ground_rewards = []
#     individual_times = []
#     for seed in range(seed_min, seed_max):
#         random.seed(seed)
#         print("\n\n\n")
#         print("Running SOFT KER test, instance "+str(seed))
#         print("\n\n\n")
    
#         env = MetareasoningEnv()
#         observation = env.reset()
#         done = False
#         total_time = 0
#         total_meta_reward = 0 
#         single_run_individual_hard_kSR_times = []
#         while not done:
#             action = 1
#             #TODO: use this to stochastically choose expansion strategy
#             prob_kSR = env.is_probably_k_step_reachable(ABSTRACT_STATE_WIDTH, 100, 100)
#             occupancy_frequency = env.get_abstract_occupancy_frequency(env.current_abstract_state)
#             if kSR:
#                 action = 0
#             elif occupancy_frequency > 0.5:
#                 action = 2

#             observation, reward, done, _ = env.step(action)
#             print("Observation:", observation)
#             print("Reward:", reward)
#             total_time += env.time
#             single_run_individual_times.append(env.time)
#             total_meta_reward += reward
        
#         individual_times.append(single_run_individual_times)
#         times.append(total_time)
#         meta_rewards.append(total_meta_reward)
#         ground_rewards.append(env.ground_reward)

#     return ground_rewards, times, individual_times, meta_rewards

def testDQN(seed_min, seed_max):
    times = []
    meta_rewards = []
    ground_rewards = []
    individual_times = []
    for seed in range(seed_min, seed_max):
        random.seed(seed)
        print("\n\n\n")
        print("Running DQN test, instance "+str(seed))
        print("\n\n\n")

        model = DQN.load(MODEL_PATH)
 
        env = MetareasoningEnv()
        observation = env.reset()
        done = False
        total_time = 0
        total_meta_reward = 0 
        single_run_individual_times = []
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(int(action))
            total_time += env.time
            single_run_individual_times.append(env.time)
            total_meta_reward += reward
    
        individual_times.append(single_run_individual_times)
        times.append(total_time)
        meta_rewards.append(total_meta_reward)
        ground_rewards.append(env.ground_reward)

    return ground_rewards, times, individual_times, meta_rewards

def testGround(seed_min, seed_max):
    ground_rewards = []
    times = []
    
    for seed in range(seed_min, seed_max):
        random.seed(seed)
        print("\n\n\n")
        print("Running GROUND test, instance "+str(seed))
        print("\n\n\n")
        
        ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST)
        ground_memory_mdp = cplex_mdp_solver.MemoryMDP(ground_mdp, parallelize=True)
        solution = cplex_mdp_solver.solve(ground_mdp, GAMMA)
        policy = utils.get_policy(solution['values'], ground_mdp, GAMMA)
        
        initial_location = (0, 0)
        initial_point_of_interest_description = {key: earth_observation_mdp.MAX_VISIBILITY for key in ground_mdp.point_of_interest_description}
        initial_ground_state = ground_mdp.get_state_from_state_factors(initial_location, initial_point_of_interest_description)
        total_time = len(ACTION_MAP) * len(solution['values']) * len(solution['values'])
        total_ground_reward = 0
        current_state = initial_ground_state
        for _ in range(HORIZON+1): # There is an off-by-one error between this function and the way MetareasoningEnv uses HORIZON. Some EOD instances allow the agent to earn reward with the final action, which the ground MDP would miss out on otherwise.
            action = policy[current_state]
            total_ground_reward += ground_mdp.reward_function(current_state, action)
            current_state = utils.get_successor_state(current_state, action, ground_mdp)

        ground_rewards.append(total_ground_reward)
        times.append(total_time)

    return ground_rewards, times

def main():

    timing_file_name = "timing_results.txt"
    reward_file_name = "reward_results.txt"

    use_PN = False
    use_PG = False
    use_PP = False
    use_HK = False
    use_SK = False
    use_DQ = True
    use_GG = False

    PN_time = []
    PN_ind_time = []
    PN_meta_reward = []
    PN_ground_reward = []
    PG_time = []
    PG_ind_time = []
    PG_meta_reward = []
    PG_ground_reward = []
    PP_time = []
    PP_ind_time = []
    PP_meta_reward = []
    PP_ground_reward = []
    HK_time = []
    HK_ind_time = []
    HK_meta_reward = []
    HK_ground_reward = []
    SK_time = []
    SK_ind_time = []
    SK_meta_reward = []
    SK_ground_reward = []
    DQ_time = []
    DQ_ind_time = []
    DQ_meta_reward = []
    DQ_ground_reward = []
    GG_time = []
    GG_ground_reward = []

    if not JUST_PLOTTING:

        seed_min = 51
        seed_max = 101
   
        if use_PN:
            PN_ground_reward, PN_time, PN_ind_time, PN_meta_reward = testPureNaive(seed_min, seed_max)
        if use_PG:
            PG_ground_reward, PG_time, PG_ind_time, PG_meta_reward = testPureGreedy(seed_min, seed_max)
        if use_PP:
            PP_ground_reward, PP_time, PP_ind_time, PP_meta_reward = testPureProactive(seed_min, seed_max)
        if use_HK:
            HK_ground_reward, HK_time, HK_ind_time, HK_meta_reward = testHardKER(seed_min, seed_max)
        if use_SK:
            SK_ground_reward, SK_time, SK_ind_time, SK_meta_reward = testSoftKER(seed_min, seed_max)
        if use_DQ:
            DQ_ground_reward, DQ_time, DQ_ind_time, DQ_meta_reward = testDQN(seed_min, seed_max)
        if use_GG:
            GG_ground_reward, GG_time = testGround(seed_min, seed_max)
 
        if use_GG:
            for i in range(len(GG_ground_reward)):
                if use_PN:
                    PN_ground_reward[i] = PN_ground_reward[i] / GG_ground_reward[i]
                if use_PG:
                    PG_ground_reward[i] = PG_ground_reward[i] / GG_ground_reward[i]
                if use_PP:
                    PP_ground_reward[i] = PP_ground_reward[i] / GG_ground_reward[i]
                if use_HK:
                    HK_ground_reward[i] = HK_ground_reward[i] / GG_ground_reward[i]
                if use_SK:
                    SK_ground_reward[i] = SK_ground_reward[i] / GG_ground_reward[i]
                if use_DQ:
                    DQ_ground_reward[i] = DQ_ground_reward[i] / GG_ground_reward[i]

        reward_file = open(reward_file_name, "a")
        if use_GG:
            reward_file.write(str(GG_ground_reward))
            reward_file.write("\n")
        if use_PN:
            reward_file.write(str(PN_ground_reward))
            reward_file.write("\n")
            reward_file.write(str(PN_meta_reward))
            reward_file.write("\n")
        if use_PG:
            reward_file.write(str(PG_ground_reward))
            reward_file.write("\n")
            reward_file.write(str(PG_meta_reward))
            reward_file.write("\n")
        if use_PP:
            reward_file.write(str(PP_ground_reward))
            reward_file.write("\n")
            reward_file.write(str(PP_meta_reward))
            reward_file.write("\n")
        if use_HK:
            reward_file.write(str(HK_ground_reward))
            reward_file.write("\n")
            reward_file.write(str(HK_meta_reward))
            reward_file.write("\n")
        if use_DQ:
            reward_file.write(str(DQ_ground_reward))
            reward_file.write("\n")
            reward_file.write(str(DQ_meta_reward))
            reward_file.write("\n")
        reward_file.close()

        time_file = open(timing_file_name, "a")
        if use_GG:
            time_file.write(str(GG_time))
            time_file.write("\n")
        if use_PN:
            time_file.write(str(PN_time))
            time_file.write("\n")
            time_file.write(str(PN_ind_time))
            time_file.write("\n")
        if use_PG:
            time_file.write(str(PG_time))
            time_file.write("\n")
            time_file.write(str(PG_ind_time))
            time_file.write("\n")
        if use_PP:
            time_file.write(str(PP_time))
            time_file.write("\n")
            time_file.write(str(PP_ind_time))
            time_file.write("\n")
        if use_HK:
            time_file.write(str(HK_time))
            time_file.write("\n")
            time_file.write(str(HK_ind_time))
            time_file.write("\n")
        if use_DQ:
            time_file.write(str(DQ_time))
            time_file.write("\n")
            time_file.write(str(DQ_ind_time))
            time_file.close()


    if len(PN_meta_reward) > 0:
        print("AVG Meta Reward PN: "+str(sum(PN_meta_reward) / len(PN_meta_reward)))
    if len(PG_meta_reward) > 0:
        print("AVG Meta Reward PG: "+str(sum(PG_meta_reward) / len(PG_meta_reward)))
    if len(PP_meta_reward) > 0:
        print("AVG Meta Reward PP: "+str(sum(PP_meta_reward) / len(PP_meta_reward)))
    if len(HK_meta_reward) > 0:
        print("AVG Meta Reward HK: "+str(sum(HK_meta_reward) / len(HK_meta_reward)))
    if len(SK_meta_reward) > 0:
        print("AVG Meta Reward SK: "+str(sum(SK_meta_reward) / len(SK_meta_reward)))
    if len(DQ_meta_reward) > 0:
        print("AVG Meta Reward DQ: "+str(sum(DQ_meta_reward) / len(DQ_meta_reward)))


    #from os.path import exists
    #if exists(timing_file_name):
    
    #open and read the file after the appending:
    #f = open("demofile2.txt", "r")
    #print(f.read()) 


    #TODO: read everything from file

    # Histogram of cumulative meta level reward
    figure = plt.figure(figsize=(7, 3))
    #bins = np.arange(0.0, 1.01, 0.05)
    plt.hist(PN_meta_reward, alpha=0.5, label='Pure Naive Strategy')
    plt.hist(PG_meta_reward, alpha=0.5, label='Pure Greedy Strategy')
    plt.hist(PP_meta_reward, alpha=0.5, label='Pure Proactive Strategy')
    plt.hist(HK_meta_reward, alpha=0.5, label='Hard kER Strategy')
    plt.hist(DQ_meta_reward, alpha=0.5, label='DQN Strategy')
    plt.xlabel('meta reward earned', fontsize=17)
    plt.ylabel('Frequency', fontsize=17)
    plt.legend(loc='upper left', handletextpad=0.3, columnspacing=0.6, labelspacing=0.15)
    plt.tight_layout()
    plt.margins(x=0.0, y=0.05)
    FILENAME = 'meta.pdf'
    figure.savefig(FILENAME, bbox_inches="tight")
    plt.show()

    # Histogram of cumulative ground reward
    figure = plt.figure(figsize=(7, 3))
    bins = np.arange(0.0, 1.01, 0.05)
    plt.hist(PN_ground_reward, alpha=0.5, label='Pure Naive Strategy')
    plt.hist(PG_ground_reward, alpha=0.5, label='Pure Greedy Strategy')
    plt.hist(PP_ground_reward, alpha=0.5, label='Pure Proactive Strategy')
    plt.hist(HK_ground_reward, alpha=0.5, label='Hard kER Strategy')
    plt.hist(DQ_ground_reward, alpha=0.5, label='DQN Strategy')
    plt.xlabel('relative ground reward earned', fontsize=17)
    plt.ylabel('Frequency', fontsize=17)
    plt.legend(loc='upper left', handletextpad=0.3, columnspacing=0.6, labelspacing=0.15)
    plt.tight_layout()
    plt.margins(x=0.0, y=0.05)
    FILENAME = 'rewards.pdf'
    figure.savefig(FILENAME, bbox_inches="tight")
    plt.show()

    # Histogram of cumulative solve "times"
    figure = plt.figure(figsize=(7, 3))
    #bins = np.arange(0.4, 1.01, 0.01)
    plt.hist(PN_time, alpha=0.5, label='Pure Naive Strategy')
    plt.hist(PG_time, alpha=0.5, label='Pure Greedy Strategy')
    plt.hist(PP_time, alpha=0.5, label='Pure Proactive Strategy')
    plt.hist(HK_time, alpha=0.5, label='Hard kER Strategy')
    plt.hist(DQ_time, alpha=0.5, label='DQN Strategy')
    plt.hist(GG_time, alpha=0.5, label='Exact Solver')
    plt.xlabel('times', fontsize=17)
    plt.ylabel('Frequency', fontsize=17)
    plt.legend(loc='upper left', handletextpad=0.3, columnspacing=0.6, labelspacing=0.15)
    plt.tight_layout()
    plt.margins(x=0.0, y=0.05)
    FILENAME = 'times.pdf'
    figure.savefig(FILENAME, bbox_inches="tight")
    plt.show()

    
    #TODO: make the graphs look pretty


if __name__ == '__main__':
    main()
