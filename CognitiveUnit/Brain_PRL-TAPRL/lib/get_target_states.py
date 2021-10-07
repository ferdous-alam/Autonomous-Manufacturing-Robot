import numpy as np
from environment.PnCMfg import PnCMfg
from data import *


def remove_duplicate_states(target_states_orig,
                            target_rewards_orig):
    target_states = []
    target_rewards = []
    for idx, s in enumerate(target_states_orig):
        if s not in target_states:
            target_states.append(s)
            target_rewards.append(target_rewards_orig[idx])

    return target_states, target_rewards


def get_target_states(options, option_states, all_visited_target_states,
                      all_visited_target_rewards):
    target_states = []
    target_rewards = []

    # load target reward function for building environment
    reward_dist = np.load('data/target_reward.npy')
    reward_dist = reward_dist.T    # important !!!!!!!!!

    for k in range(len(options)):
        last_prim_action = options[k][-1]
        state = option_states[k][-2]
        env = PnCMfg('target', reward_dist)
        next_state, reward = env.step(state, last_prim_action)
        target_rewards.append(reward)
        target_states.append(state)
    for j, k in zip(target_states, target_rewards):
        all_visited_target_states.append(j)
        all_visited_target_rewards.append(k)

    # save all target states and rewards that have been visited
    visited_target_states, visited_target_rewards = remove_duplicate_states(
        all_visited_target_states, all_visited_target_rewards)

    return visited_target_states, visited_target_rewards


if __name__ == "__main__":
    options = {0: [[1, 1], [0, 1], [1, 0], [-1, -1]],
                 1: [[1, 1], [1, -1], [1, 1], [1, 0]],
                 2: [[1, 1], [0, 1], [0, 0], [1, 1]],
                 3: [[-1, -1], [1, -1], [0, 0], [0, 0]],
                 4: [[1, 1], [0, 1], [0, 0], [0, 0]]}
    option_states = {0: [[1, 2], [2, 3], [2, 4], [3, 4], [2, 3]],
                 1: [[1, 2], [2, 3], [3, 2], [4, 3], [5, 3]],
                 2: [[1, 2], [2, 3], [2, 4], [2, 4], [3, 5]],
                 3: [[1, 2], [0, 1], [1, 0], [1, 0], [1, 0]],
                 4: [[1, 2], [2, 3], [2, 4], [2, 4], [2, 4]]}

    all_visited_target_states_init = []
    all_visited_target_rewards_init = []

    all_visited_target_states, all_visited_target_rewards = get_target_states(
        options, option_states, all_visited_target_states_init,
        all_visited_target_rewards_init)


