import numpy as np
from policy import epsilon_greedy_policy


def params():
    state_space_size = 4624
    action_space_size = 9
    action_space = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                         [-1, 1], [-1, -1], [1, -1], [0, 0]]
    x1_len = 68
    x2_len = 68
    # ---------- hyperparameters --------
    alpha = 0.5
    gamma = 0.98
    epsilon = 0.05

    return alpha, gamma, epsilon, action_space


def implement_policy(reward_history, current_state):
    # ----------------------------------
    Q_table = np.load('Q_table.npy')
    alpha, gamma, epsilon, action_space = params()

    if current_state[0] is None and current_state[1] is None:
        x1, x2 = np.random.randint(0, 67), np.random.randint(0, 67)
        current_state = [x1, x2]
    action = epsilon_greedy_policy(Q_table, action_space, current_state, epsilon)
    next_state = np.copy(current_state)
    next_state[0] = min(max(current_state[0] + action[0], 0), 67)
    next_state[1] = min(max(current_state[1] + action[1], 0), 67)

    return next_state, action


def update_Q_table(current_state, action, next_state, reward_history):
    # load previous Q-table from previous timestep
    Q_table = np.load('Q_table.npy')
    alpha, gamma, epsilon, action_space = params()
    
    action = action.tolist()
    action_index = action_space.index(action)
    best_next_action_index = np.argmax(Q_table[next_state[0], next_state[1], :])

    reward = reward_history[-1]     # last element as reward

    Q_table[current_state[0], current_state[1], action_index] += alpha * (
            reward + gamma * Q_table[next_state[0], next_state[1], best_next_action_index] -
            Q_table[current_state[0], current_state[1], action_index])

    # save updated Q-table
    np.save('Q_table.npy', Q_table)


    return None


