import numpy as np


def get_optimal_policy(Q_table, state):

    # 9 possible actions to choose from
    actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
               [-1, 1], [-1, -1], [1, -1], [0, 0]]

    action_idx = np.argmax(Q_table[state[0], state[1], :])
    action = actions[action_idx]

    return action
