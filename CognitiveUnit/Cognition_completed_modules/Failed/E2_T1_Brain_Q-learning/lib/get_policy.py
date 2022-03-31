import numpy as np


def get_policy(Q_table, state, epsilon):

    # 9 possible actions to choose from
    actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
               [-1, 1], [-1, -1], [1, -1], [0, 0]]

    # generate random number for exploration
    number = np.random.random_sample()
    if number <= epsilon:
        action_idx = np.random.choice(len(actions))
    else:
        action_idx = np.argmax(Q_table[state[0], state[1], :])

    action = actions[action_idx]

    return action
