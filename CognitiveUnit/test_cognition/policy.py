import numpy as np


def epsilon_greedy_policy(Q_table, action_space, state, epsilon):
    """
    Policy selection from the Q-table, here we use the epsilon-greedy
    method for policy selection, the value of epsilon determines how
    much exploration shou8ld be done, it is also possible to reduce the
    exploration at later time steps by introducing an annealing parameter,
    i.e. epsilon = epsilon/step_size etc.

    :param Q_table: 68 x 68x 9 dimensional array where |state space|=68x68
                    and |action space| = 9
    :param action_space: 2x9 dimensional array, all possible actions at each state
                        action_space = [a0, a1, ...., a8], a0 = [0 1] etc.
    :param state: list of lxy and dia indexes, state = [lxy, dia], lxy --> [0, 67],
                  dia --> [0, 67]
    :param epsilon: exploration factor, a value between 0 and 1 although we would like
                    epsilon to be small, not more than 0.2 usually
    :return: action: action suggested by the epsilon-greedy policy
    """
    number = np.random.random_sample()

    if number <= epsilon:
         action_idx = np.random.choice(len(action_space))
    else:
         action_idx = np.argmax(Q_table[state[0], state[1], :])

    action = action_space[action_idx]

    return action
