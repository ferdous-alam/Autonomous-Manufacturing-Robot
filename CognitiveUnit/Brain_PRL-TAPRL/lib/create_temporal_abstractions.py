import numpy as np
from environment.PnCMfg import PnCMfg


def create_temporal_abstraction(Q_table, reward_dist,
                                state_init, H,
                                num_of_options, epsilon, seed=None):
    # random seed
    np.random.seed(seed)

    options = {}
    option_states = {}

    # all possible primitive actions
    actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
               [-1, 1], [-1, -1], [1, -1], [0, 0]]

    for i in range(num_of_options-1):
        state = state_init
        prim_actions = []
        prim_states = []
        for h in range(H-1):
            rand_num = np.random.rand()
            if rand_num <= epsilon:
                action_idx = np.random.randint(9)
            else:
                action_idx = np.argmax(Q_table[state[0], state[1], :])

            # options are created using source environment
            env = PnCMfg('source', reward_dist)

            action = actions[action_idx]
            next_state, reward = env.step(state, action)
            prim_states.append(state)
            prim_actions.append(action)

            # update state
            state = next_state
        prim_states.append(state)
        options[i] = prim_actions
        option_states[i] = prim_states

    # create a completely greedy option
    state = state_init
    prim_actions = []
    prim_states = []
    for h in range(H-1):
        rand_num = np.random.rand()
        action_idx = np.argmax(Q_table[state[0], state[1], :])

        # options are created using source environment
        env = PnCMfg('source', reward_dist)
        action = actions[action_idx]
        next_state, reward = env.step(state, action)
        prim_actions.append(action)
        prim_states.append(state)

        # update state
        state = next_state
    prim_states.append(state)
    options[num_of_options-1] = prim_actions
    option_states[num_of_options-1] = prim_states

    return options, option_states

