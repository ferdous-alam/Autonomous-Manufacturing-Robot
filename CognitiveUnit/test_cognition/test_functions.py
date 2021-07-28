import numpy as np

state_space_size = 4624
action_space_size = 9
action_space = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                     [-1, 1], [-1, -1], [1, -1], [0, 0]]
x1_len = 68
x2_len = 68
Q_table = np.zeros((x1_len, x2_len, len(action_space)))

reward_history = []
current_state = None
# special case for first iteration
alpha = 0.5
gamma = 0.98
if current_state is None:
    # choose random sample
    x1, x2 = np.random.randint(20, 40), np.random.randint(20, 40)
    next_state = [x1, x2]  # index of PnC sample
else:
    action_idx = np.argmax(Q_table[current_state[0], current_state[1], :])
    action = action_space[action_idx]
    next_state = [current_state[i] + action[i] for i in range(len(current_state))]
    best_action = action

    reward = reward_history[-1]     # last element as reward
    Q_table[current_state[0], current_state[1], action] += alpha * (reward + gamma * Q_table[next_state[0], next_state[1], best_action] - Q_table[current_state[0], current_state[1], action])
