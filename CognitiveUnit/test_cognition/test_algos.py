import numpy as np
from policy import epsilon_greedy_policy


class CognitionAlgorithms:
    def __init__(self):
        self.state_space_size = 4624
        self.action_space_size = 9
        self.action_space = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                             [-1, 1], [-1, -1], [1, -1], [0, 0]]
        self.x1_len = 68
        self.x2_len = 68
        self.Q_table = np.load('Q_hat.npy')
        # ---------- hyperparameters --------
        self.alpha = 0.5
        self.gamma = 0.98
        self.epsilon = 0.05

    def Q_learning_vanilla(self, reward_history, lxy, dia):
        # ----------------------------------
        current_state = [lxy, dia]
        if current_state[0] is None and current_state[1] is None:
            x1, x2 = np.random.randint(0, 67), np.random.randint(0, 67)
            current_state = [x1, x2]
        action = epsilon_greedy_policy(self.Q_table, self.action_space, current_state, self.epsilon)
        next_state = np.copy(current_state)
        next_state[0] = min(max(current_state[0] + action[0], 0), 67)
        next_state[1] = min(max(current_state[1] + action[1], 0), 67)

        return next_state, action

    def update_Q_table(self, current_state, action, next_state, reward_history):
        action_index = self.action_space.index(action)
        best_next_action_index = np.argmax(self.Q_table[next_state[0], next_state[1], :])

        reward = reward_history[-1]     # last element as reward

        self.Q_table[current_state[0], current_state[1], action_index] += self.alpha * (
                reward + self.gamma * self.Q_table[next_state[0], next_state[1], best_next_action_index] -
                self.Q_table[current_state[0], current_state[1], action_index])

        return self.Q_table, reward


