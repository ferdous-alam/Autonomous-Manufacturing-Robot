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

    def find_next_state(self, current_state):
        # ----------------------------------
        action = epsilon_greedy_policy(self.Q_table, self.action_space, current_state, self.epsilon)
        next_state = current_state
        next_state[0] = min(max(current_state[0] + action[0], 0), 67)
        next_state[1] = min(max(current_state[1] + action[1], 0), 67)

        return next_state, action

    def Q_learning_vanilla(self, reward_history, current_state):

        if current_state[0] is None and current_state[1] is None:
            # special case for first iteration
            # choose random sample
            x1, x2 = np.random.randint(0, 67), np.random.randint(0, 67)
            next_state = [x1, x2]  # index of PnC sample
        else:
            next_state, action = self.find_next_state(current_state)

        return next_state

    def update_Q_table(self, reward_history, current_state):
        next_state, action = self.find_next_state(current_state)

        action_index = self.action_space.index(action)
        best_next_action_index = np.argmax(self.Q_table[next_state[0], next_state[1], :])

        reward = reward_history[-1]     # last element as reward

        self.Q_table[current_state[0], current_state[1], action_index] += self.alpha * (
                reward + self.gamma * self.Q_table[next_state[0], next_state[1], best_next_action_index] -
                self.Q_table[current_state[0], current_state[1], action_index])

        return self.Q_table, action, current_state, next_state, reward


if __name__ == "__main__":
    Q_table_before = np.load('Q_hat.npy')
    reward_history = [1000, 2000, 3000, 4000]
    state = [20, 20]
    algo = CognitionAlgorithms()
    next_state = algo.Q_learning_vanilla(reward_history, state)
    Q_table_updated = algo.update_Q_table(reward_history, state)
    print('iteration 1----------')
    print(next_state)
    print(Q_table_before[state[0], state[1], :])
    print(Q_table_updated[state[0], state[1], :])


    print('iteration 2----------')
    reward_history.append(10000)
    state = [20, 20]
    algo = CognitionAlgorithms()
    next_state = algo.Q_learning_vanilla(reward_history, state)
    Q_table_updated = algo.update_Q_table(reward_history, state)
    print(next_state)
    print(Q_table_before[state[0], state[1], :])
    print(Q_table_updated[state[0], state[1], :])



