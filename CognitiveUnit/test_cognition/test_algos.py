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

    def Q_learning_vanilla(self, reward_history, current_state):

        # ---------- hyperparameters --------
        alpha = 0.5
        gamma = 0.98
        epsilon = 0.05
        # ----------------------------------

        if current_state is None:
            # special case for first iteration
            # choose random sample
            x1, x2 = np.random.randint(20, 40), np.random.randint(20, 40)
            next_state = [x1, x2]  # index of PnC sample
        else:
            action = epsilon_greedy_policy(self.Q_table, self.action_space, current_state, epsilon)
            next_state = current_state  # initialize next state with current state
            next_state[0] = min(max(current_state[0] + action[0], 0), 67)
            next_state[1] = min(max(current_state[1] + action[1], 0), 67)
            best_action = action

            reward = reward_history[-1]     # last element as reward
            self.Q_table[current_state[0], current_state[1], action] += alpha * (
                    reward + gamma * self.Q_table[next_state[0],
                                             next_state[1], best_action] -
                    self.Q_table[current_state[0], current_state[1], action])

        return next_state


if __name__ == "__main__":
    reward_history = [1, 2, 3, 4]
    state = [20, 20]
    algo = CognitionAlgorithms()
    next_state = algo.Q_learning_vanilla(reward_history, state)
    print(next_state)




